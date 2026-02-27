"""
Query Preprocessor — runs before classification and SQL generation.

Two responsibilities:
1. correct_query()     — Fix typos, wrong words, grammar without changing intent.
2. resolve_followup()  — If the query is a follow-up to a prior successful turn,
                          rewrite it as a fully self-contained question so the
                          SQL generator never needs to guess at missing context.
"""
from __future__ import annotations

import re
from typing import Optional

from src.utils.llm_client import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Correction ─────────────────────────────────────────────────────────────────

_CORRECTION_SYSTEM = (
    "You are a query correction assistant for a business analytics chatbot. "
    "Your only job is to fix typos, spelling mistakes, and grammatical errors "
    "in the user's question while keeping the original intent and meaning completely intact. "
    "Do NOT add, remove, or change any analytical intent. "
    "Do NOT answer the question. "
    "Return ONLY the corrected question as plain text. "
    "If the question is already correct, return it unchanged."
)


async def correct_query(message: str) -> tuple[str, bool]:
    """
    Fix typos and grammar in the user's message.

    Returns:
        (corrected_message, was_changed): corrected text and whether it differed.
    """
    # Skip very short messages or greetings — not worth an LLM call.
    if len(message.strip()) < 6:
        return message, False

    try:
        llm = get_llm_client()
        corrected = await llm.complete(
            system=_CORRECTION_SYSTEM,
            messages=[{"role": "user", "content": message}],
            max_tokens=300,
            temperature=0.0,
        )
        corrected = corrected.strip()

        # Sanity: if the LLM returned something wildly different in length, skip it.
        if not corrected or len(corrected) > len(message) * 3:
            return message, False

        was_changed = _meaningful_change(message, corrected)
        if was_changed:
            logger.info("query_corrected", original=message, corrected=corrected)
        return corrected, was_changed

    except Exception as e:
        logger.warning("query_correction_failed", error=str(e))
        return message, False


def _meaningful_change(original: str, corrected: str) -> bool:
    """Return True only if the correction changed something beyond casing/spaces."""
    a = re.sub(r"\s+", " ", original.strip().lower())
    b = re.sub(r"\s+", " ", corrected.strip().lower())
    return a != b


# ── Follow-up Resolution ───────────────────────────────────────────────────────

_FOLLOWUP_SYSTEM = (
    "You are a context resolution assistant for a business analytics chatbot. "
    "You receive a conversation history of successful Q&A pairs and a new user question. "
    "\n\n"
    "Your job: decide if the new question is a follow-up to a previous question. "
    "If it IS a follow-up, rewrite it as a fully self-contained standalone question "
    "that includes all necessary context from the prior conversation. "
    "If it is NOT a follow-up (it's a new independent question), return it unchanged. "
    "\n\n"
    "Rules:\n"
    "- NEVER answer the question. Only rewrite or return as-is.\n"
    "- Do NOT add analytical interpretation.\n"
    "- Keep domain terms (department, grade, score, etc.) exactly as used.\n"
    "- Return ONLY the final question as plain text, nothing else.\n"
    "\n"
    "Signs that a question IS a follow-up:\n"
    "- Uses pronouns: 'they', 'those', 'it', 'them', 'that department'\n"
    "- Uses 'instead', 'also', 'what about', 'and', 'but what'\n"
    "- Refers back: 'the same', 'the ones', 'those results'\n"
    "- Partial flip: 'now show bottom', 'reverse that', 'top 5 instead'\n"
    "- Very short and context-dependent: 'why?', 'and engineering?', 'break it down'\n"
)


async def resolve_followup(
    message: str,
    successful_history: list[dict],
) -> tuple[str, bool]:
    """
    If the message is a follow-up to a prior turn, rewrite it as a standalone question.

    Args:
        message: The current user question.
        successful_history: List of {'question': ..., 'answer': ...} dicts,
                            only successful (non-error) turns, most recent last.

    Returns:
        (resolved_message, was_resolved): the final question and whether it changed.
    """
    if not successful_history:
        return message, False

    # Only use last 5 successful turns for context (avoid huge prompts).
    recent = successful_history[-5:]

    history_text = _format_history(recent)
    prompt = (
        f"Conversation history (successful exchanges only):\n"
        f"{history_text}\n\n"
        f"New user question: {message}\n\n"
        f"Return the standalone question:"
    )

    try:
        llm = get_llm_client()
        resolved = await llm.complete(
            system=_FOLLOWUP_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        resolved = resolved.strip()

        if not resolved or len(resolved) > len(message) * 4:
            return message, False

        was_resolved = _meaningful_change(message, resolved)
        if was_resolved:
            logger.info(
                "followup_resolved",
                original=message,
                resolved=resolved,
                history_turns=len(recent),
            )
        return resolved, was_resolved

    except Exception as e:
        logger.warning("followup_resolution_failed", error=str(e))
        return message, False


def _format_history(history: list[dict]) -> str:
    lines = []
    for i, turn in enumerate(history, 1):
        q = turn.get("question", "")
        a = turn.get("answer", "")[:300]  # truncate long answers
        lines.append(f"Turn {i}:")
        lines.append(f"  User: {q}")
        lines.append(f"  Assistant: {a}")
    return "\n".join(lines)