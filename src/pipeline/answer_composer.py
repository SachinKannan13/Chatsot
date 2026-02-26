from typing import Any
import pandas as pd
from src.utils.llm_client import get_llm_client
from src.utils.logger import get_logger
from src.utils.text_utils import extract_numbers_from_text

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an enterprise-grade analytical assistant designed to analyze and provide insights from structured company survey datasets.

SYSTEM ARCHITECTURE CONTEXT:

The backend explicitly controls which company dataset is loaded. The user selects a company through a frontend selector (dropdown), and the backend loads and stores that company's dataset in session memory. You do NOT need to detect or infer the company from the user's chat messages.

The backend provides you with the currently active company dataset and the company name. Your role is to analyze that dataset and answer analytical questions accurately.

CORE RULES:

1. COMPANY CONTEXT

* Always assume the backend has already loaded the correct company dataset.
* Never ask the user to specify the company name.
* Never attempt to detect or switch companies based on chat messages.
* Always use only the dataset provided in the current session context.
* If no dataset is provided, respond with: "No company dataset is currently loaded. Please select a company first."

2. DATA AUTHORITY

* The provided dataset is the single source of truth.
* Never fabricate, assume, or hallucinate any data.
* Only use the provided dataset for analysis.
* If information is not available in the dataset, explicitly state that it is not available.

3. ANALYTICAL RESPONSIBILITIES
   You must provide:

* descriptive insights
* comparative insights
* trend analysis
* statistical summaries
* rankings
* strengths and weaknesses
* patterns and anomalies

All insights must be grounded in the dataset.

4. RESPONSE QUALITY REQUIREMENTS

Every analytical response should include:

* Clear summary
* Supporting metrics and values
* Interpretation of meaning
* Key drivers or contributing factors when applicable

Use structured formatting when helpful.

5. DO NOT PERFORM THESE ACTIONS:

Do NOT:

* Ask which company to analyze
* Attempt to load or switch companies
* Refer to other companies unless explicitly present in the dataset
* Make assumptions outside provided data
* Mention backend, session, API, Supabase, or technical implementation

6. HANDLE COMPANY SWITCHING

If the user says something like:

"Load Screens"
"Switch company"
"Use Infosys"

Do NOT attempt to perform the switch yourself.

Instead respond:

"Company switching is controlled by the system selector. Please select the desired company using the company selection menu."

7. HANDLE GENERAL QUESTIONS

If the user asks analytical questions such as:

"Give insights"
"What are the strengths?"
"Which department performs best?"

Provide full analytical insights using the currently loaded dataset.

8. CONTEXT CONTINUITY

Maintain conversational continuity using the same dataset throughout the session unless the backend provides a new dataset.

Never reset context on your own.

9. RESPONSE STYLE

Be precise, analytical, and professional.

Prioritize correctness over verbosity.

Avoid generic responses. Provide specific insights based on actual data.

Your primary role is to transform structured survey data into clear, meaningful, and accurate analytical insights.

You are an analytical engine operating on a fixed dataset provided by the backend."""


async def compose_answer(
    question: str,
    classification: dict[str, Any],
    result_df: pd.DataFrame,
    web_context: str,
    conversation_history: list[dict],
    company_name: str,
) -> str:
    """
    Compose a final LLM answer, then validate grounding.
    Falls back to deterministic answer if grounding fails.
    """
    question_type = classification.get("question_type", "simple")
    prompt = _build_prompt(
        question, classification, result_df, web_context,
        conversation_history, company_name, question_type,
    )

    llm = get_llm_client()
    answer = await llm.complete(
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.1,
    )

    # Grounding validation
    if not _is_grounded(answer, result_df, question_type):
        logger.warning("answer_not_grounded", question_type=question_type)
        return _deterministic_answer(question_type, result_df)

    return answer


def _build_prompt(
    question: str,
    classification: dict,
    result_df: pd.DataFrame,
    web_context: str,
    conversation_history: list[dict],
    company_name: str,
    question_type: str,
) -> str:
    # SQL result as JSON (max 50 rows)
    result_rows = result_df.head(50).to_dict(orient="records")
    result_json = _safe_json(result_rows)

    # Structured fallback summary
    fallback_summary = _build_structured_summary(result_df, question_type)

    # Last 3 conversation exchanges
    history_str = ""
    if conversation_history:
        last_3 = conversation_history[-3:]
        history_lines = []
        for item in last_3:
            history_lines.append(f"User: {item.get('question', '')}")
            history_lines.append(f"Assistant: {item.get('answer', '')[:200]}")
        history_str = "\n".join(history_lines)

    parts = [
        f"Company: {company_name}",
        f"Question type: {question_type}",
        f"Intent: {classification.get('intent', question)}",
        "",
        f"SQL result ({len(result_df)} rows):",
        result_json,
        "",
        f"Summary: {fallback_summary}",
    ]

    if web_context:
        parts.extend(["", web_context])

    if history_str:
        parts.extend(["", "Recent conversation:", history_str])

    parts.extend(["", f"User question: {question}"])

    return "\n".join(parts)


def _safe_json(data: list[dict]) -> str:
    """Convert a list of dicts to a JSON string, truncated if very long."""
    import json
    try:
        raw = json.dumps(data, default=str, ensure_ascii=False)
        if len(raw) > 8000:
            # Truncate to first 30 rows
            raw = json.dumps(data[:30], default=str, ensure_ascii=False)
            raw += "\n... (truncated)"
        return raw
    except Exception:
        return str(data)[:4000]


def _build_structured_summary(result_df: pd.DataFrame, question_type: str) -> str:
    """Build a structured text summary of the result DataFrame."""
    if result_df.empty:
        return "No data returned."

    rows = len(result_df)
    cols = result_df.columns.tolist()

    numeric_cols = result_df.select_dtypes(include=["number"]).columns.tolist()

    if question_type in ("ranking", "aggregation", "calculation"):
        lines = [f"{rows} groups:"]
        for _, row in result_df.head(10).iterrows():
            row_parts = [f"{c}={row[c]}" for c in cols[:4]]
            lines.append("  " + ", ".join(row_parts))
        return "\n".join(lines)

    elif question_type == "count":
        if rows == 1 and len(cols) == 1:
            return f"Count: {result_df.iloc[0, 0]}"
        return f"{rows} rows returned."

    elif question_type == "list":
        values = result_df.iloc[:, 0].tolist()
        return f"Values: {', '.join(str(v) for v in values[:20])}"

    elif numeric_cols:
        stats = result_df[numeric_cols].describe().to_dict()
        return str(stats)[:500]

    return f"{rows} rows, columns: {', '.join(cols[:6])}"


def _is_grounded(answer: str, result_df: pd.DataFrame, question_type: str) -> bool:
    """
    Validate that the LLM answer is grounded in the result data.

    Checks:
    1. At least one concrete string value from result_df appears in the answer
    2. For non-analytical types: numbers in answer appear in result_df
    """
    if result_df.empty:
        return True  # Nothing to ground against

    answer_lower = answer.lower()

    # Check string values
    str_cols = result_df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        for val in result_df[col].dropna().unique()[:30]:
            if isinstance(val, str) and len(val) > 2 and val.lower() in answer_lower:
                return True

    # Check numeric values
    numeric_cols = result_df.select_dtypes(include=["number"]).columns
    answer_numbers = set(extract_numbers_from_text(answer))
    if answer_numbers:
        for col in numeric_cols:
            for val in result_df[col].dropna().unique()[:50]:
                try:
                    rounded = round(float(val), 2)
                    if any(
                        abs(n - rounded) < 0.5 or abs(n - round(float(val), 0)) < 0.5
                        for n in answer_numbers
                    ):
                        return True
                except (ValueError, TypeError):
                    pass

    # If no values matched but answer is non-trivial, allow it for analytical types
    if question_type in ("insights", "analytics", "recommendations"):
        return len(answer) > 100

    logger.debug("grounding_check_failed", answer_preview=answer[:100])
    return False


def _deterministic_answer(
    question_type: str,
    result_df: pd.DataFrame,
) -> str:
    """Build a factual answer directly from result_df without LLM."""
    if result_df.empty:
        return "No data was found matching your query."

    cols = result_df.columns.tolist()
    if question_type in ("ranking", "aggregation"):
        lines = []
        for i, (_, row) in enumerate(result_df.head(10).iterrows(), 1):
            row_str = " | ".join(f"{c}: {_fmt(row[c])}" for c in cols[:4])
            lines.append(f"{i}. {row_str}")
        return "\n".join(lines)

    elif question_type == "count":
        if len(cols) == 1 and len(result_df) == 1:
            return f"The count is: {result_df.iloc[0, 0]}"
        return f"Query returned {len(result_df)} rows."

    elif question_type == "list":
        values = result_df.iloc[:, 0].dropna().tolist()
        return "Values: " + ", ".join(str(v) for v in values[:20])

    elif question_type == "comparison":
        lines = []
        for _, row in result_df.iterrows():
            lines.append(" | ".join(f"{c}: {_fmt(row[c])}" for c in cols))
        return "\n".join(lines)

    else:
        # Generic tabular answer
        lines = [f"Results ({len(result_df)} rows):"]
        for _, row in result_df.head(15).iterrows():
            lines.append("  " + ", ".join(f"{c}: {_fmt(row[c])}" for c in cols[:5]))
        return "\n".join(lines)


def _fmt(val) -> str:
    """Format a value for display."""
    if val is None:
        return "N/A"
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return f"{f:.2f}"
    except (ValueError, TypeError):
        return str(val)
