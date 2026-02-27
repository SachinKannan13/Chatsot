"""
Main chat agent orchestrating analytics on the currently active company dataset.
"""
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.agent.session_manager import get_session_manager, SessionState
from src.cache.company_cache import get_company_cache
from src.cache.query_cache import get_query_cache
from src.data.data_cleaner import clean_dataframe
from src.data.schema_analyzer import analyze_schema
from src.data.supabase_loader import get_supabase_loader
from src.pipeline.answer_composer import compose_answer
from src.pipeline.question_classifier import classify_question
from src.pipeline.sql_executor import SQLExecutionError, build_empty_hint, execute_multiple, merge_results
from src.pipeline.sql_generator import generate_sql
from src.pipeline.web_researcher import format_web_context, research_topic
from src.pipeline.query_preprocessor import correct_query, resolve_followup
from src.utils.logger import get_logger
from src.utils.text_utils import sanitize_company_name

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are an enterprise-grade analytical assistant designed to analyze and provide insights from structured company survey datasets.

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

You are an analytical engine operating on a fixed dataset provided by the backend.
"""

NO_DATASET_MESSAGE = "No company dataset is currently loaded. Please select a company first."
SWITCH_CONTROLLED_MESSAGE = (
    "Company switching is controlled by the system selector. "
    "Please select the desired company using the company selection menu."
)
_SWITCH_KEYWORDS = ("load ", "switch", "change", "use ", "set ", "fetch ", "show data for", "give data for")
sessions: dict[str, dict[str, Any]] = {}


def get_session(session_id: str) -> dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "company": None,
            "dataset": None,
            "loaded": False,
        }
    return sessions[session_id]


@dataclass
class ChatResponse:
    session_id: str
    response: str
    question_type: str
    company: Optional[str]
    from_cache: bool
    pipeline_steps: list[str]
    error: Optional[str] = None


async def set_active_company(session_id: str, company_name: str) -> ChatResponse:
    """
    Backend-controlled company selection entry point.
    Loads and validates dataset for the selected company.
    """
    session = get_session_manager().get_or_create(session_id)
    session_store = get_session(session_id)
    selected = sanitize_company_name(company_name)

    if session.company_name != selected:
        session.switch_company(selected)
        get_query_cache().clear_session(session_id)

    df, metadata = await _ensure_company_data(session)
    if df is None:
        session_store["company"] = None
        session_store["dataset"] = None
        session_store["loaded"] = False
        return ChatResponse(
            session_id=session_id,
            response=metadata.get("error", "Failed to load company data."),
            question_type="error",
            company=session.company_name,
            from_cache=False,
            pipeline_steps=["set_company"],
            error=metadata.get("error"),
        )

    session_store["company"] = selected
    session_store["dataset"] = df.to_dict(orient="records")
    session_store["loaded"] = True

    return ChatResponse(
        session_id=session_id,
        response=f"{company_name} data loaded successfully.",
        question_type="company_set",
        company=session.company_name,
        from_cache=False,
        pipeline_steps=["set_company"],
    )


async def handle_message(session_id: str, message: str) -> ChatResponse:
    """
    Process an analytical question against the active company dataset.
    """
    steps: list[str] = []
    original_message = message  # preserve original before any preprocessing
    session = get_session_manager().get_or_create(session_id)
    session_store = get_session(session_id)

    # Keep SessionManager state aligned with lightweight session storage.
    if not session.company_name and session_store.get("company"):
        session.company_name = str(session_store["company"])
    if not session_store.get("loaded") and session.company_name:
        session_store["loaded"] = True
        session_store["company"] = session.company_name

    if _is_company_switch_message(message):
        return ChatResponse(
            session_id=session_id,
            response=SWITCH_CONTROLLED_MESSAGE,
            question_type="switch_controlled",
            company=session.company_name,
            from_cache=False,
            pipeline_steps=["switch_controlled"],
        )

    if not session.company_name or not session_store.get("loaded"):
        return ChatResponse(
            session_id=session_id,
            response=NO_DATASET_MESSAGE,
            question_type="no_dataset",
            company=None,
            from_cache=False,
            pipeline_steps=["no_dataset"],
        )

    steps.append("load_data")
    df, metadata = await _ensure_company_data(session)
    if df is None:
        return ChatResponse(
            session_id=session_id,
            response=metadata.get("error", "Failed to load company data."),
            question_type="error",
            company=session.company_name,
            from_cache=False,
            pipeline_steps=steps,
            error=metadata.get("error"),
        )

    # ── Preprocess: correct typos then resolve follow-ups ────────────────────
    steps.append("correct_query")
    message, was_corrected = await correct_query(message)
    if was_corrected:
        logger.info("query_was_corrected", original=original_message, corrected=message)

    steps.append("resolve_followup")
    successful_history = session.get_successful_history_dicts()
    message, was_resolved = await resolve_followup(message, successful_history)
    if was_resolved:
        logger.info("query_was_resolved", resolved=message)

    steps.append("cache_check")
    query_cache = get_query_cache()
    cached = query_cache.get(session_id, message)
    if cached:
        _, cached_response = cached
        return ChatResponse(
            session_id=session_id,
            response=cached_response,
            question_type="cached",
            company=session.company_name,
            from_cache=True,
            pipeline_steps=steps + ["cache_hit"],
        )

    steps.append("classify")
    try:
        classification = await classify_question(message, metadata)
    except Exception as e:
        logger.error("classify_error", error=str(e))
        classification = {
            "question_type": "simple",
            "intent": message,
            "requires_web_search": False,
            "aggregation": None,
            "dimensions": [],
            "metric": None,
            "filters": [],
            "complexity": "simple",
        }

    question_type = classification.get("question_type", "simple")

    steps.append("sql_generate")
    result_df = pd.DataFrame()
    hint_msg = ""

    try:
        sql_queries = await generate_sql(message, classification, metadata, df)

        steps.append("sql_execute")
        result_dfs = execute_multiple(sql_queries, df)
        result_df = merge_results(result_dfs)

        if result_df.empty and sql_queries:
            hint_msg = build_empty_hint(sql_queries[0], df, classification, metadata)

    except SQLExecutionError as e:
        logger.error("sql_execution_failed", error=str(e))
        hint_msg = str(e)
    except Exception as e:
        logger.error("sql_pipeline_error", error=str(e))
        hint_msg = "I encountered an issue processing your query."

    web_context = ""
    if classification.get("requires_web_search") and session.company_name:
        steps.append("web_research")
        try:
            web_results = await research_topic(
                company_name=session.company_name,
                question=message,
                question_type=question_type,
                classification=classification,
            )
            web_context = format_web_context(web_results)
        except Exception as e:
            logger.warning("web_research_failed", error=str(e))

    steps.append("compose")
    if result_df.empty and hint_msg:
        response = hint_msg
    else:
        try:
            response = await compose_answer(
                question=message,
                classification=classification,
                result_df=result_df,
                web_context=web_context,
                conversation_history=session.get_history_dicts(),
                company_name=session.company_name or "",
            )
        except Exception as e:
            logger.error("compose_answer_error", error=str(e))
            response = "I'm sorry, I encountered an error composing the answer. Please try rephrasing your question."

    if not result_df.empty:
        query_cache.set(
            session_id=session_id,
            question=message,
            result_df=result_df,
            response=response,
        )

    # Flag turn as error if result was empty+hint or compose failed
    is_error_turn = (result_df.empty and bool(hint_msg)) or question_type == "error"
    session.add_turn(
        question=original_message,   # store original so history stays readable
        answer=response,
        question_type=question_type,
        is_error=is_error_turn,
    )

    return ChatResponse(
        session_id=session_id,
        response=response,
        question_type=question_type,
        company=session.company_name,
        from_cache=False,
        pipeline_steps=steps,
    )


async def _ensure_company_data(session: SessionState) -> tuple[Optional[pd.DataFrame], dict]:
    """
    Return (df, metadata) from cache or fresh fetch + analyze.
    On error returns (None, {"error": "..."}).
    """
    company_name = session.company_name
    if not company_name:
        return None, {"error": NO_DATASET_MESSAGE}

    company_cache = get_company_cache()
    cached = company_cache.get(company_name)
    if cached:
        return cached

    loader = get_supabase_loader()
    try:
        raw_df = loader.fetch_company_data(company_name)
    except ValueError as e:
        return None, {"error": str(e)}
    except Exception as e:
        logger.error("supabase_fetch_error", company=company_name, error=str(e))
        return None, {"error": f"Failed to load data for '{company_name}': {e}"}

    df = clean_dataframe(raw_df)
    metadata = await analyze_schema(df)
    company_cache.set(company_name, df, metadata)
    return df, metadata


def _is_company_switch_message(message: str) -> bool:
    msg = message.lower().strip()
    if not msg:
        return False
    return any(k in msg for k in _SWITCH_KEYWORDS)


def fetch_company_data(company: str) -> list[dict[str, Any]]:
    """
    Compatibility helper for session-dataset loaders.
    Returns list[dict] records for the requested company table.
    """
    selected = sanitize_company_name(company)
    df = get_supabase_loader().fetch_company_data(selected)
    if df.empty:
        raise ValueError(f"No data found for company {company}")
    return df.to_dict(orient="records")


def generate_llm_response(message: str, dataset: list[dict[str, Any]]) -> str:
    """
    Compatibility helper signature for fixed-dataset chat integrations.
    Existing production flow uses handle_message() with full SQL+LLM pipeline.
    """
    if not dataset:
        return NO_DATASET_MESSAGE
    return "Dataset is loaded. Please use the /chat analytical flow for full insights."