import functools
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

SESSION_TIMEOUT_HOURS = 2
MAX_HISTORY = 10  # increased from 5 to 10


@dataclass
class ConversationTurn:
    question: str
    answer: str
    question_type: str
    is_error: bool = False          # True when the answer was an error/failure
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SessionState:
    session_id: str
    company_name: Optional[str] = None
    awaiting_company: bool = False
    conversation_history: list[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.last_active + timedelta(hours=SESSION_TIMEOUT_HOURS)

    def touch(self) -> None:
        self.last_active = datetime.utcnow()

    def add_turn(
        self,
        question: str,
        answer: str,
        question_type: str,
        is_error: bool = False,
    ) -> None:
        self.conversation_history.append(
            ConversationTurn(
                question=question,
                answer=answer,
                question_type=question_type,
                is_error=is_error,
            )
        )
        if len(self.conversation_history) > MAX_HISTORY:
            self.conversation_history = self.conversation_history[-MAX_HISTORY:]

    def get_history_dicts(self) -> list[dict]:
        """All turns including errors — used for full context."""
        return [
            {
                "question": t.question,
                "answer": t.answer,
                "question_type": t.question_type,
                "is_error": t.is_error,
            }
            for t in self.conversation_history
        ]

    def get_successful_history_dicts(self) -> list[dict]:
        """
        Only turns that produced a real answer (no errors).
        Used for follow-up resolution so errors don't break context.
        """
        return [
            {
                "question": t.question,
                "answer": t.answer,
                "question_type": t.question_type,
            }
            for t in self.conversation_history
            if not t.is_error
        ]

    def switch_company(self, new_company: str) -> None:
        """Reset history and state when user switches to a new company."""
        old = self.company_name
        self.company_name = new_company
        self.conversation_history = []
        self.awaiting_company = False
        logger.info(
            "session_company_switch",
            session=self.session_id,
            from_company=old,
            to_company=new_company,
        )


class SessionManager:
    """In-memory session store with TTL expiry."""

    def __init__(self):
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        session = self._sessions.get(session_id)
        if session is None or session.is_expired():
            if session and session.is_expired():
                logger.info("session_expired", session_id=session_id)
            session = SessionState(session_id=session_id)
            self._sessions[session_id] = session
            logger.info("session_created", session_id=session_id)
        session.touch()
        return session

    def reset(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
        logger.info("session_reset", session_id=session_id)

    def active_count(self) -> int:
        return sum(1 for s in self._sessions.values() if not s.is_expired())


@functools.cache
def get_session_manager() -> SessionManager:
    return SessionManager()