"""Conversation memory for multi-turn RAG interactions."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import logging

LOGGER = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[str] = field(default_factory=list)  # Document sources used
    confidence: float = 1.0


class ConversationMemory:
    """
    Manages conversation history for context-aware responses.
    
    Features:
    - Sliding window of recent messages
    - Context summarization for long conversations
    - Source tracking across turns
    """

    def __init__(self, max_messages: int = 10, max_tokens: int = 2000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.messages: List[Message] = []
        self._summary: Optional[str] = None

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))
        self._trim()

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[str]] = None,
        confidence: float = 1.0,
    ) -> None:
        """Add an assistant response to history."""
        self.messages.append(
            Message(
                role="assistant",
                content=content,
                sources=sources or [],
                confidence=confidence,
            )
        )
        self._trim()

    def _trim(self) -> None:
        """Keep only recent messages within limits."""
        if len(self.messages) > self.max_messages:
            # Keep first message for context and recent ones
            self.messages = self.messages[-self.max_messages:]

    def get_context_string(self, max_chars: int = 2000) -> str:
        """Get conversation history as a formatted string."""
        if not self.messages:
            return ""

        parts = []
        total_chars = 0

        # Work backwards from most recent
        for msg in reversed(self.messages):
            role = "User" if msg.role == "user" else "Assistant"
            text = f"{role}: {msg.content}"
            
            if total_chars + len(text) > max_chars:
                break
            
            parts.insert(0, text)
            total_chars += len(text)

        return "\n".join(parts)

    def get_recent_sources(self, n: int = 5) -> List[str]:
        """Get sources from recent assistant messages."""
        sources = []
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.sources:
                sources.extend(msg.sources)
                if len(sources) >= n:
                    break
        return list(dict.fromkeys(sources))[:n]  # Dedupe, keep order

    def get_last_user_query(self) -> Optional[str]:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages = []
        self._summary = None
        LOGGER.debug("Conversation memory cleared")

    def to_openai_messages(self, system_prompt: str = "") -> List[dict]:
        """Convert to OpenAI chat format."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        for msg in self.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        
        return messages

    def __len__(self) -> int:
        return len(self.messages)

    def __bool__(self) -> bool:
        return bool(self.messages)

