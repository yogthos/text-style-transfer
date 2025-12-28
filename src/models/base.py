"""Base data models for the style transfer pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class MessageRole(Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dict format for API calls."""
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ValidationResult:
    """Result of input validation."""
    valid: bool
    issues: List[str] = field(default_factory=list)
    recommendation: Optional[str] = None


class InputIssue(Enum):
    """Types of input validation issues."""
    TOO_SHORT = "too_short"
    ALREADY_IN_STYLE = "already_in_style"
    LIST_ONLY = "list_only"
    MALFORMED = "malformed"
