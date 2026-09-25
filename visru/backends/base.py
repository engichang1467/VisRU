"""The backend contract, apart from `load`, so each backend can import it without
importing the package that lazily imports each backend."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from visru.agent.context import ToolCall


@dataclass
class Response:
    """One model turn, normalised."""

    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = ""
    # The provider's own content blocks, to be handed back verbatim next turn.
    # Thinking blocks must be replayed unchanged on the same model, and they
    # cannot be rebuilt from `text` — the raw reasoning is never returned.
    raw: Any = None


class Backend(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]]) -> Response:
        ...
