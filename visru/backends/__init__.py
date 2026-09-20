"""What the loop needs from a model, and nothing more.

A backend takes the context manager's provider-neutral messages and returns a
`Response`. That is the whole contract. Keeping it this small is what lets the
same loop drive Anthropic, a canned replay for testing, and later an open model
behind vLLM, without any of them leaking into the harness.

Blocks arriving here are the ones `agent.context.Context.render` produces:
`{"type": "text"}`, `{"type": "image", "image": <PIL>}`, `{"type": "tool_use"}`
and `{"type": "tool_result", "content": [...]}`. Encoding an image is the
backend's job, because only it knows what its API wants.
"""

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


class Backend(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]]) -> Response:
        ...


def load(cfg, **kwargs) -> Backend:
    """Build the backend named in visru.toml.

    Imported lazily: a replay run must not need the anthropic SDK configured, and
    an Anthropic run must not need whatever the open-model path drags in.
    """
    name = cfg.model.backend
    if name == "anthropic":
        from visru.backends.anthropic import AnthropicBackend

        return AnthropicBackend(cfg, **kwargs)
    if name == "replay":
        from visru.backends.replay import ReplayBackend

        return ReplayBackend(cfg, **kwargs)
    raise ValueError(
        f"unknown backend {name!r}; visru ships 'anthropic' and 'replay' "
        "('openai_compatible' arrives with the open-model work in Phase 5)")
