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

from visru.backends.base import Backend, Response

__all__ = ["Backend", "Response", "load"]


def load(cfg, **kwargs) -> Backend:
    """Build the backend named in visru.toml.

    Imported lazily: a replay run must not need the anthropic SDK configured, and
    an Anthropic run must not need whatever the open-model path drags in.
    """
    name = cfg.model.backend
    if name == "anthropic":
        from visru.backends.anthropic import AnthropicBackend

        return AnthropicBackend(cfg, **kwargs)
    if name == "openai_compatible":
        from visru.backends.openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(cfg, **kwargs)
    if name == "replay":
        from visru.backends.replay import ReplayBackend

        return ReplayBackend(cfg, **kwargs)
    raise ValueError(
        f"unknown backend {name!r}; visru ships 'anthropic', 'openai_compatible' "
        "and 'replay'")
