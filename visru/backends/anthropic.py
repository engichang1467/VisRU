"""Anthropic backend: neutral messages in, a normalised turn out.

Three things here are not obvious and are load-bearing:

Streaming. Requests carry up to `max_live_images` full renders — tens of
thousands of input tokens — and a non-streaming call of that size can exceed the
SDK's HTTP timeout. `.stream()` with `.get_final_message()` costs one line and
removes the failure mode.

Replaying assistant turns verbatim. Thinking is on by default on this model
family, and thinking blocks must be handed back unchanged on the same model. The
raw chain of thought is never returned, so a turn rebuilt from its text is not
the same turn. The context manager keeps the original blocks in `Step.raw` for
exactly this.

Caching. The render order is tools, then system, then messages, so the stable
half of every request is the tool schemas and the system prompt. Marking the
system prompt cacheable makes that prefix reusable, and a second breakpoint on
the last message extends it over the history so far — which is also why elision
works in batches rather than per turn: rewriting history every turn would throw
the prefix away each time.
"""

import base64
import io
import os
from typing import Any

from PIL import Image

from visru.agent.context import ToolCall
from visru.backends.base import Response
from visru.config import Config

CACHE = {"type": "ephemeral"}


def _png(image: Image.Image) -> dict[str, Any]:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def _block(block: dict[str, Any]) -> dict[str, Any]:
    """One neutral block to Anthropic's wire shape."""
    kind = block.get("type")
    if kind == "image":
        return _png(block["image"])
    if kind == "tool_result":
        out = {"type": "tool_result", "tool_use_id": block["tool_use_id"],
               "content": [_block(b) for b in block["content"]]}
        if block.get("is_error"):
            # Dropping this teaches the model its bad call succeeded.
            out["is_error"] = True
        return out
    return block  # text and tool_use already match


def to_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for message in messages:
        if message["role"] == "assistant" and message.get("raw") is not None:
            out.append({"role": "assistant", "content": message["raw"]})
            continue
        out.append({"role": message["role"],
                    "content": [_block(b) for b in message["content"]]})
    return out


def _cache_tail(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the end of the history cacheable, so next turn reads it back.

    The system breakpoint alone caches only tools and system; without this every
    turn re-bills the whole history at full input price. The tail is always a
    user message the context just built, so its blocks are plain dicts.
    """
    if messages and messages[-1]["role"] == "user" and messages[-1]["content"]:
        content = list(messages[-1]["content"])
        content[-1] = {**content[-1], "cache_control": CACHE}
        messages[-1] = {**messages[-1], "content": content}
    return messages


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, cfg: Config, client: Any = None):
        self.cfg = cfg
        self.client = client or _client()

    def complete(self, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]]) -> Response:
        with self.client.messages.stream(
            model=self.cfg.model.name,
            max_tokens=self.cfg.model.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": CACHE}],
            tools=tools,
            messages=_cache_tail(to_messages(messages)),
        ) as stream:
            message = stream.get_final_message()
        return _response(message)


def _response(message: Any) -> Response:
    text = "".join(b.text for b in message.content if b.type == "text")
    calls = [ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
             for b in message.content if b.type == "tool_use"]

    usage = {}
    for field in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                  "cache_creation_input_tokens"):
        value = getattr(message.usage, field, None)
        if value:
            usage[field.replace("_tokens", "").replace("_input", "")] = value

    if message.stop_reason == "refusal":
        # A refusal is a real ending, not a crash, and the loop should record it
        # rather than treat an empty turn as the model having finished thinking.
        details = getattr(message, "stop_details", None)
        text = text or (f"[refused: {getattr(details, 'category', 'unspecified')}] "
                        f"{getattr(details, 'explanation', '')}").strip()

    return Response(text=text, calls=calls, usage=usage,
                    stop_reason=message.stop_reason or "", raw=message.content)


def _client() -> Any:
    import anthropic

    # Refuse here, before the loop starts, not as a traceback from the first request.
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise ValueError("ANTHROPIC_API_KEY is not set. Export it (visru does not read "
                         ".env files), or pick another --backend; `visru init --no-model` "
                         "needs no key")

    # An org-scoped key must say which workspace to bill; a workspace-scoped key
    # already knows, and the header is absent. Same rule as evals/baseline.py.
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    return anthropic.Anthropic(
        default_headers={"anthropic-workspace-id": workspace} if workspace else None)
