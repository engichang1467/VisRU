"""OpenAI-compatible backend: the open-model path.

The same adapter serves a hosted router and a local vLLM server, because both
speak chat completions. Only `base_url` changes.

Three things here are not obvious and are load-bearing:

Tool results are their own messages. Anthropic nests a `tool_result` inside a
user turn; OpenAI wants `role="tool"` messages that follow the assistant turn
that asked for them, and their content is text only. So a tool result carrying
images is split: the text goes in the tool message, and the images follow in a
user message. Dropping that split is how images silently stop reaching the model.

Provider pinning belongs in the model name. The router routes `Qwen/...:novita`
to one provider, and `supports_tools` differs between providers for the same
weights. An unpinned name is a different experiment every run.

Visual tokens are checked, not assumed. `geometry.fit_render_size` sizes every
render so the provider will not resize it again; a provider that tiles on its own
terms breaks the coordinate contract the whole evidence ledger rests on. The
provider reports how many image tokens it actually charged, so the two numbers
can be compared and the drift logged.
"""

import base64
import io
import json
import logging
import math
import os
from typing import Any

from PIL import Image

from visru.agent.context import ToolCall
from visru.backends import Response
from visru.config import Config

log = logging.getLogger(__name__)

# ponytail: patch-grid estimate with the usual 2x2 merge. Exact only for
# Qwen-family vision towers; it is a drift alarm, not an accounting record.
MERGE = 4
DRIFT = 2.0  # factor away from the estimate before we say something


# ponytail: JPEG, not PNG. A 2000px render is megabytes as PNG and routers reject
# the request outright (413) once a few of them are live. Quality 90 costs nothing
# visible on photographs; if a mask or an overlay ever reads wrong here, that is
# where to look first.
JPEG_QUALITY = 90


def _data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.standard_b64encode(buf.getvalue()).decode()


def expected_image_tokens(messages: list[dict[str, Any]], patch: int) -> int:
    """What the render sizes imply the provider should charge."""
    total = 0
    for message in messages:
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                w, h = block["image"].size
                total += math.ceil(w / patch) * math.ceil(h / patch) // MERGE
            elif block.get("type") == "tool_result":
                total += expected_image_tokens(
                    [{"content": block.get("content") or []}], patch)
    return total


def _parts(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Neutral text and image blocks to OpenAI content parts."""
    parts = []
    for block in blocks:
        if block.get("type") == "image":
            parts.append({"type": "image_url",
                          "image_url": {"url": _data_url(block["image"])}})
        elif block.get("type") == "text":
            parts.append({"type": "text", "text": block["text"]})
    return parts


def _assistant(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    text = "".join(b["text"] for b in blocks if b.get("type") == "text")
    calls = [{"id": b["id"], "type": "function",
              "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
             for b in blocks if b.get("type") == "tool_use"]
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = calls
    return message


def _tool_result(block: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One tool_result to a tool message, plus the image parts that ride behind it.

    A `role="tool"` message is text only, so images go in a user message after
    it. The error flag goes into the text: without it the model reads a failed
    call as having worked, and OpenAI has no `is_error` field to carry it.
    """
    inner = block.get("content") or []
    text = "\n".join(b["text"] for b in inner if b.get("type") == "text")
    if block.get("is_error"):
        text = f"ERROR: {text}" if text else "ERROR"
    message = {"role": "tool", "tool_call_id": block["tool_use_id"],
               "content": text or "(no output)"}
    images = _parts([b for b in inner if b.get("type") == "image"])
    if images:  # parallel calls share one user message; say whose images these are
        images.insert(0, {"type": "text", "text": f"images from {block['tool_use_id']}:"})
    return message, images


def to_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        content = message.get("content") or []
        if message["role"] == "assistant":
            raw = message.get("raw")
            out.append(raw if isinstance(raw, dict) else _assistant(content))
            continue
        # Every tool message must directly follow the assistant turn that called
        # it, so all of them go first and the images collect into one user message.
        images: list[dict[str, Any]] = []
        for block in content:
            if block.get("type") == "tool_result":
                tool_message, parts = _tool_result(block)
                out.append(tool_message)
                images.extend(parts)
        rest = images + _parts([b for b in content if b.get("type") != "tool_result"])
        if rest:
            out.append({"role": "user", "content": rest})
    return out


def to_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """visru tool schemas to OpenAI function schemas."""
    return [{"type": "function",
             "function": {"name": t["name"], "description": t.get("description", ""),
                          "parameters": t["input_schema"]}}
            for t in tools]


def _token() -> str:
    for name in ("VISRU_API_KEY", "HF_TOKEN", "OPENAI_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    try:  # the HF router's usual home, written by `hf auth login`
        from huggingface_hub import get_token

        token = get_token()
    except Exception:  # noqa: BLE001
        token = None
    if token:
        return token
    # vLLM ignores the key but the SDK insists on one.
    return "not-needed"


class OpenAICompatBackend:
    """Any chat-completions endpoint: the HF router, vLLM, anything OpenAI-shaped."""

    name = "openai_compatible"

    def __init__(self, cfg: Config, client: Any = None):
        self.cfg = cfg
        self.warned = False
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(base_url=cfg.model.base_url, api_key=_token())
        # Sampling knobs tuned for vLLM/Qwen. OpenAI's reasoning models refuse some
        # of them, so complete() prunes whatever the endpoint rejects, once.
        self.params: dict[str, Any] = {
            "max_tokens": cfg.model.max_tokens,
            "temperature": cfg.model.temperature,
            "top_p": cfg.model.top_p,
            "presence_penalty": cfg.model.presence_penalty,
        }

    def complete(self, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]]) -> Response:
        from openai import BadRequestError

        wire = to_messages(system, messages)
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.cfg.model.name,
                    tools=to_tools(tools) or None,
                    messages=wire,
                    **self.params,
                )
                break
            except BadRequestError as e:
                if not self._adapt(e):
                    raise
        self._check_visual_tokens(messages, completion)
        return _response(completion)

    def _adapt(self, error: Any) -> bool:
        """Drop or rename the param the endpoint refused; False if nothing to fix."""
        param = getattr(error, "param", None)
        if getattr(error, "code", None) not in ("unsupported_parameter",
                                                  "unsupported_value"):
            return False
        if param not in self.params:
            return False
        value = self.params.pop(param)
        if param == "max_tokens":
            self.params["max_completion_tokens"] = value
        log.warning("endpoint refused %r; %s it for this session", param,
                    "renamed" if param == "max_tokens" else "dropped")
        return True

    def _check_visual_tokens(self, messages: list[dict[str, Any]],
                             completion: Any) -> None:
        """Warn when the provider charged a wildly different number of image tokens.

        A provider that resizes on its own terms invalidates every coordinate the
        model emits, which is exactly the failure the evidence ledger cannot see.
        """
        expected = expected_image_tokens(messages, self.cfg.model.patch)
        details = getattr(getattr(completion, "usage", None),
                          "prompt_tokens_details", None)
        charged = getattr(details, "image_tokens", None)
        if not expected or not charged or self.warned:
            return
        ratio = charged / expected
        if ratio > DRIFT or ratio < 1 / DRIFT:
            self.warned = True  # once per session; the loop would repeat it every turn
            log.warning(
                "visual-token drift: sent renders worth ~%d image tokens, provider "
                "charged %d (%.2fx). The provider is likely resizing or tiling on its "
                "own terms, so model coordinates may not be source-asset pixels.",
                expected, charged, ratio)


def _response(completion: Any) -> Response:
    choice = completion.choices[0]
    message = choice.message
    calls = []
    for call in message.tool_calls or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            # A malformed call is the open-model failure mode worth measuring, so
            # it reaches the loop as a call the tool layer will reject, not as a
            # crash here.
            arguments = {"_malformed": call.function.arguments}
        calls.append(ToolCall(id=call.id, name=call.function.name,
                              arguments=arguments))

    usage = {}
    raw_usage = getattr(completion, "usage", None)
    for field, key in (("prompt_tokens", "input"), ("completion_tokens", "output")):
        value = getattr(raw_usage, field, None)
        if value:
            usage[key] = value
    details = getattr(raw_usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None)
    if cached:
        usage["cache_read"] = cached

    return Response(text=message.content or "", calls=calls, usage=usage,
                    stop_reason=choice.finish_reason or "",
                    raw=message.model_dump(exclude_none=True)
                    if hasattr(message, "model_dump") else None)
