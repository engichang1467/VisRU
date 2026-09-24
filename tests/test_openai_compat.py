"""The open-model path: wire shape out, normalised turn back.

No network. What can actually be wrong here is the translation — tool results
split into their own messages, images hoisted out of a text-only tool message,
malformed arguments surviving as a rejectable call rather than a crash.
"""

import json
from types import SimpleNamespace

from PIL import Image

from visru.agent.context import Context, ToolCall
from visru.backends.openai_compat import (
    OpenAICompatBackend, expected_image_tokens, to_messages, to_tools, _response)
from visru.config import Config, ModelConfig
from visru.tools import ToolResult


def filled_context() -> Context:
    ctx = Context("read the sign", Config())
    call = ToolCall(id="t0", name="view", arguments={"asset_id": "im_a"})
    ctx.add_turn("Looking.", [call])
    ctx.add_output(call, ToolResult(text="artifact=a_0001 scale=2.0",
                                    images=[Image.new("RGB", (56, 56))]))
    return ctx


def test_tool_result_becomes_its_own_message_with_images_behind_it():
    wire = to_messages("sys", filled_context().render())
    roles = [m["role"] for m in wire]
    assert roles == ["system", "user", "assistant", "tool", "user"]

    assistant = wire[2]
    assert assistant["tool_calls"][0]["function"]["name"] == "view"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "asset_id": "im_a"}

    tool = wire[3]
    assert tool["tool_call_id"] == "t0"
    assert "a_0001" in tool["content"]
    # Text only: an image left in here never reaches the model.
    assert isinstance(tool["content"], str)

    assert wire[4]["content"][0] == {"type": "text", "text": "images from t0:"}
    assert wire[4]["content"][1]["type"] == "image_url"
    assert wire[4]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_parallel_calls_keep_tool_messages_together():
    """OpenAI 400s if a user message lands between the tool messages of one turn."""
    ctx = Context("read the sign", Config())
    calls = [ToolCall(id=f"t{i}", name="view", arguments={}) for i in range(2)]
    ctx.add_turn("", calls)
    for call in calls:
        ctx.add_output(call, ToolResult(text="ok", images=[Image.new("RGB", (8, 8))]))
    roles = [m["role"] for m in to_messages("sys", ctx.render())]
    assert roles == ["system", "user", "assistant", "tool", "tool", "user"]


def test_failed_call_says_so_in_the_text():
    ctx = Context("read the sign", Config())
    call = ToolCall(id="t0", name="view", arguments={})
    ctx.add_turn("", [call])
    ctx.add_error(call, "box is outside the asset")
    tool = [m for m in to_messages("sys", ctx.render()) if m["role"] == "tool"][0]
    assert tool["content"].startswith("ERROR:")


def test_schemas_translate():
    tools = to_tools([{"name": "view", "description": "look",
                       "input_schema": {"type": "object", "properties": {}}}])
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_malformed_arguments_arrive_as_a_call_not_a_crash():
    completion = SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="tool_calls",
            message=SimpleNamespace(
                content="", tool_calls=[SimpleNamespace(
                    id="t1", function=SimpleNamespace(name="view",
                                                      arguments="{not json"))]))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))
    response = _response(completion)
    assert response.calls[0].name == "view"
    assert response.calls[0].arguments == {"_malformed": "{not json"}


def test_sampling_params_reach_the_server():
    """Left off the request, vLLM samples at its own defaults and the model loops
    on one sentence until max_tokens. The penalty is what breaks that."""
    sent = {}

    class Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    sent.update(kwargs)
                    return SimpleNamespace(
                        choices=[SimpleNamespace(finish_reason="stop", message=(
                            SimpleNamespace(content="ok", tool_calls=[])))],
                        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    cfg = Config(model=ModelConfig(backend="openai_compatible", max_tokens=1500,
                                    temperature=0.3, top_p=0.7, presence_penalty=1.5))
    OpenAICompatBackend(cfg, client=Client()).complete("sys", [], [])

    assert (sent["max_tokens"], sent["temperature"], sent["top_p"],
            sent["presence_penalty"]) == (1500, 0.3, 0.7, 1.5)


def test_refused_params_are_renamed_or_dropped_and_stay_that_way():
    """OpenAI's reasoning models 400 on max_tokens and on the Qwen sampling knobs."""
    import httpx
    from openai import BadRequestError

    calls = []

    def refuse(param, code):
        return BadRequestError(
            "no", response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
            body={"param": param, "code": code})

    class Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    if "max_tokens" in kwargs:
                        raise refuse("max_tokens", "unsupported_parameter")
                    if "temperature" in kwargs:
                        raise refuse("temperature", "unsupported_value")
                    return SimpleNamespace(
                        choices=[SimpleNamespace(finish_reason="stop", message=(
                            SimpleNamespace(content="ok", tool_calls=[])))],
                        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    cfg = Config(model=ModelConfig(backend="openai_compatible", max_tokens=1500))
    backend = OpenAICompatBackend(cfg, client=Client())
    backend.complete("sys", [], [])
    assert len(calls) == 3
    assert calls[-1]["max_completion_tokens"] == 1500
    assert "max_tokens" not in calls[-1] and "temperature" not in calls[-1]

    backend.complete("sys", [], [])
    assert len(calls) == 4  # learned once, no retries on later turns


def test_visual_token_drift_warns_once(caplog):
    cfg = Config(model=ModelConfig(backend="openai_compatible", patch=28))
    backend = OpenAICompatBackend(cfg, client=object())
    messages = filled_context().render()
    expected = expected_image_tokens(messages, 28)
    assert expected > 0

    charged = SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens_details=SimpleNamespace(image_tokens=expected * 50)))
    with caplog.at_level("WARNING"):
        backend._check_visual_tokens(messages, charged)
        backend._check_visual_tokens(messages, charged)
    assert sum("visual-token drift" in r.message for r in caplog.records) == 1
