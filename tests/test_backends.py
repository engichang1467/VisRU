"""Backends: the wire shape going out, and the normalised turn coming back.

The Anthropic tests use a fake client rather than the network. What is being
checked is the translation — images encoded, tool results paired, errors flagged,
thinking replayed — and none of that needs a live model to be wrong.
"""

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from visru.agent.context import Context, ToolCall
from visru.backends import Response, load
from visru.backends.anthropic import AnthropicBackend, to_messages
from visru.backends.replay import ReplayBackend, sweep
from visru.config import Config, ModelConfig
from visru.tools import ToolResult


# --- a fake Anthropic client --------------------------------------------
class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeMessages:
    def __init__(self, message):
        self.message = message
        self.seen: dict = {}

    def stream(self, **kwargs):
        self.seen = kwargs
        return FakeStream(self.message)


class FakeClient:
    def __init__(self, message):
        self.messages = FakeMessages(message)


def block(**kw):
    return SimpleNamespace(**kw)


def reply(text="ok", calls=(), stop_reason="end_turn", **usage):
    content = [block(type="text", text=text)]
    content += [block(type="tool_use", id=c[0], name=c[1], input=c[2]) for c in calls]
    return SimpleNamespace(
        content=content, stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=usage.get("input", 10),
                              output_tokens=usage.get("output", 2),
                              cache_read_input_tokens=usage.get("cache_read", 0)))


def filled_context() -> Context:
    ctx = Context("read the sign", Config())
    call = ToolCall(id="t0", name="view", arguments={"asset_id": "im_a"})
    ctx.add_turn("Looking.", [call])
    ctx.add_output(call, ToolResult(text="artifact=a_0001 scale=2.0",
                                    images=[Image.new("RGB", (64, 48))],
                                    artifacts=["a_0001"]))
    return ctx


# --- outgoing translation ------------------------------------------------
def test_images_are_encoded_and_text_is_left_alone():
    out = to_messages(filled_context().render())
    result = out[-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t0"

    image, text = result["content"]
    assert image["source"]["type"] == "base64"
    assert image["source"]["media_type"] == "image/png"
    assert image["source"]["data"]  # actually encoded something
    assert text["type"] == "text"


def test_all_tool_results_ride_in_one_user_message():
    """Splitting them across messages trains the model out of parallel calls."""
    ctx = Context("t", Config())
    a = ToolCall(id="t0", name="view", arguments={})
    b = ToolCall(id="t1", name="view", arguments={})
    ctx.add_turn("two at once", [a, b])
    ctx.add_output(a, ToolResult(text="one"))
    ctx.add_output(b, ToolResult(text="two"))

    users = [m for m in to_messages(ctx.render()) if m["role"] == "user"]
    carrying = [m for m in users
                if any(b["type"] == "tool_result" for b in m["content"])]
    assert len(carrying) == 1
    assert len(carrying[0]["content"]) == 2


def test_a_failed_call_is_flagged_as_an_error():
    ctx = Context("t", Config())
    call = ToolCall(id="t0", name="view", arguments={})
    ctx.add_turn("bad box", [call])
    ctx.add_error(call, "x0 < x1 required")

    result = to_messages(ctx.render())[-1]["content"][0]
    assert result["is_error"] is True


def test_a_successful_call_is_not_flagged():
    result = to_messages(filled_context().render())[-1]["content"][0]
    assert "is_error" not in result


def test_an_assistant_turn_is_replayed_verbatim_when_the_backend_kept_one():
    """Thinking blocks cannot be rebuilt from text, so the original goes back."""
    ctx = Context("t", Config())
    original = [block(type="thinking", thinking="..."), block(type="text", text="hi")]
    ctx.add_turn("hi", [], raw=original)

    assistant = [m for m in to_messages(ctx.render()) if m["role"] == "assistant"][0]
    assert assistant["content"] is original


# --- incoming normalisation ---------------------------------------------
def test_tool_calls_and_usage_come_back_normalised():
    client = FakeClient(reply(text="Zooming.",
                              calls=[("tu_1", "view", {"asset_id": "im_a"})],
                              stop_reason="tool_use", input=120, output=8))
    out = AnthropicBackend(Config(), client=client).complete("sys", [], [])

    assert out.text == "Zooming."
    assert [(c.id, c.name, c.arguments) for c in out.calls] == [
        ("tu_1", "view", {"asset_id": "im_a"})]
    assert out.usage == {"input": 120, "output": 8}
    assert out.stop_reason == "tool_use"
    assert out.raw is not None, "nothing kept to replay next turn"


def test_the_system_prompt_is_sent_cacheable():
    """Tools and system are the stable prefix; caching them is the whole point of
    eliding in batches rather than every turn."""
    client = FakeClient(reply())
    AnthropicBackend(Config(), client=client).complete("be careful", [], [])

    system = client.messages.seen["system"]
    assert system[0]["text"] == "be careful"
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_the_history_is_sent_cacheable_at_its_tail():
    """Without this every turn re-bills the whole history at full price."""
    client = FakeClient(reply())
    AnthropicBackend(Config(), client=client).complete(
        "s", filled_context().render(), [])

    messages = client.messages.seen["messages"]
    assert messages[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    marked = [b for m in messages[:-1] for b in m["content"]
              if isinstance(b, dict) and "cache_control" in b]
    assert not marked, "only the tail carries the breakpoint"


def test_max_tokens_comes_from_config():
    client = FakeClient(reply())
    cfg = Config(model=ModelConfig(max_tokens=1234))
    AnthropicBackend(cfg, client=client).complete("s", [], [])
    assert client.messages.seen["max_tokens"] == 1234


def test_a_refusal_is_reported_rather_than_looking_like_silence():
    message = reply(text="", stop_reason="refusal")
    message.stop_details = SimpleNamespace(category="cyber", explanation="no")
    out = AnthropicBackend(Config(), client=FakeClient(message)).complete("s", [], [])

    assert out.stop_reason == "refusal"
    assert "refused" in out.text and "cyber" in out.text


# --- replay --------------------------------------------------------------
def test_a_script_is_played_in_order_then_runs_dry():
    backend = ReplayBackend(Config(), script=[
        {"text": "one", "calls": [{"name": "view", "arguments": {"asset_id": "a"}}]},
        {"text": "two"}])

    first = backend.complete("s", [], [])
    assert first.text == "one" and first.calls[0].name == "view"
    assert backend.complete("s", [], []).calls == []
    assert backend.complete("s", [], []).stop_reason == "end_turn"  # past the end


def test_a_script_loads_from_a_file(tmp_path):
    path = tmp_path / "script.json"
    path.write_text(json.dumps([{"text": "hello"}]))
    assert ReplayBackend(Config(), path=path).complete("s", [], []).text == "hello"


def test_replay_records_what_it_was_sent():
    backend = ReplayBackend(Config(), script=[])
    backend.complete("the prompt", [{"role": "user", "content": []}], [{"name": "view"}])
    assert backend.requests[0]["system"] == "the prompt"
    assert backend.requests[0]["tools"] == [{"name": "view"}]


def test_replay_needs_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        ReplayBackend(Config())
    with pytest.raises(ValueError, match="exactly one"):
        ReplayBackend(Config(), script=[], path="x.json")


def test_sweep_makes_one_view_per_asset():
    script = sweep(["im_a", "im_b"], box=[0, 0, 10, 10])
    assert len(script) == 2
    assert script[0]["calls"][0]["arguments"] == {"asset_id": "im_a",
                                                  "box": [0, 0, 10, 10]}


# --- selection -----------------------------------------------------------
def test_an_unknown_backend_names_the_ones_that_exist():
    cfg = Config(model=ModelConfig(backend="openai_compatible"))
    with pytest.raises(ValueError, match="Phase 5"):
        load(cfg)
