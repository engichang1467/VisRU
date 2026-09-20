"""The live-image budget, elision, and compaction.

These failures are silent ones. A context manager that leaks images does not
raise, it just gets slower and more expensive until a request is refused or
truncated, so the bounds are asserted directly rather than inferred from a run.
"""

import pytest
from PIL import Image

from visru.agent.context import Context, ToolCall
from visru.config import Config, ContextConfig
from visru.geometry import visual_tokens
from visru.tools import ToolResult


def cfg(**overrides) -> Config:
    return Config(context=ContextConfig(**overrides)) if overrides else Config()


def shot(n: int = 1) -> ToolResult:
    """A tool result shaped like a real view: one image, text naming the artifact."""
    return ToolResult(
        text=(f"artifact=a_{n:04d} source=im_abc box=[0, 0, 400, 300] scale=2.000 "
              "— at or above inspected_scale"),
        images=[Image.new("RGB", (800, 600))],
        artifacts=[f"a_{n:04d}"])


def drive(ctx: Context, steps: int) -> None:
    for i in range(steps):
        call = ToolCall(id=f"t{i}", name="view", arguments={"asset_id": "im_abc"})
        ctx.add_turn(f"Looking at region {i}.", [call])
        ctx.add_output(call, shot(i))


# --- the budget ----------------------------------------------------------
def test_live_images_never_exceed_the_limit():
    ctx = Context("find the sign", cfg())
    drive(ctx, 200)
    assert ctx.live_images <= Config().context.max_live_images


def test_elision_happens_in_batches_not_one_per_step():
    """Rewriting history invalidates the cached prefix; doing it every turn would
    pay that cost on every turn."""
    c = cfg(max_live_images=16, elide_batch=8)
    ctx = Context("t", c)
    drive(ctx, 17)

    assert ctx.elisions == 8           # one batch, not one image
    assert ctx.live_images == 9        # 17 taken, 8 swapped for stubs
    drive(ctx, 7)                      # back up to 16: still no second batch
    assert ctx.elisions == 8


def test_an_elided_image_leaves_a_stub_naming_its_artifact():
    ctx = Context("t", cfg(max_live_images=2, elide_batch=1))
    drive(ctx, 4)

    blocks = [b for m in ctx.render() for b in m["content"]
              if b.get("type") == "tool_result"]
    stubs = [b["content"][0]["text"] for b in blocks
             if b["content"][0]["type"] == "text"]
    assert stubs, "nothing was elided"
    assert stubs[0].startswith("[a_0000 elided:")
    # The stub has to carry enough to decide whether to ask for it back.
    assert "box=[0, 0, 400, 300]" in stubs[0]
    assert "artifact=a_0000" not in stubs[0]  # not repeated inside its own stub


def test_elided_outputs_keep_their_place_in_the_conversation():
    """Every tool_use still needs its tool_result, elided or not."""
    ctx = Context("t", cfg(max_live_images=2, elide_batch=2))
    drive(ctx, 6)

    calls, results = [], []
    for message in ctx.render():
        for block in message["content"]:
            if block.get("type") == "tool_use":
                calls.append(block["id"])
            if block.get("type") == "tool_result":
                results.append(block["tool_use_id"])
    assert calls == results


def test_tokens_drop_when_images_are_elided():
    ctx = Context("t", cfg(max_live_images=100, elide_batch=1))
    drive(ctx, 10)
    before = ctx.estimate_tokens()

    ctx.cfg = cfg(max_live_images=2, elide_batch=8)
    ctx.enforce()
    assert ctx.estimate_tokens() < before
    assert ctx.live_images == 2


def test_the_estimate_counts_images_the_way_the_renderer_does():
    ctx = Context("", cfg())
    call = ToolCall(id="t0", name="view", arguments={})
    ctx.add_turn("", [call])
    ctx.add_output(call, ToolResult(text="", images=[Image.new("RGB", (800, 600))]))
    assert ctx.estimate_tokens() >= visual_tokens(800, 600, Config().model.patch)


# --- compaction ----------------------------------------------------------
def test_compaction_keeps_the_task_and_every_artifact_id():
    """A summary that dropped an ID would strand the evidence behind it: the model
    could no longer name an artifact the ledger still expects it to cite."""
    ctx = Context("count the red vehicles", cfg(compact_at_tokens=2_000))
    drive(ctx, 40)

    assert ctx.compactions >= 1
    opening = ctx.render()[0]["content"][0]["text"]
    assert opening.startswith("count the red vehicles")
    for n in range(20):  # the oldest steps, the ones most likely folded away
        assert f"a_{n:04d}" in opening


def test_compaction_folds_whole_steps_only():
    """Half a step would leave a tool call unanswered and the request rejected."""
    ctx = Context("t", cfg(compact_at_tokens=2_000))
    drive(ctx, 40)

    calls, results = [], []
    for message in ctx.render():
        for block in message["content"]:
            if block.get("type") == "tool_use":
                calls.append(block["id"])
            if block.get("type") == "tool_result":
                results.append(block["tool_use_id"])
    assert calls == results


def test_a_short_session_is_never_compacted():
    ctx = Context("t", cfg(compact_at_tokens=1))
    drive(ctx, 2)
    assert ctx.compactions == 0 and ctx.summary == ""


# --- shape -------------------------------------------------------------
def test_render_alternates_roles_and_opens_with_the_task():
    ctx = Context("read the sign", cfg())
    drive(ctx, 3)
    roles = [m["role"] for m in ctx.render()]
    assert roles[0] == "user"
    assert all(a != b for a, b in zip(roles, roles[1:])), roles


def test_a_failed_call_still_answers_its_tool_use():
    ctx = Context("t", cfg())
    call = ToolCall(id="t0", name="view", arguments={"asset_id": "nope"})
    ctx.add_turn("looking", [call])
    ctx.add_error(call, "unknown asset 'nope'; run visru index")

    result = ctx.render()[-1]["content"][0]
    assert result["tool_use_id"] == "t0"
    assert "visru index" in result["content"][0]["text"]


def test_a_tool_result_needs_a_turn_to_answer():
    ctx = Context("t", cfg())
    with pytest.raises(ValueError, match="needs a model turn"):
        ctx.add_output(ToolCall(id="t0", name="view", arguments={}), shot())
