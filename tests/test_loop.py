"""The agent loop, driven by a scripted backend.

The real backends are Phase 1 step 6; what matters here is the control flow —
that tool calls are answered, that errors come back as messages the model can act
on rather than ending the run, and that every ending is recorded honestly.
"""

import pytest
from PIL import Image

from visru.agent import loop
from visru.agent.context import ToolCall
from visru.agent.loop import run
from visru.agent.trajectory import read as read_log
from visru.backends import Response
from visru.tools import open_workspace
from visru.workspace.registry import Registry


@pytest.fixture
def ctx(marker_workspace):
    reg = Registry(marker_workspace)
    reg.scan()
    reg.save()
    return open_workspace(marker_workspace)


class Scripted:
    """Returns the responses it was given, then stops calling tools."""

    name = "scripted"

    def __init__(self, *responses: Response):
        self.queue = list(responses)
        self.seen: list[list[dict]] = []
        self.systems: list[str] = []
        self.tools: list[list[dict]] = []

    def complete(self, system, messages, tools) -> Response:
        self.seen.append(messages)
        self.systems.append(system)
        self.tools.append(tools)
        if self.queue:
            return self.queue.pop(0)
        return Response(text="Done looking.", usage={"input": 10, "output": 2})


def call(n: int, name: str, **arguments) -> ToolCall:
    return ToolCall(id=f"t{n}", name=name, arguments=arguments)


def looks(n: int, asset_id: str, **extra) -> Response:
    return Response(text=f"Step {n}.", calls=[call(n, "view", asset_id=asset_id, **extra)],
                    usage={"input": 100, "output": 20})


# --- the happy path ------------------------------------------------------
def test_a_submitted_report_ends_the_run(ctx):
    im_id = ctx.registry.list(kind="image")[0].id
    art = "a_0001"
    backend = Scripted(
        looks(0, im_id),
        Response(text="The marker is red.", calls=[call(1, "submit_report", claims=[
            {"claim_id": "c_1", "text": "There is a red marker.",
             "status": "supported", "evidence": [art]}])]),
        Response(text="should never be asked for", calls=[call(9, "view",
                                                               asset_id=im_id)]))

    out = run("find the marker", ctx, backend, max_steps=10)
    assert out.reported and out.reason == "reported"
    assert out.steps == 2, "the loop kept going after the report"
    assert "c_1" in out.report


def test_every_tool_call_is_answered_and_logged(ctx):
    im_id = ctx.registry.list(kind="image")[0].id
    backend = Scripted(looks(0, im_id), looks(1, im_id))

    out = run("look", ctx, backend, max_steps=3)
    records = read_log(out.trajectory)
    tools = [r for r in records if r.kind == "tool"]
    assert len(tools) == 2
    assert all(r.shown and r.artifacts for r in tools)
    assert [r.kind for r in records[:2]] == ["model", "tool"]


def test_the_backends_own_turn_is_handed_back_next_request(ctx):
    """Thinking blocks live in `raw`; dropping it loses the model's reading."""
    im_id = ctx.registry.list(kind="image")[0].id
    first = looks(0, im_id)
    first.raw = ["the original blocks"]
    backend = Scripted(first)
    run("look", ctx, backend, max_steps=2)

    assistant = [m for m in backend.seen[1] if m["role"] == "assistant"][0]
    assert assistant["raw"] is first.raw


def test_the_model_is_given_the_tools_and_the_prompt(ctx):
    backend = Scripted()
    run("look", ctx, backend, max_steps=1)

    names = {t["name"] for t in backend.tools[0]}
    assert "view" in names and "submit_report" in names
    assert "source-asset pixels" in backend.systems[0].lower()


# --- errors are messages, not endings ------------------------------------
def test_a_bad_call_comes_back_as_a_message_the_model_can_act_on(ctx):
    im_id = ctx.registry.list(kind="image")[0].id
    backend = Scripted(
        Response(text="Trying a bad box.",
                 calls=[call(0, "view", asset_id=im_id, box=[900, 700, 100, 500])]),
        looks(1, im_id))

    out = run("look", ctx, backend, max_steps=4)
    assert out.reason != "reported"  # it never reported, but it did not crash

    failed = [r for r in read_log(out.trajectory) if r.kind == "tool" and r.error]
    assert len(failed) == 1 and "x0 < x1" in failed[0].error
    # and the model got told, rather than the run ending
    assert "error:" in failed[0].text


def test_an_unknown_tool_name_does_not_end_the_run(ctx):
    backend = Scripted(Response(text="guessing", calls=[call(0, "zoom", asset_id="x")]))
    out = run("look", ctx, backend, max_steps=4)  # room for the nudge cycle after
    assert out.reason == "stopped"
    assert any(r.error and "no tool named" in r.error
               for r in read_log(out.trajectory) if r.kind == "tool")


# --- endings -------------------------------------------------------------
def test_a_model_that_stops_calling_tools_is_nudged_once_then_believed(ctx):
    backend = Scripted(Response(text="I think that is all."))
    out = run("look", ctx, backend, max_steps=10)

    assert out.reason == "stopped"
    assert out.steps == 2, "nudged more than once, or not at all"
    nudges = [b["text"] for m in backend.seen[-1] if m["role"] == "user"
              for b in m["content"] if b.get("type") == "text"]
    assert any("submit_report" in n for n in nudges)


def test_running_out_of_steps_is_recorded_as_such(ctx):
    im_id = ctx.registry.list(kind="image")[0].id
    backend = Scripted(*[looks(i, im_id) for i in range(5)])
    out = run("look", ctx, backend, max_steps=3)

    assert out.reason == "exhausted" and out.steps == 3
    assert not out.reported


def test_usage_accumulates_across_steps(ctx):
    im_id = ctx.registry.list(kind="image")[0].id
    out = run("look", ctx, Scripted(looks(0, im_id), looks(1, im_id)), max_steps=2)
    assert out.usage["input"] == 200 and out.usage["output"] == 40


# --- the budget is enforced by the loop, not just available to it --------
def test_a_long_run_stays_inside_the_image_budget(ctx):
    im_id = ctx.registry.list(kind="image")[0].id
    backend = Scripted(*[looks(i, im_id) for i in range(60)])
    out = run("look", ctx, backend, max_steps=60)

    assert out.peak_live_images <= ctx.cfg.context.max_live_images
    assert out.elisions > 0, "60 images and nothing was elided"
    images = [b for m in backend.seen[-1] for b in m["content"]
              if b.get("type") == "tool_result"
              for b in b["content"] if b["type"] == "image"]
    assert len(images) <= ctx.cfg.context.max_live_images
