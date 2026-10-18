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
    assert [r.kind for r in records[:3]] == ["session", "model", "tool"]


def test_the_log_opens_with_a_header_that_names_the_model(ctx):
    """A trajectory nobody can attribute is not training data."""
    im_id = ctx.registry.list(kind="image")[0].id
    out = run("look", ctx, Scripted(looks(0, im_id)), max_steps=2)
    header = read_log(out.trajectory)[0]
    assert header.kind == "session"
    assert header.meta["model"] == ctx.cfg.model.name
    assert header.meta["backend"] == ctx.cfg.model.backend
    assert header.meta["task"] == "look"
    assert "view" in header.meta["tools"]


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
    assert "Workspace context" not in backend.systems[0]


def test_the_workspace_visru_md_is_appended_to_the_prompt(ctx):
    (ctx.root / "VISRU.md").write_text("All boxes are in millimetres.")
    backend = Scripted()
    run("look", ctx, backend, max_steps=1)

    assert backend.systems[0].startswith(loop.PROMPT_PATH.read_text())
    assert backend.systems[0].endswith("All boxes are in millimetres.")


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
    # Distinct boxes: a model repeating one call verbatim is "stuck", not "exhausted".
    backend = Scripted(*[looks(i, im_id, box=[0, 0, 100 + i, 100 + i]) for i in range(5)])
    out = run("look", ctx, backend, max_steps=3)

    assert out.reason == "exhausted" and out.steps == 3
    assert not out.reported


def test_a_model_that_reissues_one_call_is_cut_off(ctx):
    """The real failure mode: submit_report is rejected, the model reruns the same
    failing check, says nothing callable, gets nudged, reruns it — to max_steps."""
    im_id = ctx.registry.list(kind="image")[0].id
    same = lambda: looks(0, im_id, box=[0, 0, 100, 100])  # noqa: E731
    backend = Scripted(same(), Response(text="thinking"), same(),
                        Response(text="thinking"), same(), *[same() for _ in range(20)])
    out = run("look", ctx, backend, max_steps=50)

    assert out.reason == "stuck"
    assert out.steps < 10, "the ping-pong ran on"


def test_a_model_cycling_between_two_calls_is_cut_off(ctx):
    """Observed with a 30B: view and ocr on one box, alternating, to the step limit.
    Neither turn repeats the turn before it, so consecutive-repeat counting is blind
    to it."""
    im_id = ctx.registry.list(kind="image")[0].id
    box = [0, 0, 100, 100]
    cycle = [Response(text="zoom", calls=[call(0, "view", asset_id=im_id, box=box)],
                        usage={"input": 100, "output": 20}),
            Response(text="read", calls=[call(1, "ocr", asset_id=im_id, box=box)],
                        usage={"input": 100, "output": 20})]
    backend = Scripted(*(cycle * 12))
    out = run("look", ctx, backend, max_steps=50)


    assert out.reason == "stuck"
    assert out.steps < 12, "the cycle ran on"


def test_a_truncated_turn_is_told_so_rather_than_nudged_for_a_report(ctx):
    ramble = Response(text="a long ramble that never ends", stop_reason="length")
    backend = Scripted(ramble, ramble)
    out = run("look", ctx, backend, max_steps=2)

    said = [b["text"] for m in backend.seen[-1] if m["role"] == "user"
            for b in m["content"] if b.get("type") == "text"]
    # Told what happened, and never asked for a report it was cut off before writing.
    assert any("cut off" in s for s in said) and not any("submit_report" in s for s in said)
    assert out.reason == "exhausted"


def test_turns_that_keep_dying_at_max_tokens_end_the_run(ctx):
    """The degenerate open-model pattern: one turn repeats a sentence to the token
    cap, the next makes one small real call, repeat. Every consecutive counter
    resets on the call in between, so the truncation count must not."""
    im_id = ctx.registry.list(kind="image")[0].id
    ramble = Response(text="and again " * 50, stop_reason="length")
    backend = Scripted(ramble, looks(1, im_id, box=[0, 0, 110, 110]),
                        ramble, looks(2, im_id, box=[0, 0, 120, 120]),
                        ramble, looks(3, im_id, box=[0, 0, 130, 130]))
    out = run("look", ctx, backend, max_steps=50)

    assert out.reason == "stuck" and out.steps == 5


def test_usage_accumulates_across_steps(ctx):
    im_id = ctx.registry.list(kind="image")[0].id
    out = run("look", ctx, Scripted(looks(0, im_id), looks(1, im_id)), max_steps=2)
    assert out.usage["input"] == 200 and out.usage["output"] == 40


# --- the budget is enforced by the loop, not just available to it --------
def test_a_long_run_stays_inside_the_image_budget(ctx):
    im_id = ctx.registry.list(kind="image")[0].id
    backend = Scripted(*[looks(i, im_id, box=[0, 0, 100 + i, 100 + i]) for i in range(60)])
    out = run("look", ctx, backend, max_steps=60)

    assert out.peak_live_images <= ctx.cfg.context.max_live_images
    assert out.elisions > 0, "60 images and nothing was elided"
    images = [b for m in backend.seen[-1] for b in m["content"]
              if b.get("type") == "tool_result"
              for b in b["content"] if b["type"] == "image"]
    assert len(images) <= ctx.cfg.context.max_live_images
