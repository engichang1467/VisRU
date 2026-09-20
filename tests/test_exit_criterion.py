"""The Phase 1 exit criterion, run rather than asserted.

README.md:194 — "a 200-step session over a 1,000-image folder stays inside
context limits, and every image the model saw can be regenerated from the log."

Two halves, two tests. The session is driven by the replay backend: what is
being checked is that the harness holds over a long run, and what a real model
would have said has nothing to do with that. Replay makes it free to re-run
after every change to the context manager, which is the point — a budget bug is
silent, and a check you only run when you remember is a check that rots.
"""

import pytest

from visru.agent import trajectory
from visru.agent.loop import run
from visru.backends.replay import ReplayBackend, sweep
from visru.tools import open_workspace

STEPS = 200


@pytest.fixture(scope="session")
def session_over_the_corpus(corpus, tmp_path_factory):
    """One 200-step run, shared by both halves so it is driven only once."""
    ctx = open_workspace(corpus)
    assets = [a.id for a in ctx.registry.list()][:STEPS]
    backend = ReplayBackend(ctx.cfg, script=sweep(assets))
    result = run("Find every damaged plate label.", ctx, backend, max_steps=STEPS)
    return ctx, backend, result


def test_the_corpus_really_is_a_thousand_images(corpus):
    ctx = open_workspace(corpus)
    assert len(ctx.registry.assets) == 1000


# --- half one: the session stays inside its limits ----------------------
def test_a_200_step_session_stays_inside_the_context_limits(session_over_the_corpus):
    ctx, backend, result = session_over_the_corpus
    limits = ctx.cfg.context

    assert result.steps == STEPS, "the run ended early; it is not testing what it claims"
    assert result.peak_live_images <= limits.max_live_images
    assert result.elisions > 0, "200 images and nothing was elided"

    # Asserted on what the backend was actually handed, not on our own counter:
    # a bug that miscounts would otherwise agree with itself.
    for request in backend.requests:
        images = [block
                  for message in request["messages"]
                  for block in message["content"]
                  if block.get("type") == "tool_result"
                  for block in block["content"] if block["type"] == "image"]
        assert len(images) <= limits.max_live_images


def test_no_request_in_the_session_exceeded_the_token_budget(session_over_the_corpus):
    ctx, backend, _ = session_over_the_corpus
    from visru.agent.context import CHARS_PER_TOKEN
    from visru.geometry import visual_tokens

    patch = ctx.cfg.model.patch
    peak = 0
    for request in backend.requests:
        chars, tokens = len(request["system"]), 0
        for message in request["messages"]:
            for block in message["content"]:
                chars += len(str(block.get("text", "")))
                for inner in block.get("content", []) if isinstance(block, dict) else []:
                    if inner["type"] == "image":
                        im = inner["image"]
                        tokens += visual_tokens(im.width, im.height, patch)
                    else:
                        chars += len(inner.get("text", ""))
        peak = max(peak, tokens + chars // CHARS_PER_TOKEN)

    assert peak <= ctx.cfg.context.compact_at_tokens, (
        f"peak request was {peak:,} tokens against a "
        f"{ctx.cfg.context.compact_at_tokens:,} limit")


def test_every_step_was_answered(session_over_the_corpus):
    """A dropped tool result is a 400 from the provider, not a quiet degradation."""
    _, backend, _ = session_over_the_corpus
    final = backend.requests[-1]["messages"]

    calls, results = [], []
    for message in final:
        for block in message["content"]:
            if block.get("type") == "tool_use":
                calls.append(block["id"])
            if block.get("type") == "tool_result":
                results.append(block["tool_use_id"])
    assert calls == results


# --- half two: every image the model saw can be remade ------------------
def test_every_image_the_model_saw_regenerates_from_the_log(session_over_the_corpus):
    ctx, _, result = session_over_the_corpus
    records = trajectory.read(result.trajectory)

    shown = trajectory.shown_artifacts(records)
    assert len(shown) == STEPS, f"expected {STEPS} shown images, logged {len(shown)}"
    assert trajectory.verify_regenerable(ctx, records) == []


def test_regeneration_reads_the_log_not_the_session(session_over_the_corpus):
    """The claim is that the log is sufficient — so reopen the workspace cold and
    rebuild from the file, with nothing carried over from the run."""
    ctx, _, result = session_over_the_corpus
    cold = open_workspace(ctx.root)
    records = trajectory.read(result.trajectory)

    remade = trajectory.replay_artifact(cold, trajectory.shown_artifacts(records)[0])
    stored = cold.artifacts.load_image(trajectory.shown_artifacts(records)[0])
    assert remade.convert("RGB").tobytes() == stored.convert("RGB").tobytes()
