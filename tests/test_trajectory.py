"""The session record, and the regeneration claim the exit criterion rests on."""

import json

import pytest
from PIL import Image

from visru.agent import trajectory
from visru.agent.trajectory import Record, Trajectory
from visru.tools import open_workspace
from visru.tools.navigate import sample_frames, view
from visru.tools.verify import overlay
from visru.workspace.registry import Registry


@pytest.fixture
def ctx(marker_workspace):
    reg = Registry(marker_workspace)
    reg.scan()
    reg.save()
    return open_workspace(marker_workspace)


def _ids(ctx):
    return (ctx.registry.list(kind="image")[0].id, ctx.registry.list(kind="video")[0].id)


# --- writing -------------------------------------------------------------
def test_records_append_and_read_back_in_order(ctx):
    log = Trajectory(ctx.root, "s_test")
    log.model_turn(0, "I will start with a wide view.", usage={"input": 12})
    log.tool_call(0, "view", {"asset_id": "im_x"}, "artifact=a_0001", ["a_0001"], True)
    log.model_turn(1, "Zooming in.")

    records = trajectory.read(log.path)
    assert [r.kind for r in records] == ["model", "tool", "model"]
    assert records[1].tool == "view" and records[1].shown
    assert records[0].usage == {"input": 12}
    assert all(r.ts for r in records)


def test_a_crashed_run_keeps_everything_before_the_crash(ctx):
    log = Trajectory(ctx.root, "s_killed")
    log.model_turn(0, "first")
    with log.path.open("a") as sink:
        sink.write('{"step": 1, "kind": "mod')  # killed mid-write

    records = trajectory.read(log.path)
    assert [r.text for r in records] == ["first"]


def test_shown_artifacts_skips_ones_the_model_never_saw(ctx):
    log = Trajectory(ctx.root, "s_shown")
    log.tool_call(0, "view", {}, "", ["a_0001"], True)
    log.tool_call(1, "view", {}, "", ["a_0001"], True)  # same image twice
    log.tool_call(2, "submit_report", {}, "", ["a_0001", "a_0009"], False)

    assert trajectory.shown_artifacts(trajectory.read(log.path)) == ["a_0001"]


# --- regeneration: the exit criterion's second half ----------------------
def test_a_view_is_remade_pixel_for_pixel(ctx):
    im_id, _ = _ids(ctx)
    shown = view(ctx, im_id, box=[2800, 1800, 3200, 2200])
    remade = trajectory.replay_artifact(ctx, shown.artifacts[0])
    assert remade.tobytes() == shown.images[0].tobytes()


def test_a_grid_view_is_remade_with_its_grid(ctx):
    im_id, _ = _ids(ctx)
    shown = view(ctx, im_id, box=[2800, 1800, 3200, 2200], grid=True)
    remade = trajectory.replay_artifact(ctx, shown.artifacts[0])
    assert remade.tobytes() == shown.images[0].tobytes()


def test_a_contact_sheet_is_remade_from_its_parameters(ctx):
    _, vid_id = _ids(ctx)
    shown = sample_frames(ctx, vid_id, n=4)
    remade = trajectory.replay_artifact(ctx, shown.artifacts[0])
    assert remade.tobytes() == shown.images[0].tobytes()


def test_marks_are_redrawn_without_loading_the_model(ctx):
    """An overlay carries marks and masks in its sidecar, so regenerating it needs
    no detector — which is what makes checking a log cheap."""
    im_id, _ = _ids(ctx)
    shown = overlay(ctx, im_id, boxes=[[2800, 1800, 3200, 2200]], labels=["a box"])
    remade = trajectory.replay_artifact(ctx, shown.artifacts[0])
    assert remade.convert("RGB").tobytes() == shown.images[0].convert("RGB").tobytes()


def test_re_viewing_an_artifact_is_itself_regenerable(ctx):
    """Elision hands the model an artifact ID back; whatever it does with it has to
    stay in the log's reach."""
    im_id, _ = _ids(ctx)
    first = view(ctx, im_id, box=[2800, 1800, 3200, 2200]).artifacts[0]
    again = view(ctx, first)
    remade = trajectory.replay_artifact(ctx, again.artifacts[0])
    assert remade.tobytes() == again.images[0].tobytes()


def test_verify_regenerable_reports_every_shown_image_as_sound(ctx):
    im_id, vid_id = _ids(ctx)
    log = Trajectory(ctx.root, "s_exit")
    for result, name in ((view(ctx, im_id, box=[2800, 1800, 3200, 2200]), "view"),
                         (view(ctx, im_id, grid=True), "view"),
                         (sample_frames(ctx, vid_id, n=4), "sample_frames")):
        log.tool_call(0, name, {}, result.text, result.artifacts, shown=True)

    assert trajectory.verify_regenerable(ctx, trajectory.read(log.path)) == []


def test_a_deleted_artifact_is_reported_not_silently_passed(ctx):
    im_id, _ = _ids(ctx)
    shown = view(ctx, im_id)
    log = Trajectory(ctx.root, "s_gone")
    log.tool_call(0, "view", {}, shown.text, shown.artifacts, shown=True)

    (ctx.artifacts.dir / f"{shown.artifacts[0]}.json").unlink()
    assert trajectory.verify_regenerable(
        ctx, trajectory.read(log.path)) == shown.artifacts
