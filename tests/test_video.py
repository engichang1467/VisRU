"""Phase 3: the keyframe index, temporal search, tracking, and subagents.

The embedder is faked with one that genuinely reads the frame: `make_video`'s
frame k is flat gray 8*k, so a vector peaked at bin k identifies the frame, and
the query "frame 12" peaks at bin 12. Search logic is then tested on content,
not on a stub that returns whatever the test wanted.
"""

import re
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch
from PIL import Image

from visru import models
from visru.tools import open_workspace
from visru.tools.search import search, track
from visru.workspace import index
from visru.workspace.registry import Registry

BINS = 40


def _bump(centre: float) -> np.ndarray:
    v = np.exp(-0.5 * ((np.arange(BINS) - centre) / 1.5) ** 2)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    name = "fake-gray"

    def __init__(self):
        self.images_seen = 0

    def images(self, images):
        self.images_seen += len(images)
        return np.stack([_bump(np.asarray(im.convert("L")).mean() / 8) for im in images])

    def text(self, text):
        return _bump(float(re.search(r"\d+", text).group()))


@pytest.fixture
def fake(monkeypatch):
    emb = FakeEmbedder()
    monkeypatch.setattr(models, "embedder", lambda: emb)
    return emb


@pytest.fixture
def vws(tmp_path, make_video):
    make_video("clip.mp4", frames=30, fps=10)
    Image.new("RGB", (64, 48), (8 * 25,) * 3).save(tmp_path / "still.png")
    reg = Registry(tmp_path)
    reg.scan()
    reg.save()
    return tmp_path


def _vid(ctx):
    return ctx.registry.list(kind="video")[0].id


def test_index_embeds_every_sampled_frame_and_every_image(vws, fake):
    idx, added = index.build(vws, Registry.load(vws), fps=10)
    assert added == 2
    # 30 midpoints at 10 fps each decode the frame at or after them: frames 1..29,
    # the last one twice, stored once. Plus the still.
    assert len(idx) == 30
    loaded = index.load(vws)
    assert loaded.refs == idx.refs and loaded.model == "fake-gray"
    frames = [r for r in idx.refs if "#" in r]
    assert [int(r.split("#")[1]) for r in frames] == list(range(1, 30))


def test_reindex_skips_assets_already_embedded(vws, fake):
    index.build(vws, Registry.load(vws), fps=10)
    seen = fake.images_seen
    idx, added = index.build(vws, Registry.load(vws), fps=10)
    assert added == 0 and fake.images_seen == seen and len(idx) == 30


def test_a_different_model_rebuilds_the_index(vws, fake, monkeypatch):
    index.build(vws, Registry.load(vws), fps=10)
    other = FakeEmbedder()
    other.name = "fake-other"
    monkeypatch.setattr(models, "embedder", lambda: other)
    idx, added = index.build(vws, Registry.load(vws), fps=10)
    assert added == 2 and idx.model == "fake-other"


def test_search_ranks_the_matching_frame_first_and_merges_a_window(vws, fake):
    index.build(vws, Registry.load(vws), fps=10)
    ctx = open_workspace(vws)
    vid = _vid(ctx)
    res = search(ctx, "frame 12", ref=vid, k=3)

    art = ctx.artifacts.load(res.artifacts[0])
    assert art.kind == "search" and art.parent == vid
    hits = art.params["hits"]
    assert hits[0]["ref"] == f"{vid}#12"
    assert {h["ref"] for h in hits} == {f"{vid}#{i}" for i in (11, 12, 13)}
    [window] = art.params["windows"]
    # three samples 0.1 s apart, padded by half a step: 1.1..1.3 -> 1.05..1.35
    assert window["t0"] == pytest.approx(1.05, abs=0.01)
    assert window["t1"] == pytest.approx(1.35, abs=0.01)
    assert "mark 1:" in res.text and len(res.images) == 1


def test_search_splits_hits_that_are_far_apart(vws, fake, monkeypatch):
    index.build(vws, Registry.load(vws), fps=10)
    # A query that matches two separate moments equally.
    monkeypatch.setattr(fake, "text", lambda _t: (_bump(5) + _bump(20)) / np.linalg.norm(
        _bump(5) + _bump(20)))
    ctx = open_workspace(vws)
    res = search(ctx, "either", ref=_vid(ctx), k=4)
    wins = ctx.artifacts.load(res.artifacts[0]).params["windows"]
    assert len(wins) == 2


def test_search_honours_a_time_range(vws, fake):
    index.build(vws, Registry.load(vws), fps=10)
    ctx = open_workspace(vws)
    res = search(ctx, "frame 12", ref=_vid(ctx), t0=2.0, k=2)
    hits = ctx.artifacts.load(res.artifacts[0]).params["hits"]
    assert all(h["time"] >= 2.0 for h in hits)


def test_search_can_rank_an_image_first(vws, fake):
    index.build(vws, Registry.load(vws), fps=10)
    ctx = open_workspace(vws)
    still = ctx.registry.list(kind="image")[0].id
    res = search(ctx, "frame 25", k=1)
    assert still in res.text and not res.artifacts


def test_search_without_an_index_says_how_to_build_one(vws):
    ctx = open_workspace(vws)
    with pytest.raises(ValueError, match="--embed"):
        search(ctx, "anything")


def test_search_sheet_is_regenerable(vws, fake):
    from visru.agent.trajectory import replay_artifact

    index.build(vws, Registry.load(vws), fps=10)
    ctx = open_workspace(vws)
    art_id = search(ctx, "frame 12", ref=_vid(ctx), k=3).artifacts[0]
    remade = replay_artifact(ctx, art_id)
    assert remade.tobytes() == ctx.artifacts.load_image(art_id).tobytes()


# --- track ---------------------------------------------------------------------
SQUARE = 24


def _square_at(k: int) -> tuple[int, int]:
    return 10 + 4 * k, 40  # moves right 4 px a frame


@pytest.fixture
def square_ws(tmp_path):
    """160x120, 10 fps, 30 frames; a white square that vanishes from frame 20 on."""
    with av.open(str(tmp_path / "square.mp4"), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        for k in range(30):
            img = Image.new("RGB", (160, 120), (0, 0, 0))
            if k < 20:
                x, y = _square_at(k)
                img.paste((255, 255, 255), (x, y, x + SQUARE, y + SQUARE))
            container.mux(stream.encode(av.VideoFrame.from_image(img)))
        container.mux(stream.encode(None))
    reg = Registry(tmp_path)
    reg.scan()
    reg.save()
    return tmp_path


class _StubVideoProcessor:
    """Stands in for Sam2VideoProcessor; the 'model' masks whatever is bright."""

    def init_video_session(self, video, inference_device, dtype):
        return SimpleNamespace(video=video, num_frames=len(video))

    def add_inputs_to_inference_session(self, inference_session, frame_idx, obj_ids, **kw):
        inference_session.obj_ids = obj_ids
        self.prompt = kw

    def post_process_masks(self, masks, original_sizes, **kw):
        return [masks[0]]


class _StubVideoModel:
    device = "cpu"

    def __call__(self, inference_session, frame_idx):
        return None

    def propagate_in_video_iterator(self, session):
        for i, im in enumerate(session.video):
            bright = torch.tensor(np.asarray(im.convert("L")) > 128)
            yield SimpleNamespace(
                frame_idx=i,
                pred_masks=bright[None, None].repeat(len(session.obj_ids), 1, 1, 1))


@pytest.fixture
def stub_sam(monkeypatch):
    proc = _StubVideoProcessor()
    monkeypatch.setattr(models, "video_segmenter", lambda: (proc, _StubVideoModel()))
    return proc


def test_track_reports_when_the_object_disappears(square_ws, stub_sam):
    ctx = open_workspace(square_ws)
    vid = _vid(ctx)
    x, y = _square_at(5)
    res = track(ctx, f"{vid}#5", boxes=[[x, y, x + SQUARE, y + SQUARE]])

    art = ctx.artifacts.load(res.artifacts[0])
    obj = art.params["objects"]["1"]
    assert obj["present"] == [pytest.approx(0.5), pytest.approx(1.9)]
    assert "gone from 2.00s (frame 20)" in res.text
    # Boxes come back in source pixels on every frame the object is there.
    for row in obj["frames"]:
        if row["index"] < 20:
            sx, sy = _square_at(row["index"])
            assert row["box"] == pytest.approx((sx, sy, sx + SQUARE, sy + SQUARE), abs=3)
        else:
            assert row["box"] is None
    # The prompt reached SAM in the frames it was given (same size here: no scaling).
    assert stub_sam.prompt["input_boxes"] == [[[x, y, x + SQUARE, y + SQUARE]]]
    assert len(art.params["cells"]) == 12


def test_track_needs_a_frame_and_one_kind_of_prompt(square_ws, stub_sam):
    ctx = open_workspace(square_ws)
    vid = _vid(ctx)
    with pytest.raises(ValueError, match="#<frame_index>"):
        track(ctx, vid, points=[[20, 50]])
    with pytest.raises(ValueError, match="not both"):
        track(ctx, f"{vid}#0")
    with pytest.raises(ValueError, match="not after"):
        track(ctx, f"{vid}#10", points=[[20, 50]], t1=0.5)


def test_track_sheet_is_regenerable_without_the_model(square_ws, stub_sam, monkeypatch):
    from visru.agent.trajectory import replay_artifact

    ctx = open_workspace(square_ws)
    x, y = _square_at(0)
    art_id = track(ctx, f"{_vid(ctx)}#0", points=[[x + 12, y + 12]]).artifacts[0]
    monkeypatch.setattr(models, "video_segmenter", lambda: pytest.fail("model called"))
    assert replay_artifact(ctx, art_id).tobytes() == \
        ctx.artifacts.load_image(art_id).tobytes()


@pytest.mark.models
def test_track_with_real_sam2_loses_the_square_when_it_vanishes(square_ws):
    ctx = open_workspace(square_ws)
    x, y = _square_at(2)
    res = track(ctx, f"{_vid(ctx)}#2", boxes=[[x, y, x + SQUARE, y + SQUARE]])
    obj = ctx.artifacts.load(res.artifacts[0]).params["objects"]["1"]
    start, end = obj["present"]
    assert start == pytest.approx(0.2) and end == pytest.approx(1.9, abs=0.15)
    row = next(r for r in obj["frames"] if r["index"] == 10)
    sx, sy = _square_at(10)
    assert row["box"] == pytest.approx((sx, sy, sx + SQUARE, sy + SQUARE), abs=4)


# --- temporal checks and coverage ------------------------------------------------
def _tracked(ctx):
    x, y = _square_at(5)
    return track(ctx, f"{_vid(ctx)}#5", boxes=[[x, y, x + SQUARE, y + SQUARE]]).artifacts[0]


def test_a_track_and_your_reading_of_a_frame_can_verify_a_window(square_ws, stub_sam):
    from visru.tools.navigate import view
    from visru.tools.verify import cross_check

    ctx = open_workspace(square_ws)
    tracked = _tracked(ctx)
    looked = view(ctx, f"{_vid(ctx)}#19").artifacts[0]

    agree = cross_check(ctx, "temporal", [looked, tracked], value="0.5..2.0")
    check = ctx.artifacts.load(agree.artifacts[0])
    assert check.params["result"] == "agree" and check.params["readings"][1] == [0.5, 1.9]
    assert check.params["detail"]["max_gap_s"] == pytest.approx(0.1)

    disagree = cross_check(ctx, "temporal", [looked, tracked], value="0.5..4.0")
    assert ctx.artifacts.load(disagree.artifacts[0]).params["result"] == "disagree"


def test_two_readings_by_the_model_are_not_independent(square_ws):
    from visru.tools.navigate import sample_frames, view
    from visru.tools.verify import cross_check

    ctx = open_workspace(square_ws)
    sheet = sample_frames(ctx, _vid(ctx)).artifacts[0]
    looked = view(ctx, f"{_vid(ctx)}#19").artifacts[0]
    with pytest.raises(ValueError, match="not independent"):
        cross_check(ctx, "temporal", [sheet, looked], value="0.5..2.0")


def test_a_temporal_reading_must_be_a_window(square_ws, stub_sam):
    from visru.tools.navigate import view
    from visru.tools.verify import cross_check

    ctx = open_workspace(square_ws)
    tracked = _tracked(ctx)
    looked = view(ctx, f"{_vid(ctx)}#19").artifacts[0]
    with pytest.raises(ValueError, match="window like"):
        cross_check(ctx, "temporal", [looked, tracked], value="about 2 s")
    with pytest.raises(ValueError, match="stands for your own reading"):
        cross_check(ctx, "temporal", [looked, tracked])


def test_coverage_counts_track_frames_and_what_subagents_saw(square_ws, stub_sam):
    from visru.tools import open_workspace as ow
    from visru.tools.navigate import sample_frames
    from visru.tools.verify import coverage_summary

    def cells(c, art_id):
        return {cell["index"] for cell in c.artifacts.load(art_id).params["cells"]}

    parent = ow(square_ws, session="s_parent")
    mine = cells(parent, _tracked(parent))
    child = ow(square_ws, session="s_parent_w1")
    theirs = cells(child, sample_frames(child, _vid(child), n=4, t0=2.0, t1=3.0).artifacts[0])
    stranger = ow(square_ws, session="s_other")
    sample_frames(stranger, _vid(stranger), n=4, t0=0.0, t1=0.4)  # frames 1..4

    frames = coverage_summary(parent)["assets"][_vid(parent)]["frames_seen"]
    # the track's sheet cells plus the child's; the stranger's early frames are not ours
    assert frames == len(mine | theirs) and min(mine | theirs) == 5


# --- delegate ----------------------------------------------------------------------
def test_delegate_runs_a_subagent_per_window_on_the_parents_model(square_ws):
    from visru.agent.loop import run
    from visru.backends.replay import ReplayBackend

    ctx = open_workspace(square_ws, session="s_p")
    vid = _vid(ctx)
    u = {"input": 100, "output": 10}

    def child(t0, t1, art, text):
        return [{"calls": [{"name": "sample_frames",
                            "arguments": {"asset_id": vid, "n": 4, "t0": t0, "t1": t1}}],
                 "usage": u},
                {"calls": [{"name": "submit_report", "arguments": {"claims": [
                    {"claim_id": "c_1", "text": text, "status": "supported",
                     "evidence": [art]}]}}], "usage": u}]

    script = (
        [{"calls": [{"name": "delegate", "arguments": {
            "asset_id": vid, "question": "When does the square vanish?", "n": 2}}],
          "usage": u}]
        + child(0.0, 1.5, "a_0001", "the square is present throughout")
        + child(1.5, 3.0, "a_0002", "the square vanishes at 2.0..2.1")
        + [{"calls": [{"name": "submit_report", "arguments": {"claims": [
            {"claim_id": "answer", "text": "2.0..2.1", "status": "supported",
             "evidence": ["a_0002"]}]}}], "usage": u}])
    backend = ReplayBackend(ctx.cfg, script=script)
    session = run("When does the square vanish?", ctx, backend, max_steps=5)

    assert session.reported
    # 2 parent turns + 2 turns in each of 2 children, all billed to the parent.
    assert session.usage == {"input": 600, "output": 60}
    child_tools = {t["name"] for t in backend.requests[1]["tools"]}
    assert "delegate" not in child_tools and "search" in child_tools
    assert backend.requests[1]["messages"][0]["content"][0]["text"].count("0.00..1.50") == 1

    from visru.agent.trajectory import read as read_log
    [delegated] = [r for r in read_log(session.trajectory) if r.tool == "delegate"]
    assert delegated.artifacts == ["a_0001", "a_0002"]
    assert "s_p_w1: reported" in delegated.text and "s_p_w2: reported" in delegated.text
    assert "vanishes at 2.0..2.1" in delegated.text

    from visru.tools.verify import coverage_summary
    assert coverage_summary(ctx)["assets"][vid]["frames_seen"] == 8


def test_delegate_needs_the_loop_and_a_video(square_ws, tmp_path):
    from visru.agent.subagents import delegate

    ctx = open_workspace(square_ws)
    with pytest.raises(ValueError, match="needs the visru agent loop"):
        delegate(ctx, _vid(ctx), "anything")
    ctx.backend = object()
    with pytest.raises(ValueError, match="1..8"):
        delegate(ctx, _vid(ctx), "anything", n=9)
    with pytest.raises(ValueError, match="empty"):
        delegate(ctx, _vid(ctx), "anything", windows=[[5.0, 9.0]])
