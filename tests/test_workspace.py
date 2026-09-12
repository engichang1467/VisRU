import json

import pytest
from PIL import Image, ImageStat

from visru.geometry import ViewTransform
from visru.workspace import video
from visru.workspace.artifacts import ArtifactStore, Mark
from visru.workspace.registry import Registry, parse_ref


# --- registry ------------------------------------------------------------
def test_scan_finds_assets_and_skips_non_media(workspace):
    reg = Registry(workspace)
    stats = reg.scan()
    assert stats.hashed == 3 and stats.new == 2  # two distinct images, one duplicated
    assert not stats.failed
    assert {a.kind for a in reg.list()} == {"image"}
    big = reg.list(prefix="pallet_01")[0]
    assert (big.width, big.height) == (4032, 3024)
    assert big.id.startswith("im_")


def test_identical_bytes_share_one_id(workspace):
    reg = Registry(workspace)
    reg.scan()
    dupe = [a for a in reg.list() if len(a.paths) > 1]
    assert len(dupe) == 1
    assert set(dupe[0].paths) == {"pallet_02.png", "sub/copy_of_02.png"}


def test_ids_are_stable_across_rescans_and_reload(workspace):
    reg = Registry(workspace)
    reg.scan()
    reg.save()
    before = {a.id for a in reg.list()}

    again = Registry.load(workspace)
    stats = again.scan()
    assert {a.id for a in again.list()} == before
    assert stats.cached == 3 and stats.hashed == 0  # nothing re-read from disk
    assert stats.new == 0


def test_edited_file_is_rehashed_and_gets_a_new_id(workspace):
    reg = Registry(workspace)
    reg.scan()
    old = reg.list(prefix="pallet_02")[0].id

    Image.new("RGB", (800, 600), (99, 99, 99)).save(workspace / "pallet_02.png")
    stats = reg.scan()
    assert stats.hashed == 1
    assert reg.list(prefix="pallet_02")[0].id != old


def test_removed_file_drops_out(workspace):
    reg = Registry(workspace)
    reg.scan()
    (workspace / "pallet_01.jpg").unlink()
    stats = reg.scan()
    assert stats.removed == 1
    assert not reg.list(prefix="pallet_01")


def test_corrupt_file_is_reported_not_fatal(workspace):
    (workspace / "broken.png").write_bytes(b"not a png")
    reg = Registry(workspace)
    stats = reg.scan()
    assert len(stats.failed) == 1 and "broken.png" in stats.failed[0]
    assert len(reg.list()) == 2  # the good assets still indexed


def test_save_is_atomic_and_round_trips(workspace):
    reg = Registry(workspace)
    reg.scan()
    reg.save()
    assert json.loads(reg.state_path.read_text())["version"] == 1

    back = Registry.load(workspace)
    assert {a.id: a for a in back.list()} == {a.id: a for a in reg.list()}


def test_unknown_asset_says_what_to_do(workspace):
    reg = Registry(workspace)
    reg.scan()
    with pytest.raises(KeyError, match="visru index"):
        reg.get("im_deadbeef0000")


# --- video ---------------------------------------------------------------
def test_video_is_indexed_with_timing_metadata(workspace, make_video):
    clip = make_video(frames=30, fps=10)
    (workspace / clip.name).write_bytes(clip.read_bytes())
    reg = Registry(workspace)
    reg.scan()
    vid = reg.list(kind="video")[0]
    assert vid.id.startswith("vid_")
    assert vid.fps == pytest.approx(10.0)
    assert vid.frames == 30
    assert vid.duration == pytest.approx(3.0, abs=0.2)
    assert "30 frames" in vid.summary()


def test_frame_ref_decodes_the_right_frame(workspace, make_video):
    clip = make_video(frames=30, fps=10)
    (workspace / clip.name).write_bytes(clip.read_bytes())
    reg = Registry(workspace)
    reg.scan()
    vid_id = reg.list(kind="video")[0].id

    for index in (0, 7, 29):
        img = reg.load_image(f"{vid_id}#{index}")
        mean = ImageStat.Stat(img.convert("L")).mean[0]
        assert mean == pytest.approx(8 * index, abs=12), f"frame {index} decoded as {mean}"


def test_frame_refs_are_checked(workspace, make_video):
    clip = make_video(frames=30, fps=10)
    (workspace / clip.name).write_bytes(clip.read_bytes())
    reg = Registry(workspace)
    reg.scan()
    vid_id = reg.list(kind="video")[0].id
    im_id = reg.list(kind="image")[0].id

    with pytest.raises(ValueError, match="address a frame"):
        reg.load_image(vid_id)
    with pytest.raises(ValueError, match="no frames"):
        reg.load_image(f"{im_id}#3")
    with pytest.raises(IndexError):
        reg.load_image(f"{vid_id}#999")
    with pytest.raises(ValueError, match="bad frame index"):
        parse_ref("vid_abc#later")


def test_frame_times_are_interval_midpoints_inside_the_range(make_video):
    meta = video.probe(make_video(frames=30, fps=10))
    times = video.frame_times(meta, n=4, t0=1.0, t1=3.0)
    assert times == pytest.approx([1.25, 1.75, 2.25, 2.75])
    assert all(1.0 <= t <= 3.0 for t in times)

    whole = video.frame_times(meta, n=3)
    assert len(whole) == 3 and all(0 <= t <= meta.duration for t in whole)
    with pytest.raises(ValueError):
        video.frame_times(meta, n=0)


def test_decode_at_returns_indices_matching_content(make_video):
    path = make_video(frames=30, fps=10)
    meta = video.probe(path)
    frames = video.decode_at(path, [0.55, 2.05], meta)
    assert [f.index for f in frames] == [6, 21]
    for f in frames:
        mean = ImageStat.Stat(f.image.convert("L")).mean[0]
        assert mean == pytest.approx(8 * f.index, abs=12)


# --- artifacts -----------------------------------------------------------
def test_artifact_save_keeps_provenance(tmp_path):
    store = ArtifactStore(tmp_path)
    tf = ViewTransform("im_abc123", 812.0, 402.0, 1.75)
    art = store.save(Image.new("RGB", (40, 30)), kind="view", parent="im_abc123",
                     params={"box": [812, 402, 1240, 690], "grid": True}, transform=tf)

    assert art.id == "a_0001"
    assert (tmp_path / art.path).is_file()
    back = store.load("a_0001")
    assert back == art
    assert back.view_transform == tf
    assert back.params["grid"] is True
    assert store.load_image("a_0001").size == (40, 30)


def test_artifact_ids_continue_across_store_instances(tmp_path):
    first = ArtifactStore(tmp_path)
    first.save(Image.new("RGB", (8, 8)), kind="view")
    first.save(Image.new("RGB", (8, 8)), kind="view")
    # A fresh process must not reuse a_0001 and overwrite evidence.
    assert ArtifactStore(tmp_path).save(Image.new("RGB", (8, 8)), kind="view").id == "a_0003"


def test_marks_round_trip_by_id(tmp_path):
    store = ArtifactStore(tmp_path)
    marks = [Mark(id=1, label="mug handle", box=(1020.0, 560.0, 1100.0, 640.0), score=0.91),
             Mark(id=2, label="gripper", box=(400.0, 300.0, 480.0, 380.0), score=0.77)]
    art = store.save(Image.new("RGB", (40, 30)), kind="detect", parent="im_abc123",
                     marks=marks)

    back = store.load(art.id)
    assert back.mark(2).label == "gripper"
    assert back.mark(1).box == (1020.0, 560.0, 1100.0, 640.0)
    with pytest.raises(KeyError, match="no mark 3"):
        back.mark(3)


def test_view_coordinates_map_back_to_source_through_the_artifact(tmp_path):
    store = ArtifactStore(tmp_path)
    tf = ViewTransform("im_abc123", 812.0, 402.0, 2.0)
    art = store.save(Image.new("RGB", (40, 30)), kind="view", transform=tf)
    assert store.to_source(art.id, 20.0, 10.0) == pytest.approx((822.0, 407.0))

    plain = store.save(Image.new("RGB", (8, 8)), kind="plot")
    with pytest.raises(ValueError, match="no transform"):
        store.to_source(plain.id, 1.0, 1.0)


def test_unknown_artifact_raises(tmp_path):
    with pytest.raises(KeyError):
        ArtifactStore(tmp_path).load("a_9999")
