import asyncio

import numpy as np
import pytest
from PIL import Image, ImageStat

from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from visru.servers.vision_mcp import build_server
from visru.tools import open_workspace
from visru.tools.annotate import _nice_step
from visru.tools.navigate import list_assets, sample_frames, view
from visru.workspace.registry import Registry

from conftest import MARKER_BOX

MARKER_CENTRE = ((MARKER_BOX[0] + MARKER_BOX[2]) / 2, (MARKER_BOX[1] + MARKER_BOX[3]) / 2)


@pytest.fixture
def ctx(marker_workspace):
    reg = Registry(marker_workspace)
    reg.scan()
    reg.save()
    return open_workspace(marker_workspace)


def _ids(ctx):
    return (ctx.registry.list(kind="image")[0].id, ctx.registry.list(kind="video")[0].id)


def _red_centroid(img: Image.Image) -> tuple[float, float]:
    """Centre of the red marker in view pixels."""
    a = np.asarray(img.convert("RGB")).astype(int)
    hit = (a[..., 0] > 200) & (a[..., 1] < 90) & (a[..., 2] < 90)
    assert hit.any(), "marker not visible in this view"
    ys, xs = np.nonzero(hit)
    return float(xs.mean()), float(ys.mean())


# --- view ----------------------------------------------------------------
def test_view_transform_maps_the_marker_back_to_its_source_pixels(ctx):
    """The whole point of the tool layer: what the model sees, located in the source."""
    im_id, _ = _ids(ctx)
    res = view(ctx, im_id, box=[2800, 1800, 3200, 2200])
    art = ctx.artifacts.load(res.artifacts[0])

    u, v = _red_centroid(res.images[0])
    x, y = art.view_transform.to_source(u, v)
    assert (x, y) == pytest.approx(MARKER_CENTRE, abs=1.0)


def test_view_survives_nested_zoom(ctx):
    """Zooming into a view's own coordinates must not drift the marker."""
    im_id, _ = _ids(ctx)
    first = view(ctx, im_id, box=[2000, 1000, 4000, 3000])
    tf = ctx.artifacts.load(first.artifacts[0]).view_transform
    u, v = _red_centroid(first.images[0])
    x, y = tf.to_source(u, v)

    second = view(ctx, im_id, box=[int(x) - 60, int(y) - 60, int(x) + 60, int(y) + 60])
    tf2 = ctx.artifacts.load(second.artifacts[0]).view_transform
    u2, v2 = _red_centroid(second.images[0])
    assert tf2.to_source(u2, v2) == pytest.approx(MARKER_CENTRE, abs=1.0)


def test_view_renders_at_the_scale_it_reports(ctx):
    im_id, _ = _ids(ctx)
    res = view(ctx, im_id, box=[2800, 1800, 3200, 2200])
    art = ctx.artifacts.load(res.artifacts[0])
    scale = art.view_transform.scale
    assert res.images[0].size == (round(400 * scale), round(400 * scale))
    assert f"scale={scale:.3f}" in res.text
    assert max(res.images[0].size) <= ctx.cfg.model.max_edge


def test_view_warns_when_below_inspected_scale(ctx):
    im_id, _ = _ids(ctx)
    whole = view(ctx, im_id)  # 4032x3024 squeezed to 2000 px: 0.5x
    assert "BELOW inspected_scale" in whole.text

    close = view(ctx, im_id, box=[2900, 1900, 3100, 2100])
    assert "at or above inspected_scale" in close.text


def test_view_clamps_and_rejects_bad_boxes(ctx):
    im_id, _ = _ids(ctx)
    res = view(ctx, im_id, box=[-500, -500, 99999, 99999])
    assert ctx.artifacts.load(res.artifacts[0]).params["box"] == [0, 0, 4032, 3024]

    with pytest.raises(ValueError, match="empty"):
        view(ctx, im_id, box=[100, 100, 100, 200])
    with pytest.raises(ValueError, match="x0, y0, x1, y1"):
        view(ctx, im_id, box=[1, 2, 3])


def test_view_records_provenance(ctx):
    im_id, _ = _ids(ctx)
    res = view(ctx, im_id, box=[2800, 1800, 3200, 2200], grid=True)
    art = ctx.artifacts.load(res.artifacts[0])
    assert art.kind == "view" and art.parent == im_id
    assert art.params == {"box": [2800, 1800, 3200, 2200], "grid": True}
    assert art.id in res.text
    # The saved PNG is the image the model was handed, not a re-render.
    assert ctx.artifacts.load_image(art.id).size == res.images[0].size


def test_view_works_on_a_video_frame_ref(ctx):
    _, vid_id = _ids(ctx)
    res = view(ctx, f"{vid_id}#45", box=[0, 0, 160, 120])
    art = ctx.artifacts.load(res.artifacts[0])
    assert art.parent == f"{vid_id}#45"
    assert art.view_transform.asset_id == f"{vid_id}#45"
    assert ImageStat.Stat(res.images[0].convert("L")).mean[0] == pytest.approx(90, abs=12)


# --- grid ----------------------------------------------------------------
def test_grid_labels_do_not_move_the_pixels(ctx):
    im_id, _ = _ids(ctx)
    box = [2800, 1800, 3200, 2200]
    plain = view(ctx, im_id, box=box)
    ruled = view(ctx, im_id, box=box, grid=True)

    assert ruled.images[0].size == plain.images[0].size
    assert ruled.images[0].tobytes() != plain.images[0].tobytes()
    # A ruler that shifts what it measures is worse than no ruler.
    tf = ctx.artifacts.load(ruled.artifacts[0]).view_transform
    assert tf.to_source(*_red_centroid(ruled.images[0])) == pytest.approx(
        MARKER_CENTRE, abs=1.0)


def test_nice_step_rounds_to_readable_numbers():
    assert [_nice_step(r) for r in (0.4, 1, 3, 7, 11, 45, 99, 140, 3000)] == [
        1, 1, 5, 10, 20, 50, 100, 200, 5000]


# --- sample_frames -------------------------------------------------------
def test_sample_frames_sheet_lists_refs_that_view_can_open(ctx):
    _, vid_id = _ids(ctx)
    res = sample_frames(ctx, vid_id, n=6)
    art = ctx.artifacts.load(res.artifacts[0])

    assert art.kind == "contact_sheet" and art.transform is None
    cells = art.params["cells"]
    assert len(cells) == 6
    assert all(c["ref"].startswith(f"{vid_id}#") for c in cells)
    assert all(c["ref"] in res.text for c in cells)

    # Each ref resolves to the frame the sheet claims: frame k is gray 2*k.
    for cell in cells:
        frame = view(ctx, cell["ref"])
        mean = ImageStat.Stat(frame.images[0].convert("L")).mean[0]
        assert mean == pytest.approx(2 * cell["index"], abs=12), cell


def test_sample_frames_honours_a_time_window(ctx):
    _, vid_id = _ids(ctx)
    res = sample_frames(ctx, vid_id, n=4, t0=1.0, t1=2.0)
    cells = ctx.artifacts.load(res.artifacts[0]).params["cells"]
    assert len(cells) == 4
    assert all(1.0 <= c["time"] <= 2.0 for c in cells)
    assert all(30 <= c["index"] <= 60 for c in cells)


def test_sample_frames_stays_inside_the_model_limits(ctx):
    _, vid_id = _ids(ctx)
    sheet = sample_frames(ctx, vid_id, n=24).images[0]
    assert max(sheet.size) <= ctx.cfg.model.max_edge


def test_sample_frames_rejects_images(ctx):
    im_id, _ = _ids(ctx)
    with pytest.raises(ValueError, match="not a video"):
        sample_frames(ctx, im_id, n=4)


# --- list_assets ---------------------------------------------------------
def test_list_assets_filters_and_truncates(ctx):
    im_id, vid_id = _ids(ctx)
    assert im_id in list_assets(ctx).text
    assert vid_id not in list_assets(ctx, kind="image").text
    assert "no assets match" in list_assets(ctx, prefix="nothing").text
    assert "1 more" in list_assets(ctx, limit=1).text


# --- MCP server ----------------------------------------------------------
def test_server_exposes_the_tools(ctx):
    tools = asyncio.run(build_server(ctx).list_tools())
    assert {t.name for t in tools} == {"list_assets", "view", "sample_frames",
                                       "detect", "segment", "ocr", "overlay",
                                       "cross_check", "coverage", "submit_report", "search",
                                       "track"}
    by_name = {t.name: t.description for t in tools}
    # Every tool that takes coordinates must say whose pixels they are.
    for name in ("view", "detect", "segment", "overlay"):
        assert "SOURCE-ASSET pixels" in by_name[name], name
    assert "NOT an independent check" in by_name["segment"]
    # The model has to know the ledger will refuse before it writes its claims.
    assert "cross_check" in by_name["submit_report"]


def test_server_returns_an_image_and_the_artifact_id(ctx):
    im_id, _ = _ids(ctx)
    result = asyncio.run(build_server(ctx).call_tool(
        "view", {"asset_id": im_id, "box": [2800, 1800, 3200, 2200]}))
    assert not result.is_error
    assert [b.type for b in result.content] == ["image", "text"]
    text = " ".join(b.text for b in result.content if b.type == "text")
    assert "artifact=a_" in text and "scale=" in text


# --- staleness and error reporting (found in the first live session) ------
def test_registry_reloads_when_index_runs_while_the_server_is_up(ctx, marker_workspace):
    """A boot-time snapshot would report an empty workspace for the server's life."""
    before = {a.id for a in ctx.registry.list()}

    Image.new("RGB", (640, 480), (7, 7, 7)).save(marker_workspace / "late_arrival.png")
    fresh = Registry.load(marker_workspace)
    fresh.scan()
    fresh.save()

    after = ctx.registry.list(prefix="late_arrival")
    assert after and after[0].id not in before
    assert view(ctx, after[0].id).images[0].size == (1280, 960)


def test_registry_is_not_reloaded_when_nothing_changed(ctx):
    assert ctx.registry is ctx.registry  # stat only; no reload per call


def test_server_started_before_the_first_index_sees_it_appear(marker_workspace):
    cold = open_workspace(marker_workspace)  # nothing indexed yet
    assert cold.registry.assets == {}

    reg = Registry(marker_workspace)
    reg.scan()
    reg.save()
    assert len(cold.registry.assets) == 2


def test_bad_input_reaches_the_model_with_its_real_message(ctx):
    """ToolError text survives to the client; a crash is reduced to "Error executing
    tool view", which the agent cannot act on. Bad input must be the former."""
    im_id, _ = _ids(ctx)
    srv = build_server(ctx)

    with pytest.raises(ToolError, match=r"x0 < x1"):
        asyncio.run(srv.call_tool("view", {"asset_id": im_id,
                                           "box": [1100, 700, 900, 500]}))
    with pytest.raises(ToolError, match="visru index"):
        asyncio.run(srv.call_tool("view", {"asset_id": "im_nosuchasset"}))
    with pytest.raises(ToolError, match="not a video"):
        asyncio.run(srv.call_tool("sample_frames", {"asset_id": im_id, "n": 4}))
    with pytest.raises(ToolError, match="address a frame"):
        asyncio.run(srv.call_tool("view", {"asset_id": _ids(ctx)[1]}))

    # UnexpectedToolError is the generic one; bad input must never land there.
    with pytest.raises(ToolError) as caught:
        asyncio.run(srv.call_tool("view", {"asset_id": im_id, "box": [1, 2, 3]}))
    assert not isinstance(caught.value, UnexpectedToolError)


# --- artifact refs (Phase 1: elision stubs must be redeemable) -----------
def test_view_takes_an_artifact_id_and_returns_the_same_pixels(ctx):
    """The context manager elides an image to a stub naming its artifact ID. That
    stub is a promise the model can have the image back; here it is kept."""
    im_id, _ = _ids(ctx)
    first = view(ctx, im_id, box=[2800, 1800, 3200, 2200])
    again = view(ctx, first.artifacts[0])

    assert again.images[0].tobytes() == first.images[0].tobytes()
    # Re-derived from the parent asset, so the transform still lands in source px.
    assert f"box=[2800, 1800, 3200, 2200]" in again.text


def test_an_explicit_box_still_wins_over_the_artifacts_own(ctx):
    """Every box the model sends is source-asset pixels, whatever ref it names."""
    im_id, _ = _ids(ctx)
    art = view(ctx, im_id, box=[2800, 1800, 3200, 2200]).artifacts[0]
    zoomed = view(ctx, art, box=[2900, 1900, 3000, 2000])
    assert "box=[2900, 1900, 3000, 2000]" in zoomed.text


def test_an_artifact_chain_resolves_back_to_the_asset(ctx):
    im_id, _ = _ids(ctx)
    a = view(ctx, im_id, box=[2800, 1800, 3200, 2200]).artifacts[0]
    b = view(ctx, a).artifacts[0]
    assert view(ctx, b).images[0].tobytes() == view(ctx, im_id,
                                                    box=[2800, 1800, 3200, 2200]).images[0].tobytes()
    # Lineage survives: the ledger must not call a re-view an independent check.
    assert ctx.artifacts.load(b).parent == a


def test_an_artifact_with_no_parent_says_so(ctx):
    orphan = ctx.artifacts.save(Image.new("RGB", (8, 8)), kind="view")
    with pytest.raises(ValueError, match="pass an asset ID instead"):
        view(ctx, orphan.id)


# --- tool registry -------------------------------------------------------
def test_server_signatures_match_the_registry():
    """MCP needs real signatures, the loop needs the table; they must agree."""
    from visru.tools import registry

    # name no single asset to work on
    assetless = {"list_assets", "submit_report", "cross_check", "coverage", "search"}
    for spec in registry.SPECS:
        if spec.name not in assetless:
            assert spec.params[0].name == "asset_id", spec.name

    srv = build_server(open_workspace("."))
    tools = {t.name: t for t in asyncio.run(srv.list_tools())}
    assert set(tools) == set(registry.TOOLS) - registry.HARNESS_ONLY
    for name, spec in registry.TOOLS.items():
        if name in registry.HARNESS_ONLY:
            continue
        declared = [p.name for p in spec.params]
        exposed = list(tools[name].input_schema.get("properties", {}))
        assert declared == exposed, f"{name}: registry {declared} != MCP {exposed}"


def test_registry_rejects_calls_the_model_can_fix(ctx):
    from visru.tools import registry

    im_id, _ = _ids(ctx)
    with pytest.raises(KeyError, match="no tool named"):
        registry.execute(ctx, "zoom", {})
    with pytest.raises(ValueError, match="unknown argument"):
        registry.execute(ctx, "view", {"asset_id": im_id, "bbox": [0, 0, 9, 9]})
    with pytest.raises(ValueError, match="needs asset_id"):
        registry.execute(ctx, "view", {})


def test_registry_defaults_match_calling_the_tool_directly(ctx):
    from visru.tools import registry

    im_id, _ = _ids(ctx)
    through = registry.execute(ctx, "view", {"asset_id": im_id})
    assert through.images[0].size == view(ctx, im_id).images[0].size


def test_artifact_ids_skip_past_another_process_writes(ctx, marker_workspace):
    from visru.workspace.artifacts import ArtifactStore

    im_id, _ = _ids(ctx)
    first = view(ctx, im_id).artifacts[0]
    other = ArtifactStore(marker_workspace)  # a second process, same workspace
    other.save(Image.new("RGB", (8, 8)), kind="view")
    second = view(ctx, im_id).artifacts[0]

    assert len({first, second, "a_0002"}) == 3  # nothing overwrote anything
    assert ctx.artifacts.load(first).kind == "view"
