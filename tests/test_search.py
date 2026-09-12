"""detect / segment / ocr.

The models are stubbed with ones that genuinely find the red marker in whatever
image they are handed. That keeps the test about the only thing that can silently
break — view pixels coming back as source pixels — without downloading weights.
Real weights run under `pytest --run-models`.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from visru import models
from visru.tools import open_workspace
from visru.tools.navigate import view
from visru.tools.search import detect, ocr, segment
from visru.workspace.registry import Registry

from conftest import MARKER_BOX

MARKER_CENTRE = ((MARKER_BOX[0] + MARKER_BOX[2]) / 2, (MARKER_BOX[1] + MARKER_BOX[3]) / 2)


def _red_mask(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB")).astype(int)
    return (a[..., 0] > 200) & (a[..., 1] < 90) & (a[..., 2] < 90)


class _Inputs(dict):
    def to(self, _device):
        return self


class _StubDetectProcessor:
    """Returns the red marker's bounding box, in the view pixels it was given."""

    def __call__(self, text=None, images=None, return_tensors=None):
        self.queries = text[0]
        ys, xs = _red_mask(images).nonzero()
        self.box = ([float(xs.min()), float(ys.min()), float(xs.max() + 1),
                     float(ys.max() + 1)] if len(xs) else None)
        return _Inputs(pixel_values=torch.zeros(1))

    def post_process_grounded_object_detection(self, outputs, threshold, target_sizes,
                                               text_labels):
        if self.box is None or threshold > 0.9:
            return [{"boxes": torch.zeros(0, 4), "scores": torch.zeros(0),
                     "text_labels": []}]
        return [{"boxes": torch.tensor([self.box]), "scores": torch.tensor([0.87]),
                 "text_labels": [self.queries[0]]}]


class _StubSegmentProcessor:
    def __call__(self, images=None, return_tensors=None, **kwargs):
        self.mask = _red_mask(images)
        self.n = len(kwargs.get("input_points", [[]])[0] or kwargs.get("input_boxes", [[]])[0])
        return _Inputs(original_sizes=torch.tensor([[images.height, images.width]]))

    def post_process_masks(self, pred_masks, original_sizes):
        one = torch.from_numpy(self.mask)[None]  # (1, H, W): the same object each time
        return [one.repeat(self.n, 1, 1)[:, None]]


class _StubModel:
    device = "cpu"

    def __call__(self, **inputs):
        n = inputs.get("original_sizes", torch.zeros(1, 2)).shape[0]
        return type("Out", (), {"pred_masks": torch.zeros(n, 1, 1, 4, 4),
                                "iou_scores": torch.full((1, 4, 1), 0.93)})()


@pytest.fixture
def ctx(marker_workspace, monkeypatch):
    reg = Registry(marker_workspace)
    reg.scan()
    reg.save()
    monkeypatch.setattr(models, "detector", lambda: (_StubDetectProcessor(), _StubModel()))
    monkeypatch.setattr(models, "segmenter", lambda: (_StubSegmentProcessor(), _StubModel()))
    return open_workspace(marker_workspace)


def _im(ctx):
    return ctx.registry.list(kind="image")[0].id


# --- detect --------------------------------------------------------------
def test_detect_reports_boxes_in_source_pixels(ctx):
    res = detect(ctx, _im(ctx), ["a red square"])
    art = ctx.artifacts.load(res.artifacts[0])
    assert art.kind == "detect" and len(art.marks) == 1

    mark = art.mark(1)
    assert mark.box == pytest.approx(MARKER_BOX, abs=2.0)
    assert mark.label == "a red square" and mark.score == 0.87


def test_detect_in_a_crop_agrees_with_detect_on_the_whole_image(ctx):
    """The same object, found at two zoom levels, must land on the same pixels.
    This is where a forgotten crop offset shows up."""
    whole = ctx.artifacts.load(detect(ctx, _im(ctx), ["x"]).artifacts[0]).mark(1)
    zoomed = ctx.artifacts.load(
        detect(ctx, _im(ctx), ["x"], box=[2900, 1900, 3200, 2200]).artifacts[0]).mark(1)
    assert zoomed.box == pytest.approx(whole.box, abs=2.0)
    assert zoomed.box == pytest.approx(MARKER_BOX, abs=2.0)


def test_detect_marks_are_numbered_and_named_in_the_text(ctx):
    res = detect(ctx, _im(ctx), ["a red square"])
    assert "mark 1: a red square" in res.text
    assert f"from_artifact='{res.artifacts[0]}'" in res.text
    assert res.images[0].size == view(ctx, _im(ctx)).images[0].size


def test_detect_with_no_hits_says_what_to_try(ctx):
    res = detect(ctx, _im(ctx), ["a unicorn"], threshold=0.95)
    assert ctx.artifacts.load(res.artifacts[0]).marks == []
    assert "no detections" in res.text and "lower threshold" in res.text


def test_detect_needs_a_query(ctx):
    with pytest.raises(ValueError, match="at least one text query"):
        detect(ctx, _im(ctx), [])


# --- segment -------------------------------------------------------------
def test_segment_from_a_point_returns_a_source_box_and_a_mask(ctx):
    res = segment(ctx, _im(ctx), points=[list(MARKER_CENTRE)])
    art = ctx.artifacts.load(res.artifacts[0])
    mark = art.mark(1)

    assert art.kind == "segment"
    assert mark.box == pytest.approx(MARKER_BOX, abs=2.0)
    assert mark.mask_path and ctx.artifacts.load_mask(art.id, 1).size == res.images[0].size


def test_segment_in_a_crop_agrees_with_the_whole_image(ctx):
    whole = ctx.artifacts.load(
        segment(ctx, _im(ctx), points=[list(MARKER_CENTRE)]).artifacts[0]).mark(1)
    zoomed = ctx.artifacts.load(
        segment(ctx, _im(ctx), points=[list(MARKER_CENTRE)],
                box=[2900, 1900, 3200, 2200]).artifacts[0]).mark(1)
    assert zoomed.box == pytest.approx(whole.box, abs=2.0)


def test_segment_can_take_marks_from_a_detection(ctx):
    found = detect(ctx, _im(ctx), ["a red square"]).artifacts[0]
    res = segment(ctx, _im(ctx), from_artifact=found, marks=[1])
    art = ctx.artifacts.load(res.artifacts[0])

    assert art.params["from_artifact"] == found
    assert f"{found} mark 1" in art.mark(1).label
    assert art.mark(1).box == pytest.approx(MARKER_BOX, abs=2.0)


def test_segment_rejects_missing_or_conflicting_prompts(ctx):
    with pytest.raises(ValueError, match="segment needs points"):
        segment(ctx, _im(ctx))
    with pytest.raises(ValueError, match="not both"):
        segment(ctx, _im(ctx), points=[[1, 2]], boxes=[[1, 2, 3, 4]])
    with pytest.raises(ValueError, match=r"\[x0, y0, x1, y1\]"):
        segment(ctx, _im(ctx), boxes=[[1, 2, 3]])
    with pytest.raises(ValueError, match="but 2 labels"):
        segment(ctx, _im(ctx), points=[[1, 2]], labels=[1, 0])


def test_segment_reports_an_unknown_mark_clearly(ctx):
    found = detect(ctx, _im(ctx), ["a red square"]).artifacts[0]
    with pytest.raises(KeyError, match="no mark 7"):
        segment(ctx, _im(ctx), from_artifact=found, marks=[7])


# --- ocr -----------------------------------------------------------------
def test_ocr_says_how_to_install_when_the_extra_is_missing(ctx):
    """Only meaningful without the extra. With it installed, the real behaviour is
    covered by the --run-models test; running it here would drag PaddleOCR's models
    into the weights-free suite."""
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("paddleocr is installed; see test_real_ocr_* under --run-models")

    models.reader.cache_clear()
    with pytest.raises(RuntimeError, match=r"\[ocr\]"):
        ocr(ctx, _im(ctx))


# --- real weights (opt-in) -----------------------------------------------
@pytest.mark.models
def test_real_detector_finds_a_known_object_in_source_pixels(tmp_path):
    """OWLv2 for real: a disc at a known source position, found inside an offset crop."""
    root = tmp_path / "real"
    root.mkdir()
    img = Image.new("RGB", (2400, 1600), (250, 250, 250))
    from PIL import ImageDraw
    ImageDraw.Draw(img).ellipse([1500 - 70, 900 - 70, 1500 + 70, 900 + 70], fill=(220, 30, 30))
    img.save(root / "scene.png")
    reg = Registry(root)
    reg.scan()
    reg.save()
    live = open_workspace(root)
    asset = live.registry.list()[0].id

    for box in (None, [1200, 700, 1800, 1100]):
        res = detect(live, asset, ["a red circle"], box=box, threshold=0.2)
        mark = live.artifacts.load(res.artifacts[0]).mark(1)
        centre = ((mark.box[0] + mark.box[2]) / 2, (mark.box[1] + mark.box[3]) / 2)
        assert centre == pytest.approx((1500, 900), abs=15), f"box={box}"


@pytest.mark.models
def test_real_segmenter_masks_the_object_under_a_source_point(tmp_path):
    root = tmp_path / "real2"
    root.mkdir()
    img = Image.new("RGB", (2400, 1600), (250, 250, 250))
    from PIL import ImageDraw
    ImageDraw.Draw(img).ellipse([1430, 830, 1570, 970], fill=(220, 30, 30))
    img.save(root / "scene.png")
    reg = Registry(root)
    reg.scan()
    reg.save()
    live = open_workspace(root)
    asset = live.registry.list()[0].id

    res = segment(live, asset, points=[[1500, 900]])
    mark = live.artifacts.load(res.artifacts[0]).mark(1)
    assert mark.box == pytest.approx((1430, 830, 1570, 970), abs=20)


@pytest.mark.models
def test_real_ocr_reads_text_and_locates_it_in_source_pixels(tmp_path):
    """PaddleOCR for real, on text drawn at a known source position.

    Also pins the scale lesson: the same label reads exactly when zoomed and loses
    characters at 0.5x, which is why ocr repeats the inspected_scale warning.
    """
    pytest.importorskip("paddleocr")
    from PIL import ImageDraw, ImageFont

    root = tmp_path / "ocr"
    root.mkdir()
    img = Image.new("RGB", (4032, 3024), (60, 80, 100))
    draw = ImageDraw.Draw(img)
    draw.rectangle([2980, 1980, 3180, 2080], fill=(240, 240, 240))
    draw.text((2990, 2010), "LOT 44-871", fill=(10, 10, 10),
              font=ImageFont.load_default(size=34))
    img.save(root / "label.png")
    reg = Registry(root)
    reg.scan()
    reg.save()
    live = open_workspace(root)
    asset = live.registry.list()[0].id

    res = ocr(live, asset, box=[2900, 1900, 3300, 2150])
    art = live.artifacts.load(res.artifacts[0])
    assert art.kind == "ocr"
    texts = [m["label"] for m in art.marks]
    assert "LOT 44-871" in texts, texts

    mark = next(m for m in art.marks if m["label"] == "LOT 44-871")
    assert mark["box"][0] == pytest.approx(2990, abs=25)
    assert mark["box"][1] == pytest.approx(2015, abs=25)
    assert "at or above inspected_scale" in res.text
