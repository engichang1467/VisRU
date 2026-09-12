"""Search tools: find things by text, by prompt, or by reading text off the pixels.

Every model here sees a *rendered* crop, so every coordinate it returns is in view
pixels and must come back through the ViewTransform before anyone stores it. Marks
are the model's handle on a result: it says "mark 3", never a pixel.
"""

from PIL import Image

from visru import models
from visru.tools import ToolContext, ToolResult, scale_note
from visru.tools.annotate import draw_marks
from visru.tools.navigate import render_region
from visru.workspace.artifacts import Mark


def _to_source_box(tf, box) -> tuple[float, float, float, float]:
    x0, y0 = tf.to_source(box[0], box[1])
    x1, y1 = tf.to_source(box[2], box[3])
    # float(): numpy scalars survive json (they subclass float) but print as
    # "np.float64(239.5)" in the text the model reads.
    return tuple(round(float(v), 1) for v in (x0, y0, x1, y1))


def _mark_lines(marks: list[Mark]) -> str:
    return "\n".join(
        f"  mark {m.id}: {m.label}"
        + (f" score={m.score:.2f}" if m.score is not None else "")
        + (f" box={list(m.box)}" if m.box else "")
        for m in marks)


def detect(ctx: ToolContext, ref: str, queries: list[str] | str,
           box: list[int] | None = None, threshold: float = 0.1,
           max_marks: int = 20) -> ToolResult:
    """Open-vocabulary detection. Boxes come back as numbered marks in source pixels."""
    import torch

    queries = [queries] if isinstance(queries, str) else list(queries)
    if not queries:
        raise ValueError("detect needs at least one text query")

    img, tf, region = render_region(ctx, ref, box)
    processor, model = models.detector()
    inputs = processor(text=[queries], images=img, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    # transformers 5.x undoes OWLv2's square padding inside post-processing, so the
    # image's own (h, w) is the right target size. Verified on 2:1, 1:2 and 1:1.
    results = processor.post_process_grounded_object_detection(
        outputs, threshold=threshold,
        target_sizes=torch.tensor([(img.height, img.width)]),
        text_labels=[queries])[0]

    order = results["scores"].argsort(descending=True)[:max_marks]
    marks = [
        Mark(id=i + 1,
             label=results["text_labels"][idx],
             box=_to_source_box(tf, results["boxes"][idx].tolist()),
             score=round(float(results["scores"][idx]), 3))
        for i, idx in enumerate(order)
    ]

    out = draw_marks(img, marks, tf) if marks else img
    art = ctx.artifacts.save(out, kind="detect", parent=ref, transform=tf, marks=marks,
                             params={"queries": queries, "box": list(region),
                                     "threshold": threshold})
    if marks:
        text = (f"artifact={art.id} source={ref} {len(marks)} marks for "
                f"{queries} in box={list(region)} at {tf.scale:.2f}x source scale — "
                f"{scale_note(ctx.cfg, tf.scale)} (boxes in source pixels):\n"
                f"{_mark_lines(marks)}\n"
                f"Refer to these as 'mark N'. Pass them to segment with "
                f"from_artifact={art.id!r}.")
    else:
        text = (f"artifact={art.id} source={ref} no detections for {queries} above "
                f"threshold={threshold} in box={list(region)}. Try a lower threshold, a "
                f"different wording, or a tighter box rendered at higher scale "
                f"(this render was {tf.scale:.2f}x source).")
    return ToolResult(text=text, images=[out], artifacts=[art.id])


def _prompt_boxes(ctx: ToolContext, from_artifact: str | None, marks: list[int] | None,
                  boxes: list[list[float]] | None) -> list[tuple[str, list[float]]]:
    """Resolve box prompts, either given directly or pulled from an artifact's marks."""
    out: list[tuple[str, list[float]]] = []
    if from_artifact:
        art = ctx.artifacts.load(from_artifact)
        wanted = marks or [m["id"] for m in art.marks]
        for mark_id in wanted:
            mark = art.mark(mark_id)
            if mark.box is None:
                raise ValueError(f"mark {mark_id} of {from_artifact} has no box to segment")
            out.append((f"{from_artifact} mark {mark_id}: {mark.label}", list(mark.box)))
    for b in boxes or []:
        if len(b) != 4:
            raise ValueError(f"each box must be [x0, y0, x1, y1] in source pixels, got {b}")
        out.append(("box prompt", list(b)))
    return out


def segment(ctx: ToolContext, ref: str, points: list[list[float]] | None = None,
            labels: list[int] | None = None, boxes: list[list[float]] | None = None,
            from_artifact: str | None = None, marks: list[int] | None = None,
            box: list[int] | None = None) -> ToolResult:
    """Mask objects from point, box, or mark prompts. All coordinates are source pixels.

    `box` is the region to render and segment within; the prompts themselves are
    always in source-asset pixels, wherever that region starts.
    """
    import torch

    box_prompts = _prompt_boxes(ctx, from_artifact, marks, boxes)
    if not points and not box_prompts:
        raise ValueError(
            "segment needs points=[[x, y], ...], boxes=[[x0, y0, x1, y1], ...], or "
            "from_artifact='a_0005' with marks=[1, 2] — all in source pixels")
    if points and box_prompts:
        raise ValueError("give either points or boxes/marks, not both")

    img, tf, region = render_region(ctx, ref, box)
    processor, model = models.segmenter()

    if points:
        labels = labels or [1] * len(points)
        if len(labels) != len(points):
            raise ValueError(f"{len(points)} points but {len(labels)} labels")
        # One object per point: the model asked about N things, not one N-point thing.
        view_points = [[list(tf.to_view(p[0], p[1]))] for p in points]
        kwargs = {"input_points": [view_points], "input_labels": [[[v] for v in labels]]}
        names = [f"point ({p[0]:.0f}, {p[1]:.0f})" for p in points]
    else:
        names = [name for name, _ in box_prompts]
        view_boxes = [list(tf.to_view(b[0], b[1])) + list(tf.to_view(b[2], b[3]))
                      for _, b in box_prompts]
        kwargs = {"input_boxes": [view_boxes]}

    inputs = processor(images=img, return_tensors="pt", **kwargs).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0]
    scores = outputs.iou_scores[0].squeeze(-1).tolist()
    if isinstance(scores, float):
        scores = [scores]

    made: list[Mark] = []
    images: dict[int, Image.Image] = {}
    for i, name in enumerate(names):
        binary = masks[i, 0].cpu().numpy()
        mask_img = Image.fromarray((binary * 255).astype("uint8"), mode="L")
        ys, xs = binary.nonzero()
        if len(xs) == 0:
            made.append(Mark(id=i + 1, label=f"{name} (empty mask)", score=round(scores[i], 3)))
            continue
        source_box = _to_source_box(tf, [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        made.append(Mark(id=i + 1, label=name, box=source_box, score=round(scores[i], 3)))
        images[i + 1] = mask_img

    out = draw_marks(img, made, tf, masks=images)
    # Masks are stored in this artifact's view frame; its transform places them.
    art = ctx.artifacts.save(out, kind="segment", parent=ref, transform=tf, marks=made,
                             mask_images=images,
                             params={"box": list(region), "points": points,
                                     "labels": labels, "boxes": boxes,
                                     "from_artifact": from_artifact, "marks": marks})
    text = (f"artifact={art.id} source={ref} {len(made)} masks in box={list(region)} "
            f"(boxes in source pixels):\n{_mark_lines(made)}\n"
            f"Masks are saved beside the artifact; overlay them to re-check a claim.")
    return ToolResult(text=text, images=[out], artifacts=[art.id])


def ocr(ctx: ToolContext, ref: str, box: list[int] | None = None,
        min_confidence: float = 0.5) -> ToolResult:
    """Read text, returning each line as a numbered mark with a source-pixel box."""
    import numpy as np

    img, tf, region = render_region(ctx, ref, box)
    engine = models.reader()
    # PaddleOCR takes ndarray or a path, never PIL, and reads ndarrays as BGR. Handed
    # a PIL image it logs a warning and returns [], which reads as "no text found".
    bgr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
    raw = engine.predict(bgr) if hasattr(engine, "predict") else engine.ocr(bgr, cls=True)

    marks: list[Mark] = []
    for quad, text, score in _paddle_lines(raw):
        if score < min_confidence:
            continue
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        marks.append(Mark(id=len(marks) + 1, label=text,
                          box=_to_source_box(tf, [min(xs), min(ys), max(xs), max(ys)]),
                          score=round(float(score), 3)))

    out = draw_marks(img, marks, tf) if marks else img
    art = ctx.artifacts.save(out, kind="ocr", parent=ref, transform=tf, marks=marks,
                             params={"box": list(region), "min_confidence": min_confidence})
    body = _mark_lines(marks) if marks else "  (no text above min_confidence)"
    text = (f"artifact={art.id} source={ref} {len(marks)} text lines in "
            f"box={list(region)} at {tf.scale:.2f}x source scale — "
            f"{scale_note(ctx.cfg, tf.scale)}.\n{body}")
    return ToolResult(text=text, images=[out], artifacts=[art.id])


def _paddle_lines(raw):
    """Normalise PaddleOCR output across its 2.x and 3.x result shapes."""
    if not raw:
        return
    first = raw[0]
    if isinstance(first, dict):  # 3.x: {'rec_texts': [...], 'rec_scores': [...], ...}
        for page in raw:
            polys = page.get("rec_polys") or page.get("dt_polys") or []
            for quad, text, score in zip(polys, page.get("rec_texts", []),
                                         page.get("rec_scores", [])):
                yield [list(p) for p in quad], text, score
        return
    for line in (first or []):  # 2.x: [[[quad, (text, score)], ...]]
        quad, (text, score) = line[0], line[1]
        yield [list(p) for p in quad], text, score
