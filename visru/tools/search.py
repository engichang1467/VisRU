"""Search tools: find things by text, by prompt, or by reading text off the pixels.

Every model here sees a *rendered* crop, so every coordinate it returns is in view
pixels and must come back through the ViewTransform before anyone stores it. Marks
are the model's handle on a result: it says "mark 3", never a pixel.
"""

from dataclasses import asdict

from PIL import Image

from visru import models
from visru.tools import ToolContext, ToolResult, resolve_ref, scale_note
from visru.tools.annotate import draw_marks
from visru.tools.navigate import build_sheet, render_region
from visru.workspace import index, video
from visru.workspace.artifacts import Mark
from visru.workspace.registry import parse_ref


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


def _point_prompts(points: list[list[float]],
                   labels: list[int] | None) -> tuple[list[int], list[str]]:
    """Labels (all foreground unless given) and one object name per point."""
    labels = labels or [1] * len(points)
    if len(labels) != len(points):
        raise ValueError(f"{len(points)} points but {len(labels)} labels")
    return labels, [f"point ({p[0]:.0f}, {p[1]:.0f})" for p in points]


def _mask_marks(names: list[str], masks, scores: list[float],
                tf) -> tuple[list[Mark], dict[int, Image.Image]]:
    """A mark per prompt, boxed in source pixels, and its mask where it is not empty."""
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
    return made, images


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
        labels, names = _point_prompts(points, labels)
        # One object per point: the model asked about N things, not one N-point thing.
        view_points = [[list(tf.to_view(p[0], p[1]))] for p in points]
        kwargs = {"input_points": [view_points], "input_labels": [[[v] for v in labels]]}
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

    made, images = _mask_marks(names, masks, scores, tf)
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


def _search_index(ctx: ToolContext, query: str, k: int):
    """The workspace's embedding index, once the query is known to be searchable."""
    if not str(query).strip():
        raise ValueError("search needs a text query, e.g. 'a person opens a door'")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    idx = index.load(ctx.root)
    if idx is None or not len(idx):
        raise ValueError("this workspace has no embedding index; run "
                         "`visru index <workspace> --embed` first")
    return idx


def _indexed_asset(ctx: ToolContext, idx, ref: str) -> str:
    asset_id, _ = parse_ref(resolve_ref(ctx, ref)[0])
    ctx.registry.get(asset_id)  # an unknown ID should say so, not return nothing
    if asset_id not in idx.assets():
        raise ValueError(f"{asset_id} is not in the embedding index; re-run "
                         "`visru index <workspace> --embed`")
    return asset_id


def _hit_lines(hits, wins: list[dict]) -> str:
    """Ranked hits, then the time windows they merge into, if any."""
    ranked = "\n".join(
        f"  {i}. {h.ref}" + (f" t={h.time:.2f}s" if h.time is not None else "")
        + f" score={h.score:.3f}" for i, h in enumerate(hits, start=1))
    if not wins:
        return ranked
    listing = "\n".join(f"  mark {i}: {w['asset']} t={w['t0']}..{w['t1']}s "
                        f"best={w['score']:.3f} ({len(w['frames'])} frames)"
                        for i, w in enumerate(wins, start=1))
    return (f"{ranked}\nTime windows, adjacent hits merged (cross_check refers to "
            f"these as marks):\n{listing}")


def search(ctx: ToolContext, query: str, ref: str | None = None, t0: float | None = None,
           t1: float | None = None, k: int = 8) -> ToolResult:
    """Rank indexed images and video frames by similarity to a text query.

    For video, hits on neighbouring samples merge into time windows — the answer to
    "when does this happen" at the index's resolution (1 frame per second by
    default), which is a place to look, not a timestamp to report.
    """
    idx = _search_index(ctx, query, k)
    asset_id = _indexed_asset(ctx, idx, ref) if ref else None

    hits = index.query(idx, query, asset_id, t0, t1, k)
    scope = asset_id or "the workspace"
    span = "" if t0 is None and t1 is None else f" within {t0 or 0}..{t1 or 'end'}s"
    if not hits:
        return ToolResult(text=f"no indexed frames in {scope}{span}")

    durations = {parse_ref(h.ref)[0]: ctx.registry.get(h.ref).duration or 0.0
                 for h in hits if h.time is not None}
    wins = index.windows(idx, hits, durations)
    text = (f"search {query!r} over {scope}{span}: {len(hits)} hits (scores are "
            f"relative; compare them within one search only)\n{_hit_lines(hits, wins)}")

    top = parse_ref(hits[0].ref)[0]
    if hits[0].time is None:  # an image ranked first: view it, there is no sheet
        return ToolResult(text=text + "\nCall view on an image ID to look at it.")

    times = sorted(h.time for h in hits if parse_ref(h.ref)[0] == top)
    sheet = build_sheet(ctx, top, times=times)
    art = ctx.artifacts.save(
        sheet.image, kind="search", parent=top,
        params={"query": query, "t0": t0, "t1": t1, "k": k, "times": times,
                "cells": sheet.cells, "hits": [asdict(h) for h in hits],
                "windows": wins, "sheet_scale": sheet.scale})
    text = (f"artifact={art.id} " + text
            + f"\nThe sheet shows {top}'s hits in time order, thumbnails at "
            f"{sheet.cell_w / sheet.source_w:.2f}x source scale. Confirm a window with "
            "sample_frames over it, then view a frame ref before claiming detail.")
    return ToolResult(text=text, images=[sheet.image], artifacts=[art.id])


TRACK_EDGE = 1024  # SAM 2 works at 1024 px; decoding frames larger only costs memory
SHEET_CELLS = 12


def _runs(indices: list[int], step: int) -> list[tuple[int, int]]:
    """Consecutive tracked frames (spaced `step` apart) collapsed to (first, last)."""
    out: list[tuple[int, int]] = []
    for i in indices:
        if out and i - out[-1][1] <= step:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def _track_frames(ctx: ToolContext, asset_id: str, start: int, t1: float | None,
                  stride: int, max_frames: int):
    """Decode the frames to track over, and the scale and size SAM 2 sees them at."""
    meta = ctx.registry.video_meta(asset_id)
    last = meta.frames - 1 if t1 is None else min(meta.frames - 1, int(t1 * meta.fps))
    if last <= start:
        raise ValueError(f"t1={t1} is not after the starting frame {start} "
                         f"(t={start / meta.fps:.2f}s)")
    wanted = list(range(start, last + 1, stride))[:max_frames]
    frames = video.decode_at(ctx.registry.abspath(asset_id),
                             [i / meta.fps for i in wanted], meta)
    s = min(1.0, TRACK_EDGE / max(meta.width, meta.height))
    size = (max(1, round(meta.width * s)), max(1, round(meta.height * s)))
    return frames, s, size


def _track_prompt(points, labels, box_prompts, s: float):
    """(labels, object names, SAM 2 prompt at scale s) from points or box prompts."""
    if points:
        labels, names = _point_prompts(points, labels)
        return labels, names, {"input_points": [[[[p[0] * s, p[1] * s]] for p in points]],
                               "input_labels": [[[v] for v in labels]]}
    names = [name for name, _ in box_prompts]
    return labels, names, {"input_boxes": [[[v * s for v in b] for _, b in box_prompts]]}


def _propagate(frames, s: float, size: tuple[int, int], prompt: dict,
               obj_ids: list[int]) -> dict[int, list[dict]]:
    """Prompt SAM 2 on the first frame; per object, its source-pixel box on each frame."""
    import numpy as np
    import torch

    images = [f.image.resize(size, Image.Resampling.BILINEAR) if s < 1 else f.image
              for f in frames]
    processor, model = models.video_segmenter()
    session = processor.init_video_session(video=images, inference_device=model.device,
                                           dtype=models.dtype())
    processor.add_inputs_to_inference_session(inference_session=session, frame_idx=0,
                                              obj_ids=list(obj_ids), **prompt)  # it mutates
    per_obj: dict[int, list] = {o: [] for o in obj_ids}
    with torch.no_grad():
        model(inference_session=session, frame_idx=0)
        for out in model.propagate_in_video_iterator(session):
            masks = processor.post_process_masks(
                [out.pred_masks], original_sizes=[[size[1], size[0]]])[0]
            frame = frames[out.frame_idx]
            for j, o in enumerate(obj_ids):
                binary = np.asarray(masks[j, 0].cpu()).astype(bool)
                ys, xs = binary.nonzero()
                box = None if len(xs) == 0 else tuple(
                    round(float(v) / s, 1)
                    for v in (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
                per_obj[o].append({"index": frame.index, "time": round(frame.time, 3),
                                   "box": box, "area": round(float(binary.mean()), 5)})
    return per_obj


def _track_summary(o: int, name: str, rows: list[dict], step: int) -> tuple[dict, str]:
    """One tracked object: its stored record, and the line the model reads."""
    rows = sorted(rows, key=lambda r: r["index"])
    present = [r["index"] for r in rows if r["box"]]
    runs = _runs(present, step)
    at = {r["index"]: r["time"] for r in rows}
    window = [at[present[0]], at[present[-1]]] if present else None
    record = {"label": name, "frames": rows, "present": window,
              "runs": [[at[a], at[b]] for a, b in runs]}
    if not present:
        return record, f"  mark {o}: {name} — lost immediately; no mask on any frame"
    gone = [r for r in rows if r["index"] > present[-1]]
    runs_text = ", ".join(f"{at[a]:.2f}..{at[b]:.2f}s" for a, b in runs)
    return record, (
        f"  mark {o}: {name} — present {runs_text}"
        + (f"; gone from {gone[0]['time']:.2f}s (frame {gone[0]['index']}) to the "
           f"end of the tracked range" if gone else "; present through the end")
        + f"; area {rows[0]['area']:.2%} → {rows[-1]['area']:.2%} of the frame")


def _sheet_boxes(frames, objects: dict[int, dict]) -> tuple[dict[int, list[Mark]], list]:
    """Up to 12 of the tracked frames, evenly spaced, first and last kept, with boxes."""
    pick = sorted({round(i * (len(frames) - 1) / (SHEET_CELLS - 1))
                   for i in range(SHEET_CELLS)}) if len(frames) > SHEET_CELLS \
        else range(len(frames))
    sheet_boxes: dict[int, list[Mark]] = {}
    for i in pick:
        idx = frames[i].index
        sheet_boxes[idx] = [Mark(id=o, label=obj["label"], box=r["box"])
                            for o, obj in objects.items() for r in obj["frames"]
                            if r["index"] == idx and r["box"]]
    return sheet_boxes, [frames[i].time for i in pick]


def track(ctx: ToolContext, ref: str, points: list[list[float]] | None = None,
          labels: list[int] | None = None, boxes: list[list[float]] | None = None,
          from_artifact: str | None = None, marks: list[int] | None = None,
          t1: float | None = None, stride: int = 1, max_frames: int = 150) -> ToolResult:
    """Follow objects from one video frame forward with SAM 2's memory.

    Prompts are source pixels on the frame `ref` names (`vid_x#k`), one object per
    point or box, as in `segment`. Reports, per object, when it is present and
    where it drops out — the shape of "when did the cup leave the gripper".

    ponytail: forward only. Add a reverse pass when a prompt frame placed mid-event
    misses the onset.
    """
    asset_id, start = parse_ref(resolve_ref(ctx, ref)[0])
    if start is None:
        raise ValueError(f"track starts from one frame: pass {asset_id}#<frame_index>, "
                         "e.g. a ref from sample_frames")
    if stride < 1 or max_frames < 2:
        raise ValueError("stride must be >= 1 and max_frames >= 2")
    box_prompts = _prompt_boxes(ctx, from_artifact, marks, boxes)
    if bool(points) == bool(box_prompts):
        raise ValueError("track needs points=[[x, y], ...] or boxes / from_artifact marks "
                         "(not both), in source pixels on the starting frame")

    frames, s, size = _track_frames(ctx, asset_id, start, t1, stride, max_frames)
    labels, names, prompt = _track_prompt(points, labels, box_prompts, s)
    obj_ids = list(range(1, len(names) + 1))
    per_obj = _propagate(frames, s, size, prompt, obj_ids)

    objects, lines = {}, []
    for o, name in zip(obj_ids, names):
        objects[o], line = _track_summary(o, name, per_obj[o], stride)
        lines.append(line)

    sheet_boxes, times = _sheet_boxes(frames, objects)
    sheet = build_sheet(ctx, asset_id, times=times, boxes=sheet_boxes)
    first = [Mark(id=o, label=objects[o]["label"], box=objects[o]["frames"][0]["box"])
             for o in obj_ids]
    art = ctx.artifacts.save(
        sheet.image, kind="track", parent=asset_id, marks=first,
        params={"start": start, "t1": t1, "stride": stride, "max_frames": max_frames,
                "points": points, "labels": labels, "boxes": boxes,
                "from_artifact": from_artifact, "marks": marks, "times": times,
                "cells": sheet.cells, "sheet_scale": sheet.scale,
                "sheet_boxes": {str(k): [asdict(m) for m in v]
                                for k, v in sheet_boxes.items()},
                "objects": {str(k): v for k, v in objects.items()}})
    text = (f"artifact={art.id} tracked {len(names)} object(s) in {asset_id} over "
            f"{len(frames)} frames, {frames[0].time:.2f}..{frames[-1].time:.2f}s "
            f"(every {stride} frame(s)):\n" + "\n".join(lines)
            + "\nThe sheet shows 12 of those frames at thumbnail scale. A mask that "
            "drifts onto something else is a lost track, not an event: view the frames "
            "either side of a change before claiming it.")
    return ToolResult(text=text, images=[sheet.image], artifacts=[art.id])
