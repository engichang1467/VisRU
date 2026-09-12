"""Navigation tools: what is here, show me this region, show me these frames."""

import math

from PIL import Image

from visru.geometry import ViewTransform, fit_render_size
from visru.tools import ToolContext, ToolResult, scale_note
from visru.tools.annotate import caption, draw_grid
from visru.workspace import video

CAPTION_H = 22


def list_assets(ctx: ToolContext, prefix: str | None = None, kind: str | None = None,
                contains: str | None = None, limit: int = 100) -> ToolResult:
    """Filter the asset registry by filename prefix, kind, or path substring."""
    found = ctx.registry.list(prefix=prefix, kind=kind, contains=contains)
    shown = found[:limit]
    lines = [a.summary() for a in shown]
    if len(found) > len(shown):
        lines.append(f"... {len(found) - len(shown)} more; narrow with prefix/contains")
    if not found:
        lines.append(f"no assets match (registry holds {len(ctx.registry.assets)})")
    return ToolResult(text="\n".join(lines))


def _clamp_box(box: list[int] | None, w: int, h: int) -> tuple[int, int, int, int]:
    if box is None:
        return 0, 0, w, h
    if len(box) != 4:
        raise ValueError(f"box must be [x0, y0, x1, y1] in source pixels, got {box}")
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"box {box} is empty; expected x0 < x1 and y0 < y1")
    cx0, cy0 = max(0, min(x0, w - 1)), max(0, min(y0, h - 1))
    cx1, cy1 = max(cx0 + 1, min(x1, w)), max(cy0 + 1, min(y1, h))
    return cx0, cy0, cx1, cy1


def render_region(ctx: ToolContext, ref: str, box: list[int] | None = None,
                  ) -> tuple[Image.Image, ViewTransform, tuple[int, int, int, int]]:
    """Crop and render one region at model resolution. Every tool that shows or
    analyses pixels goes through here, so they all share one coordinate story."""
    src = ctx.registry.load_image(ref)
    x0, y0, x1, y1 = _clamp_box(box, src.width, src.height)
    crop = src.crop((x0, y0, x1, y1))
    rw, rh, scale = fit_render_size(crop.width, crop.height, **ctx.cfg.model.limits)
    out = crop.resize((rw, rh), Image.Resampling.LANCZOS)
    return out, ViewTransform(ref, float(x0), float(y0), scale), (x0, y0, x1, y1)


def view(ctx: ToolContext, ref: str, box: list[int] | None = None,
         grid: bool = False) -> ToolResult:
    """Render a region of an asset at the highest resolution the model accepts.

    `ref` is an asset ID, or `vid_<id>#<frame_index>` for one video frame.
    `box` is [x0, y0, x1, y1] in source-asset pixels; omit it for the whole asset.
    """
    out, tf, (x0, y0, x1, y1) = render_region(ctx, ref, box)
    rw, rh, scale = out.width, out.height, tf.scale
    if grid:
        out = draw_grid(out, tf)

    art = ctx.artifacts.save(out, kind="view", parent=ref,
                             params={"box": [x0, y0, x1, y1], "grid": grid},
                             transform=tf)
    note = scale_note(ctx.cfg, scale)
    text = (f"artifact={art.id} source={ref} box=[{x0}, {y0}, {x1}, {y1}] "
            f"scale={scale:.3f} view_px={rw}x{rh} — {note}")
    return ToolResult(text=text, images=[out], artifacts=[art.id])


def sample_frames(ctx: ToolContext, ref: str, n: int = 12, t0: float | None = None,
                  t1: float | None = None) -> ToolResult:
    """Contact sheet of n frames over a time range, each cell labelled with its ref."""
    meta = ctx.registry.video_meta(ref)
    path = ctx.registry.abspath(ref)
    times = video.frame_times(meta, n, t0, t1)
    frames = video.decode_at(path, times, meta)
    if not frames:
        raise ValueError(f"no frames decoded from {ref} over {t0}..{t1}")

    cols = math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / cols)
    cell_w = min(meta.width, ctx.cfg.model.max_edge // cols)
    cell_h = max(1, round(cell_w * meta.height / meta.width))

    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + CAPTION_H)), (0, 0, 0))
    cells = []
    for i, frame in enumerate(frames):
        frame_ref = f"{ctx.registry.get(ref).id}#{frame.index}"
        thumb = frame.image.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        cx, cy = (i % cols) * cell_w, (i // cols) * (cell_h + CAPTION_H)
        sheet.paste(caption(thumb, f"{frame_ref}  t={frame.time:.2f}s", CAPTION_H), (cx, cy))
        cells.append({"ref": frame_ref, "index": frame.index, "time": round(frame.time, 3),
                      "cell": [cx, cy, cx + cell_w, cy + cell_h]})

    rw, rh, scale = fit_render_size(sheet.width, sheet.height, **ctx.cfg.model.limits)
    if (rw, rh) != sheet.size:
        sheet = sheet.resize((rw, rh), Image.Resampling.LANCZOS)

    art = ctx.artifacts.save(
        sheet, kind="contact_sheet", parent=ref,
        # No transform: a sheet has no single mapping back to the source. The cell
        # refs are the way back, and `view` on one of them is exact.
        params={"n": n, "t0": t0, "t1": t1, "cells": cells, "sheet_scale": scale})
    listing = "\n".join(f"  {c['ref']} t={c['time']}s" for c in cells)
    text = (f"artifact={art.id} source={ref} {len(cells)} frames "
            f"over {times[0]:.2f}..{times[-1]:.2f}s, thumbnails at "
            f"{cell_w / meta.width:.2f}x source scale.\n"
            f"Frames (call view on one of these refs to inspect at native scale):\n"
            f"{listing}")
    return ToolResult(text=text, images=[sheet], artifacts=[art.id])
