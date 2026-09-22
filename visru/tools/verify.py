"""Verification tools: draw the claim back onto the pixels, check it, then finish.

`overlay` exists so a claim can be re-inspected rather than re-asserted: the geometry
the model believes in is rendered on the source and handed back for a second look.
`cross_check` compares two methods' outputs and records the verdict where the ledger
can read it; `coverage` says what the session actually looked at closely.
"""

import itertools
import math
import re

import numpy as np
from PIL import Image

from visru.evidence.ledger import Check, Claim, Ledger, RejectedReport
from visru.geometry import ViewTransform
from visru.tools import ToolContext, ToolResult, resolve_ref, scale_note
from visru.tools.annotate import draw_marks
from visru.tools.navigate import render_region
from visru.workspace.artifacts import Artifact, Mark
from visru.workspace.registry import parse_ref

CHECKABLE = ("view", "detect", "segment", "ocr")  # artifacts that carry a reading
CHECK_KINDS = ("count", "text", "localize")
IOU_AGREE = 0.5
COVERAGE_CELL = 16  # source px per coverage-grid cell


def normalise(text: str) -> str:
    """Lowercase, drop punctuation and spacing, so "TRADER JOE'S" == "trader joes".

    Apostrophes are deleted rather than spaced, or the possessive splits into a
    stray token and a correct answer grades as wrong.
    """
    without_quotes = re.sub(r"['’ʼ]", "", str(text).lower())
    return re.sub(r"[^a-z0-9]+", " ", without_quotes).strip()


def _reproject_mask(mask: Image.Image, source_tf: ViewTransform,
                    target_tf: ViewTransform, target_size: tuple[int, int]) -> Image.Image:
    """Move a mask from the view frame it was made in to the one being drawn.

    Masks are stored in their own artifact's frame, so overlaying one on a different
    crop or zoom means placing it through both transforms rather than pasting it.
    """
    ratio = target_tf.scale / source_tf.scale
    scaled = mask.resize((max(1, round(mask.width * ratio)),
                          max(1, round(mask.height * ratio))), Image.Resampling.NEAREST)
    canvas = Image.new("L", target_size, 0)
    u, v = target_tf.to_view(source_tf.x0, source_tf.y0)
    canvas.paste(scaled, (round(u), round(v)))
    return canvas


def overlay(ctx: ToolContext, ref: str, boxes: list[list[float]] | None = None,
            labels: list[str] | None = None, from_artifact: str | None = None,
            marks: list[int] | None = None, box: list[int] | None = None) -> ToolResult:
    """Render claimed geometry on the source and return it for re-inspection.

    Boxes are source-asset pixels. `from_artifact` replays an earlier detection or
    segmentation — including its masks — onto a fresh render, which is how a mask
    found at one zoom gets checked at another.
    """
    drawn: list[Mark] = []
    mask_sources: list[tuple[int, str, int]] = []

    for i, b in enumerate(boxes or []):
        if len(b) != 4:
            raise ValueError(f"each box must be [x0, y0, x1, y1] in source pixels, got {b}")
        label = (labels or [])[i] if i < len(labels or []) else "claimed"
        drawn.append(Mark(id=len(drawn) + 1, label=label, box=tuple(float(v) for v in b)))

    if from_artifact:
        art = ctx.artifacts.load(from_artifact)
        wanted = marks or [m["id"] for m in art.marks]
        for mark_id in wanted:
            mark = art.mark(mark_id)
            drawn.append(Mark(id=len(drawn) + 1,
                              label=f"{from_artifact} mark {mark_id}: {mark.label}",
                              box=mark.box, score=mark.score))
            if mark.mask_path:
                mask_sources.append((len(drawn), from_artifact, mark_id))

    if not drawn:
        raise ValueError(
            "overlay needs boxes=[[x0, y0, x1, y1], ...] in source pixels, or "
            "from_artifact='a_0006' to replay an earlier result")

    img, tf, region = render_region(ctx, ref, box)
    replayed: dict[int, Image.Image] = {}
    for new_id, artifact_id, mark_id in mask_sources:
        source_tf = ctx.artifacts.load(artifact_id).view_transform
        mask = ctx.artifacts.load_mask(artifact_id, mark_id)
        replayed[new_id] = _reproject_mask(mask, source_tf, tf, img.size)

    out = draw_marks(img, drawn, tf, masks=replayed or None)
    art = ctx.artifacts.save(out, kind="overlay", parent=ref, transform=tf, marks=drawn,
                             params={"box": list(region), "boxes": boxes,
                                     "labels": labels, "from_artifact": from_artifact,
                                     "marks": marks})
    # A segment mark can carry a mask and no box (an empty mask records no extent),
    # so the listing must not assume one.
    listing = "\n".join(
        f"  mark {m.id}: {m.label}" + (f" box={list(m.box)}" if m.box else " (mask only)")
        for m in drawn)
    text = (f"artifact={art.id} source={ref} {len(drawn)} claimed shapes drawn in "
            f"box={list(region)} at {tf.scale:.2f}x source scale — "
            f"{scale_note(ctx.cfg, tf.scale)}.\n{listing}\n"
            "Look at the overlay: if a shape does not sit on the thing it names, the "
            "claim behind it is wrong.")
    return ToolResult(text=text, images=[out], artifacts=[art.id])


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _selected(art: Artifact, pick) -> list[Mark]:
    wanted = ([m["id"] for m in art.marks] if pick is None
              else [pick] if isinstance(pick, int) else list(pick))
    return [art.mark(i) for i in wanted]


def _reading(art: Artifact, kind: str, pick, value, box):
    """What one method says: the model's own reading for a view, the marks otherwise."""
    if art.kind == "view":
        # A view has no output of its own; it stands for the model reading the pixels.
        given = box if kind == "localize" else value
        if given is None:
            raise ValueError(f"{art.id} is a view, so it stands for your own reading: pass "
                             + ("box=[x0, y0, x1, y1]" if kind == "localize" else "value="))
        return (list(given), []) if kind == "localize" else (given, [])
    marks = _selected(art, pick)
    if kind == "count":
        return len(marks), marks
    if kind == "text":
        return " ".join(m.label for m in marks), marks
    boxed = [m for m in marks if m.box]
    if len(boxed) != 1:
        raise ValueError(f"localize needs exactly one mark from {art.id}; pass marks with "
                         f"one ID per artifact (it has {[m['id'] for m in art.marks]})")
    return list(boxed[0].box), boxed


def _agree(kind: str, readings: list) -> tuple[bool, dict]:
    if kind == "count":
        try:
            values = {int(r) for r in readings}
        except (TypeError, ValueError):
            raise ValueError(f"count readings must be integers, got {readings}") from None
        return len(values) == 1, {}
    if kind == "text":
        # Spaces dropped too: OCR splits a sign into lines a person reads as one.
        return len({normalise(r).replace(" ", "") for r in readings}) == 1, {}
    iou = min(_iou(a, b) for a, b in itertools.combinations(readings, 2))
    return iou >= IOU_AGREE, {"iou": round(iou, 3)}


def cross_check(ctx: ToolContext, kind: str, artifacts: list[str],
                value: str | int | None = None, box: list[float] | None = None,
                marks: list | None = None) -> ToolResult:
    """Compare what two independent methods say, and record the verdict.

    Runs no model: it reads outputs that already exist. A `view` artifact stands for
    the model's own reading (`value`, or `box` for localize); detect, segment and
    ocr artifacts speak through their marks, optionally narrowed by `marks`, one
    entry per artifact. Pairs that share a model or a lineage are refused before
    anything is compared, since their agreement would mean nothing.
    """
    if kind not in CHECK_KINDS:
        raise ValueError(f"kind must be one of {list(CHECK_KINDS)}, got {kind!r}")
    artifacts = list(dict.fromkeys(artifacts or []))
    if len(artifacts) < 2:
        raise ValueError("cross_check needs artifacts from at least two methods")
    marks = list(marks or [])
    if marks and len(marks) != len(artifacts):
        raise ValueError(f"marks has {len(marks)} entries for {len(artifacts)} artifacts; "
                         "give one per artifact (null for all of its marks)")

    arts = [ctx.artifacts.load(a) for a in artifacts]
    for art in arts:
        if art.kind not in CHECKABLE:
            raise ValueError(f"{art.id} is a {art.kind}; cross_check reads "
                             f"{', '.join(CHECKABLE)} artifacts")
    assets = {resolve_ref(ctx, a.id)[0] for a in arts}
    if len(assets) != 1:
        raise ValueError(f"the artifacts come from different assets {sorted(assets)}; "
                         "a check compares readings of the same pixels")
    asset = assets.pop()

    methods = [f"{a.kind} {a.id}" for a in arts]
    ledger = Ledger(ctx.root, ctx.session, ctx.artifacts)
    refused = ledger.check_problems(Check(kind=kind, methods=methods, result="agree",
                                          artifacts=artifacts))
    if refused:
        raise ValueError("cross_check refused, nothing recorded:\n"
                         + "\n".join(f"- {p}" for p in refused))

    readings, drawn = [], []
    for art, pick in itertools.zip_longest(arts, marks):
        reading, used = _reading(art, kind, pick, value, box)
        readings.append(reading)
        for m in used:
            if m.box:
                drawn.append(Mark(id=len(drawn) + 1, label=f"{art.id}: {m.label}", box=m.box))
        if art.kind == "view" and kind == "localize":
            drawn.append(Mark(id=len(drawn) + 1, label=f"{art.id}: claimed",
                              box=tuple(float(v) for v in reading)))
    if value is not None and not any(a.kind == "view" for a in arts):
        readings.append(value)  # the claimed answer must match the methods too
    agreed, detail = _agree(kind, readings)
    result = "agree" if agreed else "disagree"

    region = None
    if drawn:  # frame the evidence, not the whole 8K image
        x0 = min(m.box[0] for m in drawn); y0 = min(m.box[1] for m in drawn)
        x1 = max(m.box[2] for m in drawn); y1 = max(m.box[3] for m in drawn)
        pad = max(16, 0.15 * max(x1 - x0, y1 - y0))
        region = [math.floor(x0 - pad), math.floor(y0 - pad),
                  math.ceil(x1 + pad), math.ceil(y1 + pad)]
    img, tf, region = render_region(ctx, asset, region)
    out = draw_marks(img, drawn, tf) if drawn else img
    art = ctx.artifacts.save(out, kind="check", parent=asset, transform=tf, marks=drawn,
                             params={"box": list(region), "kind": kind, "methods": methods,
                                     "artifacts": artifacts, "value": value,
                                     "readings": readings, "result": result,
                                     "detail": detail})
    shown = "\n".join(f"  {m}: {r!r}" for m, r in zip(methods, readings))
    extra = f" iou={detail['iou']}" if "iou" in detail else ""
    text = (f"artifact={art.id} cross_check {kind}: {result.upper()}{extra}\n{shown}\n"
            + (f"Cite it in the claim's checks as {{\"check\": \"{art.id}\"}}; that "
               "is what earns verified." if agreed else
               f"The methods disagree. Look again, or report the claim as contested "
               f"citing {{\"check\": \"{art.id}\"}}."))
    return ToolResult(text=text, artifacts=[art.id])


def coverage_summary(ctx: ToolContext) -> dict:
    """Which parts of which assets this session saw, and at what scale.

    Read from the session's own artifacts, so it works the same under MCP, where
    there is no trajectory. Images are tracked on a coarse grid of source pixels;
    video frames are counted whole.
    """
    inspected_scale = ctx.cfg.coverage.inspected_scale
    grids: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    frames: dict[str, tuple[set, set]] = {}

    def grid(asset_id):
        if asset_id not in grids:
            a = ctx.registry.get(asset_id)
            shape = (math.ceil(a.height / COVERAGE_CELL), math.ceil(a.width / COVERAGE_CELL))
            grids[asset_id] = (np.zeros(shape, bool), np.zeros(shape, bool))
        return grids[asset_id]

    for art in ctx.artifacts.all():
        if art.session != ctx.session or art.kind == "check":
            continue  # a check is the harness's render, never shown to the model
        if art.kind == "contact_sheet":
            asset_id, _ = parse_ref(art.parent)
            seen, _ = frames.setdefault(asset_id, (set(), set()))
            seen.update(c["index"] for c in art.params.get("cells", []))
            continue
        if not art.transform or "box" not in art.params:
            continue
        asset_id, frame = parse_ref(art.transform["asset_id"])
        fine = art.transform["scale"] >= inspected_scale
        if frame is not None:
            seen, close = frames.setdefault(asset_id, (set(), set()))
            seen.add(frame)
            if fine:
                close.add(frame)
            continue
        seen, close = grid(asset_id)
        x0, y0, x1, y1 = (v // COVERAGE_CELL for v in art.params["box"])
        seen[y0:y1 + 1, x0:x1 + 1] = True
        if fine:
            close[y0:y1 + 1, x0:x1 + 1] = True

    assets = {}
    for asset_id, (seen, close) in grids.items():
        assets[asset_id] = {"kind": "image", "seen": round(float(seen.mean()), 3),
                            "inspected": round(float(close.mean()), 3)}
    for asset_id, (seen, close) in frames.items():
        assets[asset_id] = {"kind": "video", "frames": ctx.registry.get(asset_id).frames,
                            "frames_seen": len(seen), "frames_inspected": len(close)}
    untouched = [a.id for a in ctx.registry.list() if a.id not in assets]
    return {"inspected_scale": inspected_scale, "assets": assets, "untouched": untouched}


def coverage(ctx: ToolContext) -> ToolResult:
    """What this session has looked at, and how closely."""
    cov = coverage_summary(ctx)
    lines = [f"coverage at inspected_scale={cov['inspected_scale']} "
             "(rendered px per source px):"]
    for asset_id, c in cov["assets"].items():
        if c["kind"] == "image":
            lines.append(f"  {asset_id}: {c['inspected']:.0%} of the image at or above "
                         f"inspected_scale, {c['seen']:.0%} seen at any scale")
        else:
            lines.append(f"  {asset_id}: {c['frames_inspected']} of {c['frames']} frames "
                         f"viewed at or above inspected_scale, {c['frames_seen']} seen "
                         "in any form (contact-sheet cells are below scale)")
    if not cov["assets"]:
        lines.append("  nothing viewed yet")
    rest = cov["untouched"]
    if rest:
        lines.append(f"never viewed: {len(rest)} assets: {', '.join(rest[:20])}"
                     + (" …" if len(rest) > 20 else ""))
    lines.append("Report what you did not inspect; a claim about an area below "
                 "inspected_scale rests on a thumbnail.")
    return ToolResult(text="\n".join(lines))


def submit_report(ctx: ToolContext, claims: list[dict]) -> ToolResult:
    """Submit claims with evidence. Malformed claims are rejected, not downgraded.

    Each claim is {claim_id, text, status, evidence: [artifact ids], checks: [...]}.
    """
    if not isinstance(claims, list):
        raise ValueError(f"claims must be a list of claim objects, got {type(claims).__name__}")

    parsed: list[Claim] = []
    for raw in claims:
        if not isinstance(raw, dict):
            raise ValueError(f"each claim must be an object, got {raw!r}")
        unknown = set(raw) - {"claim_id", "text", "status", "evidence", "checks", "session"}
        if unknown:
            raise ValueError(f"unknown claim fields {sorted(unknown)}; expected claim_id, "
                             "text, status, evidence, checks")
        checks = []
        for c in raw.get("checks", []):
            if isinstance(c, str):  # shorthand: a bare cross_check artifact ID
                c = {"check": c}
            if not isinstance(c, dict):
                raise ValueError(f"each check must be an object or a check artifact ID, "
                                 f"got {c!r}")
            checks.append(c)
        parsed.append(Claim(claim_id=str(raw.get("claim_id", "")),
                            text=str(raw.get("text", "")),
                            status=str(raw.get("status", "")),
                            evidence=list(raw.get("evidence", [])),
                            checks=checks))

    ledger = Ledger(ctx.root, ctx.session, ctx.artifacts)
    try:
        report = ledger.submit(parsed, coverage=coverage_summary(ctx))
    except RejectedReport as exc:
        # Every problem at once: one per turn would burn the session on bookkeeping.
        raise ValueError(
            f"report rejected, {len(parsed)} claims not recorded:\n{exc}\n"
            "Fix these and call submit_report again.") from exc
    return ToolResult(text=report.text(), artifacts=list(
        dict.fromkeys(a for c in report.claims for a in c.evidence)))
