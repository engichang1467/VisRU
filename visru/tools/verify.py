"""Verification tools: draw the claim back onto the pixels, then finish.

`overlay` exists so a claim can be re-inspected rather than re-asserted: the geometry
the model believes in is rendered on the source and handed back for a second look.
"""

from PIL import Image

from visru.evidence.ledger import Claim, Ledger, RejectedReport
from visru.geometry import ViewTransform
from visru.tools import ToolContext, ToolResult, scale_note
from visru.tools.annotate import draw_marks
from visru.tools.navigate import render_region
from visru.workspace.artifacts import Mark


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
        parsed.append(Claim(claim_id=str(raw.get("claim_id", "")),
                            text=str(raw.get("text", "")),
                            status=str(raw.get("status", "")),
                            evidence=list(raw.get("evidence", [])),
                            checks=list(raw.get("checks", []))))

    ledger = Ledger(ctx.root, ctx.session, ctx.artifacts)
    try:
        report = ledger.submit(parsed)
    except RejectedReport as exc:
        # Every problem at once: one per turn would burn the session on bookkeeping.
        raise ValueError(
            f"report rejected, {len(parsed)} claims not recorded:\n{exc}\n"
            "Fix these and call submit_report again.") from exc
    return ToolResult(text=report.text(), artifacts=list(
        dict.fromkeys(a for c in report.claims for a in c.evidence)))
