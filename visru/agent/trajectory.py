"""Append-only record of a session: every model turn, tool call, and artifact ID.

Two jobs. The first is evidence: the Phase 1 exit criterion asks that every image
the model saw can be regenerated, and this log plus the artifact sidecars is what
that regeneration reads. The second is data — README.md:73 calls the trajectory
the dataset for the research phase, so records are written to be read by
something other than a person.

No images live here. An artifact ID plus its sidecar is enough to remake the
pixels, and a log that inlined PNGs would be unreadable and enormous.

Written next to the ledger (`.visru/ledger/<session>.json`), one file per session,
appended a line at a time so a crashed run still leaves everything up to the
crash.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from visru.tools import ToolContext, resolve_ref
from visru.tools.annotate import draw_grid, draw_marks
from visru.tools.navigate import build_sheet, render_region, sheet_boxes

TRAJECTORY_DIR = ".visru/trajectory"


@dataclass(frozen=True)
class Record:
    """One line of the log.

    `kind` is "model" for a turn the model produced, "tool" for a call the harness
    ran on its behalf. `shown` marks a tool result that carried images into the
    context — the exit criterion is about images the model actually saw, not every
    artifact a session happened to create, and `submit_report` cites plenty of the
    latter.
    """

    step: int
    kind: str
    ts: str = ""
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    artifacts: list[str] = field(default_factory=list)
    shown: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None


class Trajectory:
    """Writer. One per session; append as the loop runs."""

    def __init__(self, root: str | Path, session: str):
        self.root = Path(root).resolve()
        self.session = session
        self.path = self.root / TRAJECTORY_DIR / f"{session}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Record) -> Record:
        stamped = record if record.ts else Record(
            **{**asdict(record),
               "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        with self.path.open("a") as sink:
            sink.write(json.dumps(asdict(stamped)) + "\n")
        return stamped

    def model_turn(self, step: int, text: str, usage: dict[str, int] | None = None,
                   ) -> Record:
        return self.log(Record(step=step, kind="model", text=text,
                               usage=usage or {}))

    def tool_call(self, step: int, name: str, args: dict[str, Any], text: str,
                  artifacts: list[str], shown: bool,
                  error: str | None = None) -> Record:
        return self.log(Record(step=step, kind="tool", tool=name, args=args,
                               text=text, artifacts=artifacts, shown=shown,
                               error=error))


def read(path: str | Path) -> list[Record]:
    return list(stream(path))


def stream(path: str | Path) -> Iterator[Record]:
    """Records in order. Tolerates a truncated final line from a killed run."""
    with Path(path).open() as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                yield Record(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                return  # a partial write at the tail; everything before it stands


def shown_artifacts(records: list[Record]) -> list[str]:
    """Every artifact whose pixels reached the model, in order, deduped."""
    seen: dict[str, None] = {}
    for rec in records:
        if rec.kind == "tool" and rec.shown:
            for artifact_id in rec.artifacts:
                seen.setdefault(artifact_id, None)
    return list(seen)


def replay_artifact(ctx: ToolContext, artifact_id: str) -> Image.Image:
    """Remake an artifact's image from its sidecar.

    This is the claim `workspace/artifacts.py` has been making since Phase 0 —
    "given the log, any view can be remade" — actually exercised.

    It re-derives rather than re-running the tool, which matters twice over: no
    detector or OCR model has to be loaded to check a log, and no second artifact
    is created as a side effect of verifying the first. Marks and masks are in the
    sidecar, so redrawing them is deterministic even though producing them was not.
    """
    art = ctx.artifacts.load(artifact_id)
    if art.parent is None:
        raise ValueError(f"{artifact_id} has no parent; nothing to re-derive from")

    if art.kind == "contact_sheet":
        return build_sheet(ctx, art.parent, art.params["n"],
                           art.params.get("t0"), art.params.get("t1")).image
    if art.kind in ("search", "track"):
        return build_sheet(ctx, art.parent, times=art.params["times"],
                           boxes=sheet_boxes(art.params.get("sheet_boxes"))).image

    asset, _ = resolve_ref(ctx, art.parent)
    img, tf, _ = render_region(ctx, asset, art.params.get("box"))
    if art.params.get("grid"):
        img = draw_grid(img, tf)
    if art.marks:
        marks = [art.mark(m["id"]) for m in art.marks]
        masks = {m.id: ctx.artifacts.load_mask(art.id, m.id)
                 for m in marks if m.mask_path}
        img = draw_marks(img, marks, tf, masks=masks or None)
    return img


def verify_regenerable(ctx: ToolContext, records: list[Record]) -> list[str]:
    """Re-derive every image the model saw. Returns the IDs that did not match.

    An empty list is the Phase 1 exit criterion's second half, demonstrated rather
    than asserted.
    """
    failed = []
    for artifact_id in shown_artifacts(records):
        try:
            remade = replay_artifact(ctx, artifact_id)
            stored = ctx.artifacts.load_image(artifact_id)
        except (KeyError, ValueError, OSError):
            failed.append(artifact_id)
            continue
        if remade.convert("RGB").tobytes() != stored.convert("RGB").tobytes():
            failed.append(artifact_id)
    return failed
