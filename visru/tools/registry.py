"""Tool schemas, descriptions, and dispatch — one table, every driver.

The MCP server (Phase 0) and the agent loop (Phase 1) both need the same three
things per tool: the name the model calls, the text that tells it when to call,
and the arguments it may pass. Before this module those lived only as decorated
wrapper signatures inside the MCP server, so the loop had nowhere to read them.

Descriptions are prompt text, not documentation: they are the only place the
model is told that coordinates are source-asset pixels, that a SAM mask prompted
from a detector box checks nothing, and that the ledger will refuse a status the
evidence does not warrant. Keep them here, once.

`params` is declared rather than introspected because the public argument names
do not all match the Python ones: every pixel tool takes `ref` in Python and is
called `asset_id` by the model. That divergence used to be invisible, held
together only by the MCP server passing everything positionally.
"""

from dataclasses import dataclass
from typing import Any, Callable

from visru.agent import subagents
from visru.tools import ToolContext, ToolResult
from visru.tools import navigate, search, verify

# Bad input from the model, not a crash. A swallowed message ("Error executing
# tool view") leaves the agent guessing, and it will retry the same wrong call.
USER_ERRORS = (ValueError, KeyError, IndexError, FileNotFoundError, NotADirectoryError)

REQUIRED = object()  # sentinel: distinct from a legitimate None default

# Tools that only work inside visru's own loop. `delegate` runs child sessions on
# the loop's model; an MCP client such as Claude Code has its own subagents.
HARNESS_ONLY = {"delegate"}


@dataclass(frozen=True)
class Param:
    """One public argument: the name the model uses and the shape it must send."""

    name: str
    schema: dict[str, Any]
    default: Any = REQUIRED

    @property
    def required(self) -> bool:
        return self.default is REQUIRED


@dataclass(frozen=True)
class ToolSpec:
    name: str
    fn: Callable[..., ToolResult]
    description: str
    params: tuple[Param, ...]

    def json_schema(self) -> dict[str, Any]:
        """JSON Schema for the arguments object, as every provider wants it."""
        return {
            "type": "object",
            "properties": {p.name: p.schema for p in self.params},
            "required": [p.name for p in self.params if p.required],
        }

    def bind(self, arguments: dict[str, Any]) -> list[Any]:
        """Model arguments to a positional call, in declared order."""
        unknown = set(arguments) - {p.name for p in self.params}
        if unknown:
            known = ", ".join(p.name for p in self.params)
            raise ValueError(
                f"{self.name} got unknown argument(s) {', '.join(sorted(unknown))}; "
                f"it takes {known}")
        missing = [p.name for p in self.params if p.required and p.name not in arguments]
        if missing:
            raise ValueError(f"{self.name} needs {', '.join(missing)}")
        return [arguments.get(p.name, p.default) for p in self.params]


# --- schema fragments ----------------------------------------------------
_STR = {"type": "string"}
_INT = {"type": "integer"}
_NUM = {"type": "number"}
_BOOL = {"type": "boolean"}
_INTS = {"type": "array", "items": {"type": "integer"}}
_STRS = {"type": "array", "items": {"type": "string"}}
_POINTS = {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}


def _box(what: str) -> dict[str, Any]:
    return {**_INTS, "description": what}


VIEW_DESC = """Show a region of an asset at the highest resolution the model accepts.

asset_id is an image ID, or `vid_<id>#<frame_index>` for one video frame.
box is [x0, y0, x1, y1] in SOURCE-ASSET pixels; omit it for the whole asset.
With grid=True a ruler labelled in source pixels is drawn on the image.
Returns an artifact ID; cite it as evidence, and pass it to view again later."""

SEGMENT_DESC = """Mask objects from point, box, or mark prompts, all in SOURCE-ASSET pixels.

points are [[x, y], ...] with labels 1 (foreground) or 0 (background), one
object per point. from_artifact='a_0005' with marks=[1, 3] segments a previous
detection's marks. box is the region to render and work in.

A mask prompted from a detector box is NOT an independent check of that
detection: it inherits the detector's error."""

SUBMIT_DESC = """Submit the final claims with their evidence. This ends the task.

Each claim is an object:
  claim_id: "c_1"
  text:     what you are asserting
  status:   verified | supported | contested | hypothesis
  evidence: ["a_0005", "a_0007"] — artifact IDs that show it
  checks:   ["a_0009"] — artifact IDs returned by cross_check

The ledger decides what your evidence warrants and rejects any status stronger
than that, listing every problem at once:
  - a claim with no evidence must be `hypothesis`
  - `verified` needs a cross_check artifact that agreed; checks you describe
    yourself ({"methods": ..., "result": "agree"}) never earn it
  - if any check disagrees, the claim is `contested`"""

SEARCH_DESC = """Find where, and in a video when, something appears, by text query.

Ranks the workspace's embedding index (every image, and video frames sampled
about once a second) against a short description ("a person opening a fridge").
asset_id limits it to one asset; t0/t1 to a time range in seconds. For video,
hits on neighbouring samples merge into time windows, listed as marks.

Scores are relative and coarse: a window says where to look, not what happened.
Confirm it with sample_frames over the window and view on a frame ref."""

TRACK_DESC = """Follow objects through a video from one frame, with SAM 2's memory.

asset_id is the starting frame, `vid_<id>#<frame_index>`. Prompt objects there as
in segment — points [[x, y], ...] (one object per point), boxes, or
from_artifact + marks from a detect on that frame — all in SOURCE-ASSET pixels.
Tracks forward to t1 seconds (default: the end), every `stride` frames.

Reports per object when it is present and when it drops out, plus a sheet of
tracked frames with boxes. Use it for "when did X appear, leave, stop, fall".
A track sharing SAM 2 with segment is not an independent check of it."""

DELEGATE_DESC = """Split a video into time windows and give each to a subagent.

Each subagent is a fresh session with its own context and every tool but this
one; it investigates `question` inside its window and reports claims with
evidence. You get back their claims and artifact IDs as text, not their images.

windows is [[t0, t1], ...] in seconds (at most 8); omit it for n equal windows.
Use it on long videos, where one context cannot hold enough frames at useful
scale. It costs a session per window: prefer search and sample_frames first."""

CROSS_CHECK_DESC = """Compare two independent methods' readings and record the verdict.

kind: count | text | localize | temporal. artifacts: IDs from at least two
methods on the same asset — view, detect, segment, or ocr. A `view` artifact stands for YOUR
reading of it, so pass `value` (the count or the text) or, for localize, `box`
in source-asset pixels. detect/segment/ocr artifacts speak through their marks;
`marks` narrows them, one entry per artifact (an ID, a list of IDs, or null).

kind temporal compares time windows in one video: a search window or a track
(`marks` picks which), against your reading of a view or sample_frames sheet
passed as value="t0..t1" seconds.

Agreement: counts equal, text equal after normalising case, punctuation and
spacing, boxes at IoU >= 0.5, windows with both ends within 1 s. Methods sharing
a model or a lineage (a mask prompted from a detection, a view of a detect
artifact, a view and a contact sheet) are refused.
Returns a check artifact ID; cite it in the claim's checks to earn verified."""


SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="list_assets",
        fn=navigate.list_assets,
        description=("List workspace assets. Filter by filename prefix, kind "
                     "(image|video), or a substring of the path. Start here: it is "
                     "cheap and tells you what exists."),
        params=(
            Param("prefix", _STR, None),
            Param("kind", {**_STR, "enum": ["image", "video"]}, None),
            Param("contains", _STR, None),
            Param("limit", _INT, 100),
        ),
    ),
    ToolSpec(
        name="view",
        fn=navigate.view,
        description=VIEW_DESC,
        params=(
            Param("asset_id", _STR),
            Param("box", _box("[x0, y0, x1, y1] in source-asset pixels"), None),
            Param("grid", _BOOL, False),
        ),
    ),
    ToolSpec(
        name="sample_frames",
        fn=navigate.sample_frames,
        description=("Contact sheet of n frames from a video, optionally within "
                     "t0..t1 seconds.\n\nEach cell is labelled with a frame ref like "
                     "`vid_abc123#1164`. Thumbnails are below native scale: call view "
                     "on a cell's ref before claiming small detail."),
        params=(
            Param("asset_id", _STR),
            Param("n", _INT, 12),
            Param("t0", _NUM, None),
            Param("t1", _NUM, None),
        ),
    ),
    ToolSpec(
        name="search",
        fn=search.search,
        description=SEARCH_DESC,
        params=(
            Param("query", _STR),
            Param("asset_id", _STR, None),
            Param("t0", _NUM, None),
            Param("t1", _NUM, None),
            Param("k", _INT, 8),
        ),
    ),
    ToolSpec(
        name="detect",
        fn=search.detect,
        description=("Find objects by open-vocabulary text query, drawn as numbered "
                     "marks.\n\nqueries are short noun phrases (\"a torn label\", \"the "
                     "gripper\"). box limits the search to a region in SOURCE-ASSET "
                     "pixels; a tighter box renders at higher scale and finds smaller "
                     "objects. Results are marks with source-pixel boxes — refer to "
                     "them as \"mark N\" and pass them to segment via from_artifact."),
        params=(
            Param("asset_id", _STR),
            Param("queries", _STRS),
            Param("box", _box("region to search, in source-asset pixels"), None),
            Param("threshold", _NUM, 0.1),
            Param("max_marks", _INT, 20),
        ),
    ),
    ToolSpec(
        name="segment",
        fn=search.segment,
        description=SEGMENT_DESC,
        params=(
            Param("asset_id", _STR),
            Param("points", _POINTS, None),
            Param("labels", _INTS, None),
            Param("boxes", _POINTS, None),
            Param("from_artifact", _STR, None),
            Param("marks", _INTS, None),
            Param("box", _box("region to render and work in"), None),
        ),
    ),
    ToolSpec(
        name="track",
        fn=search.track,
        description=TRACK_DESC,
        params=(
            Param("asset_id", _STR),
            Param("points", _POINTS, None),
            Param("labels", _INTS, None),
            Param("boxes", _POINTS, None),
            Param("from_artifact", _STR, None),
            Param("marks", _INTS, None),
            Param("t1", _NUM, None),
            Param("stride", _INT, 1),
            Param("max_frames", _INT, 150),
        ),
    ),
    ToolSpec(
        name="ocr",
        fn=search.ocr,
        description=("Read text, returning each line as a numbered mark with a "
                     "source-pixel box.\n\nSmall text needs a tight box: the region is "
                     "rendered once at model resolution, so a whole 4K page reads far "
                     "worse than a zoomed paragraph."),
        params=(
            Param("asset_id", _STR),
            Param("box", _box("region to read, in source-asset pixels"), None),
            Param("min_confidence", _NUM, 0.5),
        ),
    ),
    ToolSpec(
        name="overlay",
        fn=verify.overlay,
        description=("Draw claimed geometry back onto the source and return it for "
                     "re-inspection.\n\nboxes are [[x0, y0, x1, y1], ...] in "
                     "SOURCE-ASSET pixels. from_artifact='a_0006' replays an earlier "
                     "detection or segmentation, masks included, onto this render — "
                     "use it to check at native scale something you found while zoomed "
                     "out.\n\nLook at what comes back. A shape that does not sit on the "
                     "thing it names means the claim behind it is wrong."),
        params=(
            Param("asset_id", _STR),
            Param("boxes", _POINTS, None),
            Param("labels", _STRS, None),
            Param("from_artifact", _STR, None),
            Param("marks", _INTS, None),
            Param("box", _box("region to render and work in"), None),
        ),
    ),
    ToolSpec(
        name="cross_check",
        fn=verify.cross_check,
        description=CROSS_CHECK_DESC,
        params=(
            Param("kind", {**_STR, "enum": ["count", "text", "localize", "temporal"]}),
            Param("artifacts", _STRS),
            Param("value", {"type": ["string", "integer"]}, None),
            Param("box", {"type": "array", "items": {"type": "number"}}, None),
            Param("marks", {"type": "array"}, None),
        ),
    ),
    ToolSpec(
        name="delegate",
        fn=subagents.delegate,
        description=DELEGATE_DESC,
        params=(
            Param("asset_id", _STR),
            Param("question", _STR),
            Param("windows", _POINTS, None),
            Param("n", _INT, 4),
            Param("max_steps", _INT, 15),
        ),
    ),
    ToolSpec(
        name="coverage",
        fn=verify.coverage,
        description=("What this session has looked at: per asset, the share viewed "
                     "at or above inspected_scale and at any scale, frames for video, "
                     "and assets never viewed. Call it before submit_report and say "
                     "what you did not inspect."),
        params=(),
    ),
    ToolSpec(
        name="submit_report",
        fn=verify.submit_report,
        description=SUBMIT_DESC,
        params=(
            Param("claims", {"type": "array", "items": {"type": "object"}}),
        ),
    ),
)

TOOLS: dict[str, ToolSpec] = {spec.name: spec for spec in SPECS}


def schemas(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Tool definitions in the shape every provider's API wants.

    `names` restricts the set, which is how the eval arms (zoom-only, full) differ.
    """
    return [{"name": s.name, "description": s.description,
             "input_schema": s.json_schema()} for s in SPECS
            if names is None or s.name in names]


def execute(ctx: ToolContext, name: str, arguments: dict[str, Any],
            allowed: list[str] | None = None) -> ToolResult:
    """Run one tool call. Raises USER_ERRORS for anything the model can fix."""
    spec = TOOLS.get(name)
    if spec is None or (allowed is not None and name not in allowed):
        have = sorted(TOOLS) if allowed is None else sorted(allowed)
        raise KeyError(f"no tool named {name!r}; have {', '.join(have)}")
    return spec.fn(ctx, *spec.bind(arguments))
