"""MCP server exposing visru's tools, so any MCP client can drive them.

mcp 2.x renamed FastMCP to MCPServer; the tool decorator is otherwise the same.
"""

import argparse
import io
import os

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from visru.tools import ToolContext, ToolResult, open_workspace
from visru.tools import navigate, search, verify

# Bad input from the model, not a crash: a swallowed message ("Error executing tool
# view") leaves the agent guessing, and it will retry the same wrong call.
USER_ERRORS = (ValueError, KeyError, IndexError, FileNotFoundError, NotADirectoryError)


def _blocks(result: ToolResult) -> list[Image | str]:
    """Images first, then the text summary carrying the artifact IDs."""
    out: list[Image | str] = []
    for img in result.images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(Image(data=buf.getvalue(), format="png"))
    out.append(result.text)
    return out


def _run(fn, *args, **kwargs) -> list[Image | str]:
    """Call a tool, turning bad input into a message the model can act on.

    ToolError reaches the model with its text intact; anything else is a crash and
    the SDK deliberately withholds the detail, logging the traceback server-side.
    """
    try:
        return _blocks(fn(*args, **kwargs))
    except USER_ERRORS as exc:
        raise ToolError(exc.args[0] if exc.args else str(exc)) from exc


def build_server(ctx: ToolContext) -> MCPServer:
    mcp = MCPServer("visru")

    @mcp.tool()
    def list_assets(prefix: str | None = None, kind: str | None = None,
                    contains: str | None = None, limit: int = 100) -> list[Image | str]:
        """List workspace assets. Filter by filename prefix, kind (image|video), or
        a substring of the path. Start here: it is cheap and tells you what exists."""
        return _run(navigate.list_assets, ctx, prefix, kind, contains, limit)

    @mcp.tool()
    def view(asset_id: str, box: list[int] | None = None,
             grid: bool = False) -> list[Image | str]:
        """Show a region of an asset at the highest resolution the model accepts.

        asset_id is an image ID, or `vid_<id>#<frame_index>` for one video frame.
        box is [x0, y0, x1, y1] in SOURCE-ASSET pixels; omit it for the whole asset.
        With grid=True a ruler labelled in source pixels is drawn on the image.
        Returns an artifact ID; cite it as evidence, and pass it to view again later.
        """
        return _run(navigate.view, ctx, asset_id, box, grid)

    @mcp.tool()
    def sample_frames(asset_id: str, n: int = 12, t0: float | None = None,
                      t1: float | None = None) -> list[Image | str]:
        """Contact sheet of n frames from a video, optionally within t0..t1 seconds.

        Each cell is labelled with a frame ref like `vid_abc123#1164`. Thumbnails are
        below native scale: call view on a cell's ref before claiming small detail.
        """
        return _run(navigate.sample_frames, ctx, asset_id, n, t0, t1)

    @mcp.tool()
    def detect(asset_id: str, queries: list[str], box: list[int] | None = None,
               threshold: float = 0.1, max_marks: int = 20) -> list[Image | str]:
        """Find objects by open-vocabulary text query, drawn as numbered marks.

        queries are short noun phrases ("a torn label", "the gripper"). box limits the
        search to a region in SOURCE-ASSET pixels; a tighter box renders at higher
        scale and finds smaller objects. Results are marks with source-pixel boxes —
        refer to them as "mark N" and pass them to segment via from_artifact.
        """
        return _run(search.detect, ctx, asset_id, queries, box, threshold, max_marks)

    @mcp.tool()
    def segment(asset_id: str, points: list[list[float]] | None = None,
                labels: list[int] | None = None, boxes: list[list[float]] | None = None,
                from_artifact: str | None = None, marks: list[int] | None = None,
                box: list[int] | None = None) -> list[Image | str]:
        """Mask objects from point, box, or mark prompts, all in SOURCE-ASSET pixels.

        points are [[x, y], ...] with labels 1 (foreground) or 0 (background), one
        object per point. from_artifact='a_0005' with marks=[1, 3] segments a previous
        detection's marks. box is the region to render and work in.

        A mask prompted from a detector box is NOT an independent check of that
        detection: it inherits the detector's error.
        """
        return _run(search.segment, ctx, asset_id, points, labels, boxes,
                    from_artifact, marks, box)

    @mcp.tool()
    def ocr(asset_id: str, box: list[int] | None = None,
            min_confidence: float = 0.5) -> list[Image | str]:
        """Read text, returning each line as a numbered mark with a source-pixel box.

        Small text needs a tight box: the region is rendered once at model resolution,
        so a whole 4K page reads far worse than a zoomed paragraph.
        """
        return _run(search.ocr, ctx, asset_id, box, min_confidence)

    @mcp.tool()
    def overlay(asset_id: str, boxes: list[list[float]] | None = None,
                labels: list[str] | None = None, from_artifact: str | None = None,
                marks: list[int] | None = None,
                box: list[int] | None = None) -> list[Image | str]:
        """Draw claimed geometry back onto the source and return it for re-inspection.

        boxes are [[x0, y0, x1, y1], ...] in SOURCE-ASSET pixels. from_artifact='a_0006'
        replays an earlier detection or segmentation, masks included, onto this render —
        use it to check at native scale something you found while zoomed out.

        Look at what comes back. A shape that does not sit on the thing it names means
        the claim behind it is wrong.
        """
        return _run(verify.overlay, ctx, asset_id, boxes, labels, from_artifact, marks, box)

    @mcp.tool()
    def submit_report(claims: list[dict]) -> list[Image | str]:
        """Submit the final claims with their evidence. This ends the task.

        Each claim is an object:
          claim_id: "c_1"
          text:     what you are asserting
          status:   verified | supported | contested | hypothesis
          evidence: ["a_0005", "a_0007"] — artifact IDs that show it
          checks:   [{"kind": "cross_tool", "methods": ["detect", "segment"],
                      "artifacts": ["a_0005", "a_0006"], "result": "agree"}]

        The ledger decides what your evidence warrants and rejects any status stronger
        than that, listing every problem at once:
          - a claim with no evidence must be `hypothesis`
          - `verified` needs a passing check whose two methods do not share a model —
            a mask prompted from a detector box inherits the detector's errors
          - if any check disagrees, the claim is `contested`
        """
        return _run(verify.submit_report, ctx, claims)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="visru vision tools over MCP")
    parser.add_argument("--workspace", default=os.environ.get("VISRU_WORKSPACE", "."),
                        help="folder that `visru index` has scanned")
    args = parser.parse_args()
    build_server(open_workspace(args.workspace)).run(transport="stdio")


if __name__ == "__main__":
    main()
