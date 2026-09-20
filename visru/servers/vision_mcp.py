"""MCP server exposing visru's tools, so any MCP client can drive them.

mcp 2.x renamed FastMCP to MCPServer; the tool decorator is otherwise the same.

The wrappers below carry signatures only. Names, descriptions, argument shapes
and dispatch all come from `visru.tools.registry`, which the Phase 1 agent loop
reads too — the prompt text that tells the model what these tools mean must not
exist twice. MCP derives its JSON schema by introspecting a real function, so
the signatures stay here; `test_server_signatures_match_the_registry` is what
stops them drifting from the table.
"""

import argparse
import io
import os
from typing import Any

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from visru.tools import ToolContext, ToolResult, open_workspace
from visru.tools import registry
from visru.tools.registry import USER_ERRORS


def _blocks(result: ToolResult) -> list[Image | str]:
    """Images first, then the text summary carrying the artifact IDs."""
    out: list[Image | str] = []
    for img in result.images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(Image(data=buf.getvalue(), format="png"))
    out.append(result.text)
    return out


def _call(ctx: ToolContext, name: str, **arguments: Any) -> list[Image | str]:
    """Dispatch through the registry, turning bad input into a message the model
    can act on.

    ToolError reaches the model with its text intact; anything else is a crash and
    the SDK deliberately withholds the detail, logging the traceback server-side.
    """
    try:
        return _blocks(registry.execute(ctx, name, arguments))
    except USER_ERRORS as exc:
        raise ToolError(exc.args[0] if exc.args else str(exc)) from exc


def build_server(ctx: ToolContext) -> MCPServer:
    mcp = MCPServer("visru")
    desc = {name: spec.description for name, spec in registry.TOOLS.items()}

    @mcp.tool(description=desc["list_assets"])
    def list_assets(prefix: str | None = None, kind: str | None = None,
                    contains: str | None = None, limit: int = 100) -> list[Image | str]:
        return _call(ctx, "list_assets", prefix=prefix, kind=kind,
                     contains=contains, limit=limit)

    @mcp.tool(description=desc["view"])
    def view(asset_id: str, box: list[int] | None = None,
             grid: bool = False) -> list[Image | str]:
        return _call(ctx, "view", asset_id=asset_id, box=box, grid=grid)

    @mcp.tool(description=desc["sample_frames"])
    def sample_frames(asset_id: str, n: int = 12, t0: float | None = None,
                      t1: float | None = None) -> list[Image | str]:
        return _call(ctx, "sample_frames", asset_id=asset_id, n=n, t0=t0, t1=t1)

    @mcp.tool(description=desc["detect"])
    def detect(asset_id: str, queries: list[str], box: list[int] | None = None,
               threshold: float = 0.1, max_marks: int = 20) -> list[Image | str]:
        return _call(ctx, "detect", asset_id=asset_id, queries=queries, box=box,
                     threshold=threshold, max_marks=max_marks)

    @mcp.tool(description=desc["segment"])
    def segment(asset_id: str, points: list[list[float]] | None = None,
                labels: list[int] | None = None, boxes: list[list[float]] | None = None,
                from_artifact: str | None = None, marks: list[int] | None = None,
                box: list[int] | None = None) -> list[Image | str]:
        return _call(ctx, "segment", asset_id=asset_id, points=points, labels=labels,
                     boxes=boxes, from_artifact=from_artifact, marks=marks, box=box)

    @mcp.tool(description=desc["ocr"])
    def ocr(asset_id: str, box: list[int] | None = None,
            min_confidence: float = 0.5) -> list[Image | str]:
        return _call(ctx, "ocr", asset_id=asset_id, box=box,
                     min_confidence=min_confidence)

    @mcp.tool(description=desc["overlay"])
    def overlay(asset_id: str, boxes: list[list[float]] | None = None,
                labels: list[str] | None = None, from_artifact: str | None = None,
                marks: list[int] | None = None,
                box: list[int] | None = None) -> list[Image | str]:
        return _call(ctx, "overlay", asset_id=asset_id, boxes=boxes, labels=labels,
                     from_artifact=from_artifact, marks=marks, box=box)

    @mcp.tool(description=desc["submit_report"])
    def submit_report(claims: list[dict]) -> list[Image | str]:
        return _call(ctx, "submit_report", claims=claims)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="visru vision tools over MCP")
    parser.add_argument("--workspace", default=os.environ.get("VISRU_WORKSPACE", "."),
                        help="folder that `visru index` has scanned")
    args = parser.parse_args()
    build_server(open_workspace(args.workspace)).run(transport="stdio")


if __name__ == "__main__":
    main()
