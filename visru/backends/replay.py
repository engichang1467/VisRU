"""A backend that answers from a script instead of a model.

This is what makes the Phase 1 exit criterion affordable. Two hundred real steps
over a thousand images is tens of dollars a run, and the thing being tested —
that the context manager keeps a long session inside its limits — has nothing to
do with what the model would actually say. Replay drives the same loop, the same
tools, and the same context manager for free, as many times as a bug takes.

It is not a substitute for a real run. It proves the harness holds; only a live
model proves the harness works.

A script is JSON: a list of turns, each with optional `text`, `calls`
(`{"name", "arguments"}`) and `usage`. Running past the end returns a turn with
no calls, which the loop reads as the model having finished.
"""

import json
from pathlib import Path
from typing import Any

from visru.agent.context import ToolCall
from visru.backends.base import Response
from visru.config import Config

DONE = "That is everything I can determine from the script."


class ReplayBackend:
    name = "replay"

    def __init__(self, cfg: Config, script: list[dict[str, Any]] | None = None,
                 path: str | Path | None = None):
        if (script is None) == (path is None):
            raise ValueError("replay needs exactly one of script= or path=")
        if path is not None:
            script = json.loads(Path(path).read_text())
        if not isinstance(script, list):
            raise ValueError("a replay script is a JSON list of turns")

        self.script = script
        self.position = 0
        self.requests: list[dict[str, Any]] = []

    def complete(self, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]]) -> Response:
        # Kept so a test can assert on what the model would have been sent —
        # how many images survived elision, whether the prompt was attached.
        self.requests.append({"system": system, "messages": messages, "tools": tools})

        if self.position >= len(self.script):
            return Response(text=DONE, stop_reason="end_turn")

        turn = self.script[self.position]
        self.position += 1
        calls = [ToolCall(id=turn.get("id") or f"r{self.position}_{i}",
                          name=call["name"], arguments=call.get("arguments", {}))
                 for i, call in enumerate(turn.get("calls", []))]
        return Response(text=turn.get("text", ""), calls=calls,
                        usage=turn.get("usage", {}),
                        stop_reason="tool_use" if calls else "end_turn")


def sweep(assets: list[str], box: list[int] | None = None) -> list[dict[str, Any]]:
    """A script that views every asset in turn.

    The shape a navigation session actually has, and the one that stresses the
    image budget hardest: a new render every step, none of them reused.
    """
    arguments: dict[str, Any] = {}
    return [{"text": f"Looking at {asset}.",
             "calls": [{"name": "view",
                        "arguments": {"asset_id": asset,
                                      **({"box": box} if box else arguments)}}]}
            for asset in assets]
