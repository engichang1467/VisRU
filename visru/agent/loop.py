"""The agent loop: ask the model, run what it asks for, write it all down.

Deliberately thin. Everything hard lives elsewhere — the image budget in
`context.py`, the record in `trajectory.py`, what the evidence warrants in
`evidence/ledger.py` — and this file is what makes them take turns.

The loop ends one of three ways: the model submits a report the ledger accepts,
the model stops asking for tools, or it runs out of steps. All three are results;
only the first is an answer.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visru.agent.context import Context, ToolCall
from visru.agent.trajectory import Trajectory
from visru.backends import Backend, Response
from visru.tools import ToolContext, ToolResult, registry
from visru.tools.registry import USER_ERRORS

PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"

# Said when the model stops calling tools without reporting. Without this a loop
# just spins: the model says "I think it is a stop sign", nobody asks for
# evidence, and 200 steps later there is still no report.
NUDGE = ("Finish by calling submit_report with a claim for each thing you are "
         "asserting, each with the artifact IDs that show it.")


def system_prompt() -> str:
    return PROMPT_PATH.read_text()


@dataclass
class Session:
    """How a run ended, and where to read what happened."""

    session: str
    steps: int
    reason: str  # reported | stopped | exhausted
    report: str = ""
    trajectory: Path | None = None
    elisions: int = 0
    compactions: int = 0
    peak_live_images: int = 0
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def reported(self) -> bool:
        return self.reason == "reported"

    def summary(self) -> str:
        return (f"{self.session}: {self.reason} after {self.steps} steps "
                f"({self.elisions} images elided, {self.compactions} compactions, "
                f"peak {self.peak_live_images} live)")


def _accrue(total: dict[str, int], usage: dict[str, int]) -> None:
    for key, value in usage.items():
        total[key] = total.get(key, 0) + value


def run(task: str, ctx: ToolContext, backend: Backend, max_steps: int = 200,
        system: str | None = None, tools: list[str] | None = None) -> Session:
    """Drive one session to an answer, a stop, or the step limit.

    `tools` limits which tools the model is offered and may call; None means all.
    """
    system = system_prompt() if system is None else system
    context = Context(task=task, cfg=ctx.cfg, system=system)
    ctx.backend, ctx.child_usage = backend, {}  # for `delegate`
    log = Trajectory(ctx.root, ctx.session)
    schemas = registry.schemas(tools)
    log.session_header(ctx.cfg, task, system, [s["name"] for s in schemas])

    usage: dict[str, int] = {}
    peak, step, nudged = 0, -1, False
    reason, report = "exhausted", ""

    for step in range(max_steps):
        response: Response = backend.complete(system, context.render(), schemas)
        _accrue(usage, response.usage)
        log.model_turn(step, response.text, response.usage)
        context.add_turn(response.text, response.calls, raw=response.raw)

        if not response.calls:
            # Ask once for a report. If it still calls nothing, it has finished
            # thinking, and spending the rest of the budget to prove that is waste.
            if nudged:
                reason = "stopped"
                break
            nudged = True
            context.add_user(NUDGE)
            continue
        nudged = False

        for call in response.calls:
            result, error = _execute(ctx, call, tools)
            context.add_output(call, result)
            log.tool_call(step, call.name, call.arguments, result.text,
                          result.artifacts, shown=bool(result.images), error=error)
            peak = max(peak, context.live_images)

            if call.name == "submit_report" and error is None:
                reason, report = "reported", result.text
                break
        if reason == "reported":
            break

    _accrue(usage, ctx.child_usage)  # subagents ran on this session's behalf
    return Session(session=ctx.session, steps=step + 1, reason=reason, report=report,
                   trajectory=log.path, elisions=context.elisions,
                   compactions=context.compactions,
                   peak_live_images=max(peak, context.live_images), usage=usage)


def _execute(ctx: ToolContext, call: ToolCall,
             allowed: list[str] | None = None) -> tuple[ToolResult, str | None]:
    """Run one call. Bad input comes back as a message, not an exception.

    A raised error would end the session; the model can usually fix a bad box or a
    misspelled asset ID on its next turn, and it only can if it is told what was
    wrong. This mirrors what the MCP server does with ToolError.
    """
    try:
        return registry.execute(ctx, call.name, call.arguments, allowed), None
    except USER_ERRORS as exc:
        message = exc.args[0] if exc.args else str(exc)
        return ToolResult(text=f"error: {message}"), str(message)
