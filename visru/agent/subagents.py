"""Subagents: one agent per time window of a video, run one after another.

A long video does not fit in one context at useful resolution: an hour at one
contact sheet a minute is sixty sheets, and the image budget holds sixteen. So
the parent splits the video into windows and hands each to a child session with
its own context, trajectory and ledger. The parent gets back text — each child's
claims and the artifact IDs behind them — never the child's images, so its own
context stays small, and it can `view` any of those artifacts itself.

A child is an ordinary `loop.run` on a session named `<parent>_w<k>`. That name
is how coverage credits the parent with what its children looked at.

ponytail: sequential. Children in threads would need a lock around artifact IDs
and one model instance per thread; add when wall-clock on long videos hurts.
"""

import json

from visru.evidence.ledger import LEDGER_DIR
from visru.tools import ToolContext, ToolResult, resolve_ref
from visru.workspace.registry import parse_ref

MAX_WINDOWS = 8

CHILD_TASK = """You are one of several investigators, each given one time window of the \
same video.

Video: {asset} ({name}), {duration:.1f} s long.
Your window: {t0:.2f}..{t1:.2f} s. Look only there; others cover the rest.

Question: {question}

Investigate with your tools — sample_frames with t0/t1 inside your window is the \
place to start — then call submit_report. If what the question asks about happens \
in your window, give a claim whose text states its time window as "start..end" in \
seconds. If it does not, give a claim saying so, citing the frames you looked at. \
Every claim needs the artifact IDs that show it."""


def _windows(duration: float, windows: list[list[float]] | None, n: int
             ) -> list[tuple[float, float]]:
    if windows is None:
        if not 1 <= n <= MAX_WINDOWS:
            raise ValueError(f"n must be 1..{MAX_WINDOWS}, got {n}")
        return [(duration * i / n, duration * (i + 1) / n) for i in range(n)]
    if not windows or len(windows) > MAX_WINDOWS:
        raise ValueError(f"give 1..{MAX_WINDOWS} windows as [[t0, t1], ...] in seconds")
    out = []
    for w in windows:
        if len(w) != 2:
            raise ValueError(f"each window is [t0, t1] in seconds, got {w}")
        t0, t1 = max(0.0, float(w[0])), min(duration, float(w[1]))
        if t1 <= t0:
            raise ValueError(f"window {w} is empty inside the video's 0..{duration:.2f}s")
        out.append((t0, t1))
    return out


def _claims(ctx: ToolContext, session: str) -> list[dict]:
    path = ctx.root / LEDGER_DIR / f"{session}.json"
    return json.loads(path.read_text())["claims"] if path.is_file() else []


def _claim_lines(head: str, claims: list[dict], cited: list[str]) -> str:
    """One window's report as text; the artifact IDs it cites are added to `cited`."""
    lines = [head]
    for c in claims:
        refs = list(c.get("evidence", [])) + [
            k["check"] for k in c.get("checks", []) if isinstance(k, dict)
            and k.get("check")]
        cited.extend(refs)
        lines.append(f"  {c['claim_id']} [{c['status']}] {c['text']}"
                     + (f"  evidence: {', '.join(refs)}" if refs else ""))
    if not claims:
        lines.append("  no report: treat this window as not inspected")
    return "\n".join(lines)


def delegate(ctx: ToolContext, ref: str, question: str,
             windows: list[list[float]] | None = None, n: int = 4,
             max_steps: int = 15) -> ToolResult:
    """Run one subagent per time window and collect what each reported."""
    if ctx.run_child is None:
        raise ValueError("delegate runs subagents, which needs the visru agent loop; "
                         "under another harness, use its own subagents")
    if not str(question).strip():
        raise ValueError("delegate needs the question each subagent should answer")
    asset_id, _ = parse_ref(resolve_ref(ctx, ref)[0])
    asset = ctx.registry.get(asset_id)
    if asset.kind != "video":
        raise ValueError(f"{asset_id} is an image; delegate splits a video by time")
    duration = asset.duration or 0.0
    spans = _windows(duration, windows, n)

    blocks, cited = [], []
    for t0, t1 in spans:
        ctx.children += 1
        child = ToolContext(ctx.root, ctx.cfg, session=f"{ctx.session}_w{ctx.children}")
        task = CHILD_TASK.format(asset=asset_id, name=asset.name, duration=duration,
                                 t0=t0, t1=t1, question=question)
        result = ctx.run_child(task, child, max_steps)  # no delegate inside: no recursion
        for key, value in result.usage.items():
            ctx.child_usage[key] = ctx.child_usage.get(key, 0) + value

        claims = _claims(ctx, child.session) if result.reported else []
        head = (f"[{t0:.2f}..{t1:.2f}s] {child.session}: {result.reason} after "
                f"{result.steps} steps")
        blocks.append(_claim_lines(head, claims, cited))

    text = (f"delegated {len(spans)} windows of {asset_id} ({duration:.1f}s) to subagents:\n"
            + "\n".join(blocks)
            + "\nTheir artifacts are yours to view or cite by ID, and a check they ran "
            "can go in your own claim's checks. Their conclusions are not yours until you "
            "have looked: view the key frames before reporting.")
    return ToolResult(text=text, artifacts=list(dict.fromkeys(cited)))

