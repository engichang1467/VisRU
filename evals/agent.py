"""Agent arms: the same question set through the visru loop, with a tool subset.

    python -m evals.agent --workspace bench/vstar --questions bench/vstar/questions.jsonl \
        --arm full --sample 100
    python -m evals.agent ... --arm zoom_only --out evals/runs/zoom_only_vstar.jsonl

Arms (README "Baselines"): `single_pass` lives in evals/baseline.py; here are
`zoom_only` (list_assets, view, submit_report), `no_subagents` (every tool but
delegate) and `full` (every tool). Subagent sessions count toward a question's
tool calls and visual tokens.

Each question is its own session, so its trajectory, ledger and coverage stand
alone. The model is told to put its final answer in a claim with claim_id
`answer`; that claim's text is the answer and its status is what the
calibration table groups by. Rows are written as they finish, and re-running
with the same --out skips questions already answered, so a crashed paid run
resumes instead of starting over.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from evals import suite
from visru import backends
from visru.agent import loop
from visru.agent.trajectory import read as read_log, shown_artifacts
from visru.config import load_config
from visru.evidence.ledger import LEDGER_DIR
from visru.geometry import visual_tokens
from visru.tools import open_workspace
from visru.tools.registry import TOOLS

ARMS = {
    "zoom_only": ["list_assets", "view", "submit_report"],
    "no_subagents": [name for name in TOOLS if name != "delegate"],
    "full": None,  # every tool in the registry
}

TASK = """Question about asset {asset} ({name}):

{prompt}

Investigate with the tools you have, then call submit_report. Include one claim
with claim_id "answer" whose text is only your final answer{choice}. Give that
claim the status your evidence warrants; add other claims if they help."""


ANSWER_FORM = {"choice": " — the option letter",
               "temporal": " — the time window as start..end in seconds"}


def answer_claim(root: Path, session: str) -> dict | None:
    path = root / LEDGER_DIR / f"{session}.json"
    if not path.is_file():
        return None
    claims = json.loads(path.read_text())["claims"]
    return next((c for c in claims if c["claim_id"] == "answer"), None)


def run_one(workspace: str, q: suite.Question, arm: str, run_id: str, model: str | None,
            max_steps: int, script: str | None = None,
            backend: str | None = None, config: str | None = None) -> dict:
    session = f"{run_id}_{q.id}"
    # An arm's config is not the workspace's: the open path wants its own render
    # limits, and both arms share one indexed workspace.
    ctx = open_workspace(workspace, cfg=load_config(config) if config else None,
                         session=session)
    overrides = {"name": model} if model else {}
    if backend:
        overrides["backend"] = backend
    if script:
        overrides["backend"] = "replay"
    ctx.cfg = replace(ctx.cfg, model=replace(ctx.cfg.model, **overrides))
    ref = suite.resolve(ctx.registry, q.asset)
    task = TASK.format(asset=ref, name=ctx.registry.get(ref).name, prompt=suite.prompt(q),
                       choice=ANSWER_FORM.get(q.kind, ""))

    started = time.monotonic()
    driver = backends.load(ctx.cfg, **({"path": script} if script else {}))
    result = loop.run(task, ctx, driver,
                      max_steps=max_steps, tools=ARMS[arm])
    wall = time.monotonic() - started

    records = read_log(result.trajectory)
    for child in sorted(result.trajectory.parent.glob(f"{session}_w*.jsonl")):
        records += read_log(child)  # subagents looked on this question's behalf
    patch = ctx.cfg.model.patch
    shown = shown_artifacts(records)
    tokens = 0
    for artifact_id in shown:
        im = ctx.artifacts.load_image(artifact_id)
        tokens += visual_tokens(im.width, im.height, patch)
    claim = answer_claim(ctx.root, session)
    return {"id": q.id, "arm": arm, "model": ctx.cfg.model.name,
            "backend": ctx.cfg.model.backend, "session": session,
            "answer": claim["text"] if claim else None,
            "status": claim["status"] if claim else None,
            "reason": result.reason, "steps": result.steps,
            "tool_calls": sum(r.kind == "tool" for r in records),
            "images_shown": len(shown), "visual_tokens": tokens,
            "usage": result.usage, "wall_s": round(wall, 1), "category": q.category}


def run(workspace: str, questions_path: str, arm: str, out: str | None = None,
        model: str | None = None, sample: int | None = None, seed: int = 0,
        max_steps: int = 30, script: str | None = None,
        backend: str | None = None, config: str | None = None) -> Path:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {list(ARMS)}; single_pass is evals.baseline")
    questions = suite.select(suite.load(questions_path), sample, seed)
    out_path = Path(out or f"evals/runs/{arm}_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.is_file():
        done = {json.loads(line)["id"] for line in out_path.read_text().splitlines()
                if line.strip()}
    run_id = f"s_{arm}_{out_path.stem}"

    # ponytail: one question at a time. ArtifactStore IDs race between threads in one
    # process; run arms or benchmarks as separate processes on separate workspaces.
    for i, q in enumerate(questions, start=1):
        if q.id in done:
            continue
        row = run_one(workspace, q, arm, run_id, model, max_steps, script, backend,
                      config)
        with out_path.open("a") as sink:
            sink.write(json.dumps(row) + "\n")
        print(f"[{i}/{len(questions)}] {q.id} {row['reason']} steps={row['steps']} "
              f"-> {row['answer']!r} ({row['status']})", flush=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    suite.add_run_args(parser)
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--out", default=None, help="run file; an existing one resumes")
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args(argv)

    out = run(args.workspace, args.questions, args.arm, args.out, args.model,
              args.sample, args.seed, args.max_steps, args.script, args.backend,
              args.config)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
