"""Single-pass baseline: the model answers from one downsized image, no tools.

This is the control the Phase 0 exit criterion is measured against. Without it,
"the tool setup answered 20 of 30" is a number with nothing to compare to.

    python -m evals.baseline --workspace sample_data --questions evals/questions.jsonl
    python -m evals.baseline ... --dry-run     # render and count, call nothing
    python -m evals.baseline ... --backend openai_compatible --config open.toml

The model comes from visru.toml like every other arm, so the same question set
runs against Anthropic, a hosted open model or a local vLLM by pointing
`--config` at a different file.

Video questions write a contact-sheet artifact into the workspace, so what the
baseline was shown stays inspectable. The run prints their IDs at the end — delete
them if you want the workspace to hold only the agent's own session.
"""

import argparse
import base64
import io
import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from evals import suite
from visru import backends
from visru.config import load_config
from visru.tools import open_workspace, navigate
from visru.tools.navigate import render_region

PROMPT = ("Answer the question about this image. Reply with the answer only — no "
          "explanation, no hedging. If you cannot tell, reply exactly: cannot tell.")


def _png_b64(image) -> tuple[str, int]:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode(), buf.tell()


def run(workspace: str, questions_path: str, out_dir: str = "evals/runs",
        model: str | None = None, dry_run: bool = False, frames: int = 12,
        sample: int | None = None, seed: int = 0, backend: str | None = None,
        config: str | None = None, script: str | None = None) -> Path:
    # Same config path as evals.agent: the open arm wants its own render limits,
    # and the baseline must be rendered under whatever limits its model has.
    ctx = open_workspace(workspace, cfg=load_config(config) if config else None)
    overrides = {k: v for k, v in (("name", model), ("backend", backend)) if v}
    if script:
        overrides["backend"] = "replay"
    ctx.cfg = replace(ctx.cfg, model=replace(ctx.cfg.model, **overrides))
    model = ctx.cfg.model.name
    questions = suite.load(questions_path)
    if sample is not None:  # the same fixed-seed subset evals.agent draws
        questions = suite.select(questions, sample, seed)

    # No tools, one user turn: the backend contract covers the whole baseline, so
    # Anthropic, vLLM and replay all run this arm without a branch here.
    driver = None if dry_run else backends.load(
        ctx.cfg, **({"path": script} if script else {}))

    created: list[str] = []
    out = Path(out_dir) / f"baseline_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as sink:
        for q in questions:
            ref = suite.resolve(ctx.registry, q.asset)
            if ctx.registry.get(ref).kind == "video":
                # "One downsized image" has no meaning for a video. The fair control
                # is the one call a person would make by hand: a contact sheet. It is
                # saved as an artifact, so what the baseline saw stays inspectable.
                sheet = navigate.sample_frames(ctx, ref, n=frames)
                image, scale, kind = sheet.images[0], None, f"contact_sheet_{frames}"
                record_extra = {"input_artifact": sheet.artifacts[0]}
            else:
                # The whole asset at model resolution: what a single pass actually sees.
                image, tf, _ = render_region(ctx, ref)
                scale, kind, record_extra = round(tf.scale, 3), "full_frame", {}

            record = {"id": q.id, "asset": ref, "arm": "single_pass", "model": model,
                        "backend": ctx.cfg.model.backend, "input": kind,
                        "render_px": list(image.size), "scale": scale,
                        "category": q.category, **record_extra}

            if dry_run:
                record["answer"] = None
            else:
                response = driver.complete(
                    PROMPT,
                    [{"role": "user", "content": [{"type": "image", "image": image},
                                                  {"type": "text",
                                                   "text": suite.prompt(q)}]}],
                    [])
                record["answer"] = response.text.strip()
                record["usage"] = response.usage
            created.extend(record_extra.values())
            sink.write(json.dumps(record) + "\n")
            print(f"{q.id} {kind} {image.size[0]}x{image.size[1]} "
                  f"-> {record['answer']!r}", flush=True)
    if created:
        print(f"\nbaseline wrote {len(created)} contact-sheet artifacts into the "
              f"workspace: {', '.join(created)}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out-dir", default="evals/runs")
    parser.add_argument("--model", default=None, help="defaults to visru.toml")
    parser.add_argument("--backend", default=None,
                        help="anthropic | openai_compatible; defaults to visru.toml. "
                            "Pin the provider in --model for a hosted open model, "
                            "e.g. Qwen/Qwen3-VL-30B-A3B-Instruct:novita")
    parser.add_argument("--config", default=None,
                        help="visru.toml for this arm; defaults to the workspace's")
    parser.add_argument("--frames", type=int, default=12,
                        help="contact-sheet frames for video questions")
    parser.add_argument("--sample", type=int, default=None,
                        help="fixed-seed subset of labelled questions, as evals.agent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="render every question's image but call no API")
    parser.add_argument("--script", default=None,
                       help="replay this turn script instead of calling a model")
    args = parser.parse_args(argv)

    out = run(args.workspace, args.questions, args.out_dir, args.model,
              args.dry_run, args.frames, args.sample, args.seed, args.backend,
              args.config, args.script)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
