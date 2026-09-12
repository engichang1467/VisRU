"""Single-pass baseline: the model answers from one downsized image, no tools.

This is the control the Phase 0 exit criterion is measured against. Without it,
"the tool setup answered 20 of 30" is a number with nothing to compare to.

    python -m evals.baseline --workspace sample_data --questions evals/questions.jsonl
    python -m evals.baseline ... --dry-run     # render and count, call nothing

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
from datetime import datetime
from pathlib import Path

from evals import suite
from visru.tools import open_workspace, navigate
from visru.tools.navigate import render_region

PROMPT = ("Answer the question about this image. Reply with the answer only — no "
          "explanation, no hedging. If you cannot tell, reply exactly: cannot tell.")


def _png_b64(image) -> tuple[str, int]:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode(), buf.tell()


def run(workspace: str, questions_path: str, out_dir: str = "evals/runs",
        model: str | None = None, dry_run: bool = False, frames: int = 12) -> Path:
    ctx = open_workspace(workspace)
    questions = suite.load(questions_path)
    model = model or ctx.cfg.model.name

    client = None
    if not dry_run:
        import anthropic

        # An org-scoped key must say which workspace to bill; a workspace-scoped
        # key already knows, and the header is absent.
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        client = anthropic.Anthropic(
            default_headers={"anthropic-workspace-id": ws} if ws else None)

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

            data, nbytes = _png_b64(image)
            record = {"id": q.id, "asset": ref, "arm": "single_pass", "model": model,
                      "input": kind, "render_px": list(image.size), "scale": scale,
                      "png_kb": round(nbytes / 1024), **record_extra}

            if dry_run:
                record["answer"] = None
            else:
                message = client.messages.create(
                    model=model, max_tokens=256,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64",
                                                     "media_type": "image/png",
                                                     "data": data}},
                        {"type": "text", "text": f"{PROMPT}\n\n{q.question}"}]}])
                record["answer"] = "".join(b.text for b in message.content
                                           if b.type == "text").strip()
                record["usage"] = {"input": message.usage.input_tokens,
                                   "output": message.usage.output_tokens}
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
    parser.add_argument("--frames", type=int, default=12,
                        help="contact-sheet frames for video questions")
    parser.add_argument("--dry-run", action="store_true",
                        help="render every question's image but call no API")
    args = parser.parse_args(argv)

    out = run(args.workspace, args.questions, args.out_dir, args.model,
              args.dry_run, args.frames)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
