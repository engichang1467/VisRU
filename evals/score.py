"""Score one or more runs against the question set.

    python -m evals.score --questions evals/questions.jsonl \
        evals/runs/baseline_*.jsonl evals/runs/tools_*.jsonl

A run file is JSONL of {"id", "answer"} — whatever wrote it. Record the tool-arm
answers by hand in the same shape; add "correct": true/false to a line to override
the grader, which is how the cases it marks `review` get settled.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from evals import suite


def load_run(path: str | Path) -> dict[str, dict]:
    rows = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def score_run(questions: list[suite.Question], run: dict[str, dict]) -> dict:
    per_question, tally = {}, Counter()
    for q in questions:
        row = run.get(q.id)
        if row is None:
            verdict = "missing"
        elif "correct" in row:  # a human settled it
            verdict = "correct" if row["correct"] else "wrong"
        elif not q.labelled:
            verdict = "unlabelled"
        else:
            graded = suite.grade(q, row.get("answer"))
            verdict = {True: "correct", False: "wrong", None: "review"}[graded]
        per_question[q.id] = verdict
        tally[verdict] += 1

    decided = tally["correct"] + tally["wrong"]
    return {"per_question": per_question, "tally": tally, "decided": decided,
            "accuracy": tally["correct"] / decided if decided else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)

    questions = suite.load(args.questions)
    unlabelled = [q.id for q in questions if not q.labelled]
    results = {Path(p).name: score_run(questions, load_run(p)) for p in args.runs}

    width = max(len(n) for n in results) if results else 10
    print(f"{'run':{width}}  correct  wrong  review  missing  accuracy")
    print("-" * (width + 42))
    for name, res in results.items():
        t = res["tally"]
        acc = f"{res['accuracy']:.0%}" if res["accuracy"] is not None else "—"
        print(f"{name:{width}}  {t['correct']:>7}  {t['wrong']:>5}  {t['review']:>6}  "
              f"{t['missing']:>7}  {acc:>8}  (of {res['decided']} decided)")

    if unlabelled:
        print(f"\n{len(unlabelled)} of {len(questions)} questions have no ground truth "
              f"and were not scored: {', '.join(unlabelled[:8])}"
              f"{' …' if len(unlabelled) > 8 else ''}")

    # Per-question disagreement is the interesting part: the exit criterion is about
    # questions single-pass fails and tools get right.
    if len(results) > 1:
        names = list(results)
        print(f"\n{'question':10}  " + "  ".join(f"{n[:18]:18}" for n in names))
        for q in questions:
            row = [results[n]["per_question"][q.id] for n in names]
            if len(set(row)) > 1:
                print(f"{q.id:10}  " + "  ".join(f"{v:18}" for v in row))

    for name, res in results.items():
        if res["tally"]["review"]:
            print(f"\n{name}: {res['tally']['review']} answers need a human verdict — "
                  "add \"correct\": true/false to those lines and re-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
