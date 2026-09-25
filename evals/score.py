"""Score one or more runs against the question set.

    python -m evals.score --questions evals/questions.jsonl \
        evals/runs/baseline_*.jsonl evals/runs/tools_*.jsonl

A run file is JSONL of {"id", "answer"} — whatever wrote it. Add "correct":
true/false to a line to override the grader, which is how the cases it marks
`review` get settled.

Rows from evals.agent also carry the answer claim's ledger `status`; for those the
scorer prints accuracy per status — the Phase 2 question of whether `verified`
predicts correctness any better than `supported`.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for k of n. On a few dozen claims a gap is often just noise."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def calibration(run: dict[str, dict], verdicts: dict[str, str]) -> dict[str, tuple[int, int]]:
    """(correct, decided) per ledger status of the answer claim."""
    table: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for qid, row in run.items():
        verdict = verdicts.get(qid)
        if verdict in ("correct", "wrong"):
            cell = table[row.get("status") or "no report"]
            cell[0] += verdict == "correct"
            cell[1] += 1
    return {k: (v[0], v[1]) for k, v in table.items()}


def _mean(rows: list[dict], key) -> float | None:
    values = [key(r) for r in rows]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def cost(run: dict[str, dict]) -> dict[str, float | None]:
    rows = list(run.values())
    return {"tool_calls": _mean(rows, lambda r: r.get("tool_calls")),
            "visual_tokens": _mean(rows, lambda r: r.get("visual_tokens")),
            "input_tokens": _mean(rows, lambda r: (r.get("usage") or {}).get("input")),
            "wall_s": _mean(rows, lambda r: r.get("wall_s"))}


def _fmt(v, spec):
    return "—" if v is None else format(v, spec)


def _print_accuracy(questions, results, width):
    unlabelled = [q.id for q in questions if not q.labelled]
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


def _print_cost(runs, width):
    print(f"\n{'run':{width}}  tool calls  visual tok  input tok  wall s   (means per question)")
    for name, run in runs.items():
        c = cost(run)
        print(f"{name:{width}}  {_fmt(c['tool_calls'], '>10.1f')}  "
              f"{_fmt(c['visual_tokens'], '>10,.0f')}  {_fmt(c['input_tokens'], '>9,.0f')}  "
              f"{_fmt(c['wall_s'], '>6.1f')}")


def _print_categories(questions, results):
    categories = sorted({q.category for q in questions if q.category})
    if not categories:
        return
    print(f"\n{'category':20}  " + "  ".join(f"{n[:18]:>18}" for n in results))
    for cat in categories:
        ids = {q.id for q in questions if q.category == cat}
        cells = []
        for res in results.values():
            v = [res["per_question"][i] for i in ids]
            k, n = v.count("correct"), v.count("correct") + v.count("wrong")
            cells.append(f"{k / n:.0%} of {n}" if n else "—")
        print(f"{cat:20}  " + "  ".join(f"{c:>18}" for c in cells))


def _print_calibration(runs, results):
    order = ["verified", "supported", "contested", "hypothesis", "no report"]
    for name, run in runs.items():
        if not any("status" in r for r in run.values()):
            continue
        table = calibration(run, results[name]["per_question"])
        print(f"\n{name}: accuracy by ledger status of the answer claim")
        print(f"  {'status':12}  {'n':>4}  accuracy  95% interval")
        for status in sorted(table, key=lambda s: order.index(s) if s in order else 99):
            k, n = table[status]
            lo, hi = wilson(k, n)
            print(f"  {status:12}  {n:>4}  {k / n:>8.0%}  {lo:.0%}–{hi:.0%}")


def _print_disagreements(questions, results):
    # Per-question disagreement is the interesting part: the exit criterion is about
    # questions single-pass fails and tools get right.
    if len(results) < 2:
        return
    names = list(results)
    print(f"\n{'question':10}  " + "  ".join(f"{n[:18]:18}" for n in names))
    for q in questions:
        row = [results[n]["per_question"][q.id] for n in names]
        if len(set(row)) > 1:
            print(f"{q.id:10}  " + "  ".join(f"{v:18}" for v in row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)

    questions = suite.load(args.questions)
    runs = {Path(p).name: load_run(p) for p in args.runs}
    results = {name: score_run(questions, run) for name, run in runs.items()}
    width = max(len(n) for n in results) if results else 10

    _print_accuracy(questions, results, width)
    _print_cost(runs, width)
    _print_categories(questions, results)
    _print_calibration(runs, results)
    _print_disagreements(questions, results)
    for name, res in results.items():
        needs = [qid for qid, v in res["per_question"].items() if v == "review"]
        if needs:
            print(f"\n{name}: {len(needs)} answers need a human verdict — "
                  "add \"correct\": true/false to those lines and re-run")
            print("  " + ", ".join(needs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
