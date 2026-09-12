"""Question sets for the Phase 0 exit criterion.

A question file is JSONL, one object per line:

    {"id": "q01",
     "asset": "scene_b_1546479680612194.jpg",   # filename or asset ID
     "question": "What does the vertical neon sign on the right read?",
     "answer": "TRADER JOE'S",                  # null until a human labels it
     "accept": ["trader joes"],                 # optional extra spellings
     "kind": "text"}                            # text | count | localize | temporal

`answer` is ground truth a person established by looking. Do not seed it from an
agent's own report: grading a run against answers that run produced measures
nothing.

Check a set before spending API calls on it:

    python -m evals.suite evals/questions.jsonl --workspace sample_data
"""

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("text", "count", "localize", "temporal")


@dataclass(frozen=True)
class Question:
    id: str
    asset: str
    question: str
    answer: str | None = None
    accept: list[str] = field(default_factory=list)
    kind: str = "text"
    box: list[int] | None = None  # optional region hint, for localize questions
    note: str = ""

    @property
    def labelled(self) -> bool:
        return self.answer is not None and str(self.answer).strip() != ""


def load(path: str | Path) -> list[Question]:
    questions, seen = [], set()
    for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: {exc}") from None
        unknown = set(raw) - {f for f in Question.__dataclass_fields__}
        if unknown:
            raise ValueError(f"{path}:{n}: unknown fields {sorted(unknown)}")
        q = Question(**raw)
        if q.id in seen:
            raise ValueError(f"{path}:{n}: duplicate question id {q.id!r}")
        if q.kind not in KINDS:
            raise ValueError(f"{path}:{n}: kind {q.kind!r} not in {list(KINDS)}")
        seen.add(q.id)
        questions.append(q)
    return questions


def resolve(registry, asset: str) -> str:
    """A question names a file; the tools need an asset ID."""
    if asset.startswith(("im_", "vid_")):
        return asset
    matches = [a for a in registry.list() if a.name == asset or asset in a.paths]
    if not matches:
        raise KeyError(f"no asset named {asset!r} in the workspace; run `visru index`")
    if len(matches) > 1:
        raise KeyError(f"{asset!r} matches {[a.id for a in matches]}")
    return matches[0].id


def normalise(text: str) -> str:
    """Lowercase, drop punctuation and spacing, so "TRADER JOE'S" == "trader joes".

    Apostrophes are deleted rather than spaced, or the possessive splits into a
    stray token and a correct answer grades as wrong.
    """
    without_quotes = re.sub(r"['’ʼ]", "", str(text).lower())
    return re.sub(r"[^a-z0-9]+", " ", without_quotes).strip()


def grade(question: Question, answer: str) -> bool | None:
    """True/False when it can be decided mechanically, None when a human must look."""
    if not question.labelled or answer is None:
        return None
    given, truth = normalise(answer), normalise(question.answer)
    accepted = [truth, *(normalise(a) for a in question.accept)]

    if question.kind == "count":
        numbers = re.findall(r"-?\d+", given)
        # A bare number is gradeable; a sentence with several is not.
        return numbers[0] == re.findall(r"-?\d+", truth)[0] if len(set(numbers)) == 1 else None
    if any(given == a for a in accepted):
        return True
    # A sentence containing the truth is usually right, but "not MANCHESTER" contains
    # it too, so a person settles those rather than the grader guessing generously.
    return None if any(a in given for a in accepted) else False


def main(argv: list[str] | None = None) -> int:
    """Validate a question file, and optionally that every asset it names exists."""
    parser = argparse.ArgumentParser(description="check a question set")
    parser.add_argument("questions")
    parser.add_argument("--workspace", default=None,
                        help="also resolve every asset against this indexed workspace")
    args = parser.parse_args(argv)

    questions = load(args.questions)
    kinds = Counter(q.kind for q in questions)
    labelled = [q for q in questions if q.labelled]
    print(f"{len(questions)} questions, {len(labelled)} labelled, "
          f"{len(questions) - len(labelled)} awaiting ground truth")
    print("  kinds: " + ", ".join(f"{k}={n}" for k, n in sorted(kinds.items())))

    missing = []
    if args.workspace:
        from visru.workspace.registry import Registry

        registry = Registry.load(args.workspace)
        for q in questions:
            try:
                resolve(registry, q.asset)
            except KeyError as exc:
                missing.append(f"{q.id}: {exc.args[0]}")
        print(f"  assets: {len(questions) - len(missing)}/{len(questions)} resolve "
              f"in {args.workspace}")
    for line in missing:
        print(f"  MISSING {line}")

    if len(questions) < 30:
        print(f"\nPhase 0 wants 30; this set has {len(questions)}.")
    unlabelled = [q.id for q in questions if not q.labelled]
    if unlabelled:
        print(f"unlabelled: {', '.join(unlabelled)}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
