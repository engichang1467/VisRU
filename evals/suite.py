"""Question sets for the Phase 0 exit criterion.

A question file is JSONL, one object per line:

    {"id": "q01",
     "asset": "scene_b_1546479680612194.jpg",   # filename or asset ID
     "question": "What does the vertical neon sign on the right read?",
     "answer": "TRADER JOE'S",                  # null until a human labels it
     "accept": ["trader joes"],                 # optional extra spellings
     "kind": "text"}                            # text | count | localize | temporal | choice

A `choice` question also carries `options` (["rubber", "cotton", ...]) and its
answer is the option letter, as in V* Bench and HR-Bench.

`answer` is ground truth a person established by looking. Do not seed it from an
agent's own report: grading a run against answers that run produced measures
nothing.

Check a set before spending API calls on it:

    python -m evals.suite evals/questions.jsonl --workspace sample_data
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from visru.tools.verify import normalise

KINDS = ("text", "count", "localize", "temporal", "choice")
LETTERS = "ABCDEFGHIJ"
TOLERANCE = 1.0  # temporal default, seconds per end, when a question names none
NUMBER = re.compile(r"\d+(?:\.\d+)?")  # times are never negative; "12-15" is a range


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
    options: list[str] = field(default_factory=list)  # choice questions, in letter order
    category: str = ""  # benchmark subset, e.g. direct_attributes, single, cross
    tolerance: float | None = None  # temporal: seconds each end may be off; fixed pre-run

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
        if (q.kind == "choice") != bool(q.options):
            raise ValueError(f"{path}:{n}: options go with kind 'choice', and only there")
        seen.add(q.id)
        questions.append(q)
    return questions


def select(questions: list[Question], sample: int | None,
           seed: int) -> list[Question]:
    """A fixed-seed sample, in file order, so every arm sees the same questions."""
    labelled = [q for q in questions if q.labelled]
    if sample is None or sample >= len(labelled):
        return labelled
    keep = set(q.id for q in random.Random(seed).sample(labelled, sample))
    return [q for q in labelled if q.id in keep]


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """The arguments every eval arm takes: where, which questions, which model."""
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--model", default=None, help="defaults to visru.toml")
    parser.add_argument("--backend", default=None,
                        help="anthropic | openai_compatible; defaults to visru.toml. "
                             "Pin the provider in --model for a hosted open model, "
                             "e.g. Qwen/Qwen3-VL-30B-A3B-Instruct:novita")
    parser.add_argument("--config", default=None,
                        help="visru.toml for this arm; defaults to the workspace's")
    parser.add_argument("--sample", type=int, default=None,
                        help="fixed-seed subset of labelled questions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--script", default=None,
                        help="replay this turn script instead of calling a model")


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


def prompt(q: Question) -> str:
    """The question as the model sees it, with lettered options for a choice."""
    if q.kind == "temporal":
        return (f"{q.question}\nAnswer with the time window in seconds as start..end, "
                "e.g. 12.4..15.0.")
    if q.kind != "choice":
        return q.question
    lines = [q.question, *(f"({LETTERS[i]}) {o}" for i, o in enumerate(q.options)),
             "Answer with the option's letter."]
    return "\n".join(lines)


def _choice(question: Question, answer: str) -> bool | None:
    letters = LETTERS[:len(question.options)]
    head = re.match(r"^\W*([A-Z])\W*(?:\s|$)", str(answer).strip())
    if head and head.group(1) in letters:  # capital only: "a cat" is not option A
        return head.group(1) == str(question.answer).strip().upper()
    given = normalise(answer)
    named = [letters[i] for i, o in enumerate(question.options) if normalise(o) == given]
    if named:
        return named[0] == str(question.answer).strip().upper()
    # A sentence naming one option is probably an answer, but a person decides.
    mentioned = [o for o in question.options if normalise(o) and normalise(o) in given]
    return None if mentioned else False


def window(text: str) -> tuple[float, float] | None:
    """(start, end) from "12.4..15.0" or "from 12.4 s to 15 s"; None unless exactly two
    numbers. Dots are not a decimal point when doubled, so "12..15" is two numbers."""
    numbers = NUMBER.findall(str(text).replace("..", " "))
    if len(numbers) != 2:
        return None
    a, b = sorted(float(n) for n in numbers)
    return a, b


def _temporal(question: Question, answer: str) -> bool | None:
    """Both ends within the question's tolerance. Anything but two numbers — "about
    12 s", "12-15 s, or maybe 20" — goes to a person rather than being guessed."""
    truth, given = window(question.answer), window(answer)
    if truth is None:
        raise ValueError(f"{question.id}: temporal answer {question.answer!r} is not "
                         "start..end")
    if given is None:
        return None
    tol = TOLERANCE if question.tolerance is None else question.tolerance
    return abs(given[0] - truth[0]) <= tol and abs(given[1] - truth[1]) <= tol


def grade(question: Question, answer: str) -> bool | None:
    """True/False when it can be decided mechanically, None when a human must look."""
    if not question.labelled or answer is None:
        return None
    if question.kind == "choice":
        return _choice(question, answer)
    if question.kind == "temporal":
        return _temporal(question, answer)
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
