"""The eval harness. Its one piece of real logic is the grader, and a grader that
marks wrong answers correct would quietly invalidate the Phase 0 exit criterion."""

import json

import pytest

from evals import score, suite


def _q(**kw):
    base = {"id": "q01", "asset": "a.jpg", "question": "what?", "answer": "TRADER JOE'S"}
    return suite.Question(**{**base, **kw})


def test_grading_ignores_case_punctuation_and_spacing():
    q = _q(accept=["traders joes"])
    assert suite.grade(q, "trader joe's") is True
    assert suite.grade(q, "  TRADER   JOES ") is True
    assert suite.grade(q, "Traders Joes") is True


def test_a_clearly_different_answer_is_wrong():
    assert suite.grade(_q(), "Whole Foods") is False
    assert suite.grade(_q(), "cannot tell") is False


def test_an_answer_that_merely_contains_the_truth_needs_a_human():
    """"not MANCHESTER" contains "MANCHESTER"; the grader must not award that."""
    q = _q(answer="Manchester")
    assert suite.grade(q, "The sign is not Manchester") is None
    assert suite.grade(q, "Manchester Ave, I think") is None


def test_counts_compare_numerically_but_only_when_unambiguous():
    q = _q(kind="count", answer="3")
    assert suite.grade(q, "3") is True
    assert suite.grade(q, "there are 3 people") is True
    assert suite.grade(q, "5") is False
    # Two different numbers in one answer is not a count the grader can read.
    assert suite.grade(q, "3 on the road, 5 on the sidewalk") is None


def test_unlabelled_questions_are_never_graded():
    assert suite.grade(_q(answer=None), "anything") is None
    assert _q(answer=None).labelled is False


def test_loader_rejects_malformed_sets(tmp_path):
    path = tmp_path / "q.jsonl"

    path.write_text('{"id": "q1", "asset": "a.jpg", "question": "?", "kind": "vibes"}\n')
    with pytest.raises(ValueError, match="kind"):
        suite.load(path)

    path.write_text('{"id": "q1", "asset": "a.jpg", "question": "?"}\n'
                    '{"id": "q1", "asset": "b.jpg", "question": "?"}\n')
    with pytest.raises(ValueError, match="duplicate question id"):
        suite.load(path)

    path.write_text('{"id": "q1", "asset": "a.jpg", "question": "?", "answr": "x"}\n')
    with pytest.raises(ValueError, match="unknown fields"):
        suite.load(path)


def test_comments_and_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('# a note\n\n{"id": "q1", "asset": "a.jpg", "question": "?"}\n')
    assert [q.id for q in suite.load(path)] == ["q1"]


def test_scoring_separates_decided_from_review_and_unlabelled(tmp_path):
    questions = [_q(id="q1"), _q(id="q2"), _q(id="q3", answer=None), _q(id="q4")]
    run = {"q1": {"id": "q1", "answer": "Trader Joe's"},
           "q2": {"id": "q2", "answer": "Whole Foods"},
           "q3": {"id": "q3", "answer": "something"},
           # q4 missing entirely
           }
    res = score.score_run(questions, run)
    assert res["tally"]["correct"] == 1 and res["tally"]["wrong"] == 1
    assert res["tally"]["unlabelled"] == 1 and res["tally"]["missing"] == 1
    assert res["accuracy"] == 0.5  # only the two decided questions count


def test_a_human_verdict_overrides_the_grader(tmp_path):
    questions = [_q(id="q1")]
    run = {"q1": {"id": "q1", "answer": "the neon reads trader joes", "correct": True}}
    assert score.score_run(questions, run)["tally"]["correct"] == 1


def test_the_shipped_starter_set_parses_and_is_honest_about_labels():
    questions = suite.load("evals/questions.example.jsonl")
    assert len(questions) >= 5
    # Every shipped entry carries a human-verified label; an unlabelled one is
    # never graded, so it would silently shrink the exit criterion's denominator.
    assert all(q.labelled for q in questions)
    assert all(q.kind in suite.KINDS for q in questions)
