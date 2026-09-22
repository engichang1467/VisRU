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


# --- Phase 2: multiple choice, sampling, calibration, the agent arms --------
def _choice(**kw):
    return _q(kind="choice", answer="B", options=["rubber", "cotton", "kevlar"], **kw)


def test_choice_reads_a_letter_in_any_usual_shape():
    q = _choice()
    for given in ("B", "(B)", "B.", "B) cotton", "**B**"):
        assert suite.grade(q, given) is True, given
    assert suite.grade(q, "A") is False
    assert suite.grade(q, "cotton") is True  # the option text itself
    assert suite.grade(q, "a cotton glove") is None  # a sentence: a person decides
    assert suite.grade(q, "cannot tell") is False


def test_choice_prompt_letters_the_options():
    assert suite.prompt(_choice()).splitlines()[1:4] == [
        "(A) rubber", "(B) cotton", "(C) kevlar"]


def test_options_only_belong_to_choice_questions(tmp_path):
    bad = tmp_path / "q.jsonl"
    bad.write_text(json.dumps({"id": "q", "asset": "a", "question": "?",
                               "options": ["x", "y"]}) + "\n")
    with pytest.raises(ValueError, match="options go with kind 'choice'"):
        suite.load(bad)


def test_every_arm_draws_the_same_sample():
    qs = [_q(id=f"q{i:02d}") for i in range(40)]
    first = suite.select(qs, 10, seed=3)
    assert first == suite.select(qs, 10, seed=3) and len(first) == 10
    assert [q.id for q in first] == sorted(q.id for q in first)  # file order kept


def test_calibration_groups_accuracy_by_status():
    run = {"a": {"status": "verified"}, "b": {"status": "verified"},
           "c": {"status": "supported"}, "d": {}}
    verdicts = {"a": "correct", "b": "wrong", "c": "correct", "d": "wrong"}
    assert score.calibration(run, verdicts) == {
        "verified": (1, 2), "supported": (1, 1), "no report": (0, 1)}
    lo, hi = score.wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.97


def test_the_agent_arm_runs_end_to_end_on_replay(marker_workspace, tmp_path):
    from evals import agent
    from visru.workspace.registry import Registry

    reg = Registry(marker_workspace)
    reg.scan()
    reg.save()
    image = [a for a in reg.list() if a.kind == "image"][0]
    questions = tmp_path / "q.jsonl"
    questions.write_text("".join(json.dumps(
        {"id": f"q{i}", "asset": image.name, "question": "What colour is the square?",
         "answer": "B", "kind": "choice", "options": ["blue", "red"],
         "category": "colour"}) + "\n" for i in range(2)))
    script = tmp_path / "turns.json"
    script.write_text(json.dumps([
        {"calls": [{"name": "view", "arguments": {"asset_id": image.id,
                                                  "box": [2900, 1900, 3200, 2200]}}]},
        {"calls": [{"name": "detect", "arguments": {"asset_id": image.id,
                                                    "queries": ["x"]}}]},
        {"calls": [{"name": "submit_report", "arguments": {"claims": [
            {"claim_id": "answer", "text": "B", "status": "supported",
             "evidence": ["$VIEW"]}]}}]}]))

    # zoom_only may not call detect: the refusal comes back as a message, and the
    # session still ends in a report. Evidence IDs differ per session, so patch.
    def run_arm(arm):
        rows = []
        for q in suite.load(questions):
            before = len(list((marker_workspace / ".visru/artifacts").glob("a_*.json")))
            text = script.read_text().replace("$VIEW", f"a_{before + 1:04d}")
            one = tmp_path / f"{arm}_{q.id}.json"
            one.write_text(text)
            rows.append(agent.run_one(str(marker_workspace), q, arm, f"s_t_{arm}",
                                      None, 5, str(one)))
        return rows

    rows = run_arm("zoom_only")
    assert [r["answer"] for r in rows] == ["B", "B"]
    assert all(r["status"] == "supported" and r["reason"] == "reported" for r in rows)
    assert all(r["tool_calls"] == 3 and r["visual_tokens"] > 0 for r in rows)
    log = (marker_workspace / ".visru/trajectory" / f"{rows[0]['session']}.jsonl")
    assert "no tool named 'detect'" in log.read_text()

    res = score.score_run(suite.load(questions), {r["id"]: r for r in rows})
    assert res["accuracy"] == 1.0


def test_importers_parse_the_published_formats():
    from evals.import_bench import parse_hrbench, parse_vstar

    v = parse_vstar({"image": "direct_attributes/sa_4690.jpg", "question_id": "0",
                     "category": "direct_attributes", "label": "A",
                     "text": "What is the material of the glove?\n(A) rubber\n(B) cotton\n"
                             "Answer with the option's letter from the given choices "
                             "directly."})
    assert v["question"] == "What is the material of the glove?"
    assert v["options"] == ["rubber", "cotton"] and v["answer"] == "A"
    h = parse_hrbench({"index": "7", "answer": "C", "question": " How many? ",
                       "A": "1", "B": "2", "C": "3", "D": "4", "category": "single"},
                      "hr4k")
    assert h["id"] == "hr4k_7" and h["asset"] == "hr4k_7.jpg"
    assert h["options"] == ["1", "2", "3", "4"] and h["question"] == "How many?"
    for row in (v, h):
        suite.Question(**row)  # the suite accepts what the importer writes
