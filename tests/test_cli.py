"""`visru run` — the seam between a shell and the loop.

What matters here is not the agent but the wiring: that bad invocations are
refused with something a person can act on, and that the exit code distinguishes
an answer from a session that merely finished.
"""

import json

import pytest
from typer.testing import CliRunner

from visru.cli import app
from visru.tools import open_workspace
from visru.workspace.registry import Registry

runner = CliRunner()


@pytest.fixture
def indexed(marker_workspace):
    reg = Registry(marker_workspace)
    reg.scan()
    reg.save()
    return marker_workspace


def script_at(path, turns):
    path.write_text(json.dumps(turns))
    return str(path)


def viewing(asset_id, n=3):
    return [{"text": f"Step {i}.",
             "calls": [{"name": "view", "arguments": {"asset_id": asset_id}}]}
            for i in range(n)]


BOX = str.maketrans({c: " " for c in "│╭╮╯╰─┃┏┓┗┛━"})


def flat(result) -> str:
    """Typer wraps errors in a box, so a message breaks across lines *and* has
    border characters pushed through the middle of it."""
    return " ".join(result.output.translate(BOX).split())


def invoke(workspace, *args):
    return runner.invoke(app, ["run", str(workspace), "--task", "what is here?",
                               "--backend", "replay", *args])


# --- refusals a person can act on ---------------------------------------
def test_an_unindexed_workspace_says_to_index_it(marker_workspace, tmp_path):
    result = invoke(marker_workspace, "--script", script_at(tmp_path / "s.json", []))
    assert result.exit_code != 0
    assert "visru index" in flat(result)


def test_a_missing_directory_is_refused(tmp_path):
    result = invoke(tmp_path / "nope", "--script",
                    script_at(tmp_path / "s.json", []))
    assert result.exit_code != 0
    assert "not a directory" in flat(result)


def test_a_script_without_the_replay_backend_is_refused(indexed, tmp_path):
    result = runner.invoke(app, ["run", str(indexed), "--task", "x",
                                 "--script", script_at(tmp_path / "s.json", [])])
    assert result.exit_code != 0
    assert "--backend replay" in flat(result)


def test_replay_without_a_script_is_refused(indexed):
    result = runner.invoke(app, ["run", str(indexed), "--task", "x",
                                 "--backend", "replay"])
    assert result.exit_code != 0
    assert "script" in flat(result)


def test_an_unknown_backend_names_the_ones_that_exist(indexed):
    result = runner.invoke(app, ["run", str(indexed), "--task", "x",
                                 "--backend", "gpt"])
    assert result.exit_code != 0
    assert "anthropic" in flat(result)


# --- endings and exit codes ---------------------------------------------
def test_a_session_without_a_report_exits_non_zero(indexed, tmp_path):
    asset = open_workspace(indexed).registry.list(kind="image")[0].id
    result = invoke(indexed, "--script",
                    script_at(tmp_path / "s.json", viewing(asset)),
                    "--max-steps", "6")

    assert result.exit_code == 1, "a session that never answered looked like success"
    assert "no report" in flat(result)


def test_a_submitted_report_exits_zero_and_names_its_ledger(indexed, tmp_path):
    ctx = open_workspace(indexed)
    asset = ctx.registry.list(kind="image")[0].id
    turns = viewing(asset, 2) + [
        {"text": "Reporting.",
         "calls": [{"name": "submit_report", "arguments": {"claims": [
             {"claim_id": "c_1", "text": "There is a red marker.",
              "status": "supported", "evidence": ["a_0001"]}]}}]}]

    result = invoke(indexed, "--script", script_at(tmp_path / "s.json", turns))

    assert result.exit_code == 0, result.output
    assert "SUPPORTED" in flat(result)
    assert "ledger:" in flat(result)
    written = list((indexed / ".visru" / "ledger").glob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["claims"][0]["claim_id"] == "c_1"

    # `visru report` with no session renders the latest ledger.
    rendered = runner.invoke(app, ["report", str(indexed)])
    assert rendered.exit_code == 0, rendered.output
    html = (indexed / ".visru" / "reports" / f"{written[0].stem}.html").read_text()
    assert "There is a red marker." in html and "../artifacts/a_0001.png" in html


def test_report_refuses_an_unknown_session(indexed):
    result = runner.invoke(app, ["report", str(indexed), "s_nope"])
    assert result.exit_code != 0 and "no ledger" in flat(result)


def test_max_steps_is_honoured(indexed, tmp_path):
    asset = open_workspace(indexed).registry.list(kind="image")[0].id
    result = invoke(indexed, "--script",
                    script_at(tmp_path / "s.json", viewing(asset, 50)),
                    "--max-steps", "4")
    assert "after 4 steps" in flat(result)


def test_the_trajectory_path_is_printed_even_when_nothing_was_reported(indexed,
                                                                      tmp_path):
    asset = open_workspace(indexed).registry.list(kind="image")[0].id
    result = invoke(indexed, "--script",
                    script_at(tmp_path / "s.json", viewing(asset, 1)),
                    "--max-steps", "3")
    assert "trajectory:" in flat(result)
    assert list((indexed / ".visru" / "trajectory").glob("*.jsonl"))


def test_init_drafts_visru_md_and_never_overwrites_it(marker_workspace):
    first = runner.invoke(app, ["init", str(marker_workspace), "--no-model"])
    assert first.exit_code == 0, first.output
    text = (marker_workspace / "VISRU.md").read_text()
    assert "`marker.png`, 4032x3024" in text and "## Domain notes" in text

    (marker_workspace / "VISRU.md").write_text("hand-edited")
    second = runner.invoke(app, ["init", str(marker_workspace), "--no-model"])
    assert second.exit_code == 0, second.output
    assert (marker_workspace / "VISRU.md").read_text() == "hand-edited"
    assert "## Assets" in (marker_workspace / "VISRU.md.new").read_text()


def test_init_refuses_a_missing_directory(tmp_path):
    result = runner.invoke(app, ["init", str(tmp_path / "nope"), "--no-model"])
    assert result.exit_code != 0


def test_init_survey_claims_land_in_visru_md(marker_workspace, tmp_path):
    reg = Registry(marker_workspace)
    reg.scan()
    asset = reg.list(kind="image")[0].id
    turns = viewing(asset, 1) + [
        {"text": "Reporting.",
         "calls": [{"name": "submit_report", "arguments": {"claims": [
             {"claim_id": "c_1", "text": "A red square on a dark background.",
              "status": "supported", "evidence": ["a_0001"]},
             {"claim_id": "c_2", "text": "Probably a synthetic test image.",
              "status": "hypothesis"}]}}]}]

    result = runner.invoke(app, ["init", str(marker_workspace), "--backend", "replay",
                                 "--script", script_at(tmp_path / "s.json", turns)])

    assert result.exit_code == 0, result.output
    text = (marker_workspace / "VISRU.md").read_text()
    assert "- A red square on a dark background. (a_0001)" in text
    unconfirmed = text[text.index("## Unconfirmed"):]
    assert "Probably a synthetic test image." in unconfirmed
    assert text.index("## Assets") < text.index("## Observed") < text.index("## Domain notes")


def test_init_writes_facts_and_exits_non_zero_when_the_survey_never_reports(
        marker_workspace, tmp_path):
    result = runner.invoke(app, ["init", str(marker_workspace), "--backend", "replay",
                                 "--script", script_at(tmp_path / "s.json", [])])

    assert result.exit_code == 1
    text = (marker_workspace / "VISRU.md").read_text()
    assert "## Assets" in text and "## Observed" not in text
