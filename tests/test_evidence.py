"""The ledger and the tools that feed it.

The ledger's only job is to refuse. A status the model can assert at will measures
nothing, so most of these tests are about what does NOT get recorded.
"""

import json

import numpy as np
import pytest
from PIL import Image

from visru.evidence.ledger import Claim, Ledger, RejectedReport
from visru.geometry import ViewTransform
from visru.tools import open_workspace
from visru.tools.navigate import view
from visru.tools.verify import cross_check, overlay, submit_report
from visru.workspace.artifacts import Mark
from visru.workspace.registry import Registry

from conftest import MARKER_BOX


@pytest.fixture
def ctx(marker_workspace):
    reg = Registry(marker_workspace)
    reg.scan()
    reg.save()
    return open_workspace(marker_workspace, session="s_test_01")


@pytest.fixture
def evidence(ctx):
    """Two artifacts from unrelated tools, plus one derived from the other."""
    im = ctx.registry.list(kind="image")[0].id
    looked = view(ctx, im, box=[2900, 1900, 3200, 2200]).artifacts[0]
    tf = ViewTransform(im, 2900.0, 1900.0, 2.0)
    found = ctx.artifacts.save(Image.new("RGB", (600, 600)), kind="detect", parent=im,
                               transform=tf,
                               marks=[Mark(id=1, label="red square", box=MARKER_BOX,
                                           score=0.9)])
    derived = ctx.artifacts.save(Image.new("RGB", (600, 600)), kind="segment", parent=im,
                                 transform=tf, params={"from_artifact": found.id},
                                 marks=[Mark(id=1, label="from mark 1", box=MARKER_BOX)])
    return {"view": looked, "detect": found.id, "segment_of_detect": derived.id}


@pytest.fixture
def agreed(ctx, evidence):
    """A harness-computed check: my box for the square against the detector's."""
    return cross_check(ctx, "localize", [evidence["view"], evidence["detect"]],
                       box=list(MARKER_BOX)).artifacts[0]


def _claim(**kw):
    base = {"claim_id": "c_1", "text": "The label is torn.", "status": "supported",
            "evidence": [], "checks": []}
    return Claim(**{**base, **kw})


def _ledger(ctx):
    return Ledger(ctx.root, ctx.session, ctx.artifacts)


# --- what the evidence warrants -----------------------------------------
def test_no_evidence_is_a_hypothesis(ctx):
    led = _ledger(ctx)
    assert led.warranted_status(_claim(status="hypothesis")) == "hypothesis"
    problems = led.problems([_claim(status="supported")])
    assert any("is a hypothesis" in p for p in problems)


def test_evidence_alone_is_supported_not_verified(ctx, evidence):
    led = _ledger(ctx)
    claim = _claim(status="verified", evidence=[evidence["view"]])
    assert led.warranted_status(claim) == "supported"
    assert any("verified needs a passing independent check" in p
               for p in led.problems([claim]))


def test_two_models_that_agree_earn_verified(ctx, evidence, agreed):
    led = _ledger(ctx)
    claim = _claim(status="verified", evidence=[evidence["view"], evidence["detect"]],
                   checks=[{"check": agreed}])
    assert led.warranted_status(claim) == "verified"
    assert led.problems([claim]) == []


def test_a_check_the_model_wrote_itself_never_earns_verified(ctx, evidence):
    """Otherwise `verified` measures what the model says about itself."""
    led = _ledger(ctx)
    claim = _claim(status="verified", evidence=[evidence["view"], evidence["detect"]],
                   checks=[{"kind": "cross_tool", "methods": ["detect", "ocr"],
                            "artifacts": [evidence["detect"]], "result": "agree"}])
    assert led.warranted_status(claim) == "supported"
    assert any("run cross_check" in p for p in led.problems([claim]))


def test_retyping_a_disagreement_as_agree_does_not_help(ctx, evidence):
    led = _ledger(ctx)
    ref = cross_check(ctx, "localize", [evidence["view"], evidence["detect"]],
                      box=[0, 0, 50, 50]).artifacts[0]
    claim = _claim(status="verified", evidence=[evidence["view"]],
                   checks=[{"check": ref, "result": "agree"}])
    assert led.warranted_status(claim) == "contested"


def test_citing_something_that_is_not_a_check_is_refused(ctx, evidence):
    led = _ledger(ctx)
    claim = _claim(status="supported", evidence=[evidence["view"]],
                   checks=[{"check": evidence["view"]}])
    assert any("not a cross_check artifact" in p for p in led.problems([claim]))


def test_one_model_checking_itself_is_not_independent(ctx, evidence):
    led = _ledger(ctx)
    claim = _claim(status="verified", evidence=[evidence["detect"]],
                   checks=[{"kind": "cross_tool", "methods": ["detect", "detect"],
                            "result": "agree"}])
    problems = led.problems([claim])
    assert any("share a model" in p or "not independent" in p for p in problems)


def test_a_mask_prompted_from_a_detection_does_not_check_it(ctx, evidence):
    """The README's own example: SAM masks prompted from detector boxes inherit the
    detector's errors, so agreement between them is not evidence of anything."""
    led = _ledger(ctx)
    claim = _claim(status="verified",
                   evidence=[evidence["detect"], evidence["segment_of_detect"]],
                   checks=[{"kind": "cross_tool", "methods": ["detect", "segment"],
                            "artifacts": [evidence["detect"],
                                          evidence["segment_of_detect"]],
                            "result": "agree"}])
    problems = led.problems([claim])
    assert any("derived from" in p for p in problems), problems
    assert led.warranted_status(claim) == "supported"


def test_a_disagreeing_check_forces_contested(ctx, evidence):
    led = _ledger(ctx)
    claim = _claim(status="verified", evidence=[evidence["view"]],
                   checks=[{"kind": "cross_tool", "methods": ["detect", "ocr"],
                            "result": "disagree"}])
    assert led.warranted_status(claim) == "contested"
    assert any("contested" in p for p in led.problems([claim]))
    assert _ledger(ctx).problems([_claim(**{**claim.__dict__, "status": "contested"})]) == []


def test_a_check_with_a_meaningless_result_does_not_count(ctx, evidence):
    led = _ledger(ctx)
    claim = _claim(status="verified", evidence=[evidence["view"]],
                   checks=[{"kind": "cross_tool", "methods": ["detect", "ocr"],
                            "result": "looks about right"}])
    assert any("is not one of" in p for p in led.problems([claim]))


def test_conservative_statuses_are_allowed(ctx, evidence, agreed):
    """Claiming less than the evidence supports is not an error."""
    led = _ledger(ctx)
    claim = _claim(status="supported", evidence=[evidence["view"]],
                   checks=[{"check": agreed}])
    assert led.warranted_status(claim) == "verified"
    assert led.problems([claim]) == []


# --- submit_report -------------------------------------------------------
def test_submit_report_rejects_everything_at_once(ctx, evidence):
    with pytest.raises(ValueError) as caught:
        submit_report(ctx, [
            {"claim_id": "c_1", "text": "no evidence", "status": "supported"},
            {"claim_id": "c_2", "text": "bad status", "status": "probably"},
            {"claim_id": "c_1", "text": "duplicate id", "status": "hypothesis"},
            {"claim_id": "c_4", "text": "dangling", "status": "supported",
             "evidence": ["a_9999"]},
        ])
    message = str(caught.value)
    # One problem per turn would spend the session on bookkeeping.
    for fragment in ("c_1", "c_2", "duplicate", "a_9999", "not recorded"):
        assert fragment in message, message
    assert not (ctx.root / ".visru/ledger").exists()


def test_submit_report_writes_the_ledger_and_groups_by_status(ctx, evidence, agreed):
    res = submit_report(ctx, [
        {"claim_id": "c_1", "text": "Gripper closed early.", "status": "verified",
         "evidence": [evidence["view"], evidence["detect"]],
         "checks": [agreed]},
        {"claim_id": "c_2", "text": "Finger occluded the handle.", "status": "supported",
         "evidence": [evidence["view"]]},
        {"claim_id": "c_3", "text": "Probably a calibration drift.",
         "status": "hypothesis"},
    ])

    assert "VERIFIED (1)" in res.text and "SUPPORTED (1)" in res.text
    assert "HYPOTHESES — no evidence (1)" in res.text
    assert res.text.index("VERIFIED") < res.text.index("HYPOTHESES")

    saved = json.loads((ctx.root / ".visru/ledger/s_test_01.json").read_text())
    assert [c["claim_id"] for c in saved["claims"]] == ["c_1", "c_2", "c_3"]
    # The cited check is written out in full, read from its artifact.
    check = saved["claims"][0]["checks"][0]
    assert check["check"] == agreed and check["result"] == "agree"
    assert check["iou"] == 1.0 and check["kind"] == "cross_check:localize"
    assert "assets" in saved["coverage"]

    from visru.evidence.report import render

    html = render(ctx.root, ctx.session).read_text()
    assert html.index("Gripper closed early.") < html.index("Probably a calibration")
    assert f"../artifacts/{agreed}.png" in html and "IoU 1.0" in html
    assert "Hypotheses: no evidence (1)" in html
    assert all(c["session"] == "s_test_01" for c in saved["claims"])


def test_submit_report_rejects_junk_input(ctx):
    with pytest.raises(ValueError, match="must be a list"):
        submit_report(ctx, {"claim_id": "c_1"})
    with pytest.raises(ValueError, match="unknown claim fields"):
        submit_report(ctx, [{"claim_id": "c_1", "conclusion": "x"}])
    with pytest.raises(ValueError, match="at least one claim"):
        submit_report(ctx, [])


def test_a_hypothesis_may_not_smuggle_in_evidence(ctx, evidence):
    with pytest.raises(ValueError, match="hypothesis carries no evidence"):
        submit_report(ctx, [{"claim_id": "c_1", "text": "guess", "status": "hypothesis",
                             "evidence": [evidence["view"]]}])


def test_ledger_reloads_what_it_wrote(ctx, evidence):
    submit_report(ctx, [{"claim_id": "c_1", "text": "A torn label.",
                         "status": "supported", "evidence": [evidence["view"]]}])
    back = Ledger.load(ctx.root, ctx.session, ctx.artifacts)
    assert [c.claim_id for c in back.claims] == ["c_1"]
    assert back.claims[0].evidence == [evidence["view"]]


def test_rejected_report_is_raised_not_returned(ctx):
    led = _ledger(ctx)
    with pytest.raises(RejectedReport):
        led.submit([_claim(status="supported")])


# --- overlay -------------------------------------------------------------
def _tint_bbox_source(img, tf):
    """Bounding box of mark-1 tint, in source pixels."""
    a = np.asarray(img.convert("RGB")).astype(int)
    hit = (a[..., 0] > 200) & (a[..., 1] < 90) & (a[..., 2] < 90)
    assert hit.any(), "nothing was drawn"
    ys, xs = np.nonzero(hit)
    x0, y0 = tf.to_source(float(xs.min()), float(ys.min()))
    x1, y1 = tf.to_source(float(xs.max() + 1), float(ys.max() + 1))
    return x0, y0, x1, y1


def test_overlay_draws_a_claimed_box_where_it_belongs(ctx):
    im = ctx.registry.list(kind="image")[0].id
    res = overlay(ctx, im, boxes=[list(MARKER_BOX)], labels=["the marker"],
                  box=[2900, 1900, 3200, 2200])
    art = ctx.artifacts.load(res.artifacts[0])

    assert art.kind == "overlay"
    assert art.mark(1).label == "the marker"
    assert art.mark(1).box == pytest.approx(MARKER_BOX, abs=1.0)
    assert "does not sit on the thing it names" in res.text


MASK_BOX = (2900, 1900, 2940, 1940)  # plain background, far from the red marker


def test_overlay_replays_a_mask_onto_a_different_zoom(ctx):
    """A mask made at one scale must land on the same source pixels at another, or
    re-inspection would be checking the wrong part of the image.

    The mask covers background, not the red marker, and the mark carries no box —
    draw_marks would otherwise outline the box straight from source coordinates,
    which is correct whatever the mask does, and the measurement would prove nothing.
    """
    im = ctx.registry.list(kind="image")[0].id
    wide_tf = ViewTransform(im, 2800.0, 1800.0, 1.0)
    mask = Image.new("L", (600, 600), 0)
    mask.paste(255, (int(MASK_BOX[0] - 2800), int(MASK_BOX[1] - 1800),
                     int(MASK_BOX[2] - 2800), int(MASK_BOX[3] - 1800)))
    wide = ctx.artifacts.save(Image.new("RGB", (600, 600)), kind="segment", parent=im,
                              transform=wide_tf,
                              marks=[Mark(id=1, label="a scuff")],  # mask, no box
                              mask_images={1: mask})

    # Replay into a tighter crop, rendered at a different scale (2.0x, not 1.0x).
    res = overlay(ctx, im, from_artifact=wide.id, box=[2860, 1860, 2980, 1980])
    tf = ctx.artifacts.load(res.artifacts[0]).view_transform
    assert tf.scale != wide_tf.scale

    # Extent, not just position: the crop renders at 2.0x while the mask was made at
    # 1.0x, so a mask pasted without rescaling would cover half the source area and
    # still centre in roughly the right place.
    painted = _tint_bbox_source(res.images[0], tf)
    assert painted == pytest.approx(MASK_BOX, abs=4.0)


def test_overlay_needs_something_to_draw(ctx):
    im = ctx.registry.list(kind="image")[0].id
    with pytest.raises(ValueError, match="overlay needs boxes"):
        overlay(ctx, im)
    with pytest.raises(ValueError, match=r"\[x0, y0, x1, y1\]"):
        overlay(ctx, im, boxes=[[1, 2, 3]])


# --- regressions from the first real session -----------------------------
def test_the_same_model_reading_two_images_is_not_a_cross_check(ctx, evidence):
    """From the first live session: four claims reached `verified` on checks like
    ["view im_A a_0003 @2.0x", "view im_B a_0033 @2.0x"] — one VLM reading two crops.
    Free-text method names each became their own family, so any two differently
    worded strings looked independent. A systematic misreading repeats across frames.
    """
    led = _ledger(ctx)
    claim = _claim(status="verified", evidence=[evidence["view"]],
                   checks=[{"kind": "cross-frame",
                            "methods": ["view im_9b9e3fc49d75 a_0003 @2.0x",
                                        "view im_0f91bbd97367 a_0033 @2.0x"],
                            "result": "agree"}])
    assert led.warranted_status(claim) == "supported"
    assert any("all use vlm" in p for p in led.problems([claim]))


def test_prose_methods_still_resolve_to_their_tools(ctx, evidence):
    """Agents write methods as prose, and view-vs-detector is two models however it
    is worded. (Written by the model, it still only reaches supported.)"""
    led = _ledger(ctx)
    claim = _claim(status="supported", evidence=[evidence["view"], evidence["detect"]],
                   checks=[{"kind": "visual-vs-detector",
                            "methods": ["view crop a_0008 (pair + 1 on lane markings)",
                                        "detect a_0010 (marks 1,2,3 in roadway)"],
                            "result": "agree"}])
    assert led.check_problems(claim.as_checks()[0]) == []
    assert led.problems([claim]) == []


def test_a_method_naming_no_tool_is_rejected(ctx, evidence):
    led = _ledger(ctx)
    claim = _claim(status="verified", evidence=[evidence["view"]],
                   checks=[{"kind": "vibes", "methods": ["looked again", "seemed right"],
                            "result": "agree"}])
    problems = led.problems([claim])
    assert any("cannot tell which model" in p for p in problems), problems


def test_a_method_naming_two_tools_is_rejected(ctx, evidence):
    led = _ledger(ctx)
    claim = _claim(status="verified", evidence=[evidence["detect"]],
                   checks=[{"kind": "mixed",
                            "methods": ["segment prompted from detect boxes", "ocr a_0028"],
                            "result": "agree"}])
    assert any("names more than one tool" in p for p in led.problems([claim]))


def test_lineage_is_caught_even_when_ids_are_only_in_the_prose(ctx, evidence):
    """Agents name artifacts inside method text rather than in `artifacts`, and a
    derived mask must not corroborate the detection it came from either way."""
    led = _ledger(ctx)
    claim = _claim(status="verified", evidence=[evidence["detect"]],
                   checks=[{"kind": "cross_tool",
                            "methods": [f"detect {evidence['detect']}",
                                        f"segment {evidence['segment_of_detect']}"],
                            "result": "agree"}])
    assert any("derived from" in p for p in led.problems([claim]))


# --- cross_check ---------------------------------------------------------
def test_cross_check_counts_marks_against_my_reading(ctx, evidence):
    pair = [evidence["view"], evidence["detect"]]
    assert "AGREE" in cross_check(ctx, "count", pair, value=1).text
    assert "DISAGREE" in cross_check(ctx, "count", pair, value="2").text


def test_cross_check_reads_text_across_ocr_lines(ctx, evidence):
    im = ctx.registry.list(kind="image")[0].id
    tf = ViewTransform(im, 2900.0, 1900.0, 2.0)
    read = ctx.artifacts.save(Image.new("RGB", (600, 600)), kind="ocr", parent=im,
                              transform=tf, params={"box": [2900, 1900, 3200, 2200]},
                              marks=[Mark(id=1, label="TRADER", box=(2900, 1900, 3000, 1950)),
                                     Mark(id=2, label="JOE'S", box=(2900, 1960, 3000, 2000)),
                                     Mark(id=3, label="OPEN", box=(2900, 2100, 3000, 2150))])
    pair = [evidence["view"], read.id]
    assert "AGREE" in cross_check(ctx, "text", pair, value="Trader Joe's",
                                  marks=[None, [1, 2]]).text
    assert "DISAGREE" in cross_check(ctx, "text", pair, value="Trader Joe's").text


def test_cross_check_refuses_pairs_that_share_errors(ctx, evidence):
    with pytest.raises(ValueError, match="derived from"):
        cross_check(ctx, "count", [evidence["detect"], evidence["segment_of_detect"]])
    im = ctx.registry.list(kind="image")[0].id
    again = view(ctx, im, box=[0, 0, 400, 400]).artifacts[0]
    with pytest.raises(ValueError, match="not independent"):
        cross_check(ctx, "count", [evidence["view"], again], value=1)
    assert not [a for a in ctx.artifacts.all() if a.kind == "check"]


def test_cross_check_needs_my_reading_for_a_view(ctx, evidence):
    with pytest.raises(ValueError, match="pass value"):
        cross_check(ctx, "count", [evidence["view"], evidence["detect"]])


def test_a_check_artifact_is_regenerable(ctx, agreed):
    from visru.agent.trajectory import replay_artifact

    remade = replay_artifact(ctx, agreed)
    assert remade.tobytes() == ctx.artifacts.load_image(agreed).tobytes()


# --- coverage ------------------------------------------------------------
def test_coverage_counts_only_this_sessions_close_views(ctx, marker_workspace):
    from visru.tools.verify import coverage_summary

    im = ctx.registry.list(kind="image")[0]
    view(ctx, im.id)  # whole 4032 px image: seen, but far below scale
    view(ctx, im.id, box=[0, 0, 1008, 1512])  # an eighth, above native scale
    other = open_workspace(marker_workspace, session="s_other")
    view(other, im.id, box=[1008, 0, 4032, 3024])

    cov = coverage_summary(ctx)["assets"][im.id]
    assert cov["seen"] == 1.0
    assert 0.12 <= cov["inspected"] <= 0.14
    vid = ctx.registry.list(kind="video")[0].id
    assert vid in coverage_summary(ctx)["untouched"]


def test_coverage_counts_video_frames(ctx):
    from visru.tools.navigate import sample_frames
    from visru.tools.verify import coverage, coverage_summary

    vid = ctx.registry.list(kind="video")[0].id
    sample_frames(ctx, vid, n=6)
    view(ctx, f"{vid}#12")
    c = coverage_summary(ctx)["assets"][vid]
    assert c["frames_seen"] == 7 and c["frames_inspected"] == 1 and c["frames"] == 90
    assert "1 of 90 frames" in coverage(ctx).text
