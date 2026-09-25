"""Claims, the evidence behind them, and the checks run against them.

The ledger is the only place a status is decided. The model proposes one; the ledger
computes what the evidence actually warrants and refuses anything stronger. That
refusal is the whole mechanism — a status the model can assert at will measures
nothing, and Phase 2 exists to test whether `verified` predicts correctness.

For the same reason a check the model writes itself — `{"result": "agree"}` — can
make a claim `contested` but never `verified`. Only a check the harness computed
counts: `cross_check` compares two methods' outputs, saves the verdict as a
`check` artifact, and the claim cites that artifact's ID. The ledger reads the
verdict from the sidecar, not from whatever the model copied into its report.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from visru.workspace.artifacts import ArtifactStore

LEDGER_DIR = ".visru/ledger"

VERIFIED, SUPPORTED, CONTESTED, HYPOTHESIS = "verified", "supported", "contested", "hypothesis"
STATUSES = (VERIFIED, SUPPORTED, CONTESTED, HYPOTHESIS)
RANK = {HYPOTHESIS: 0, SUPPORTED: 1, VERIFIED: 2}

PASSING = {"agree", "pass", "consistent"}
FAILING = {"disagree", "fail", "inconsistent"}

# Two methods are independent only if they do not share a model. A VLM count checked
# against a detector count qualifies; SAM masks prompted from detector boxes do not,
# which is why lineage is checked as well as family.
MODEL_FAMILY = {
    "view": "vlm", "vlm": "vlm", "read": "vlm", "count": "vlm", "inspect": "vlm",
    "contact_sheet": "vlm",  # the model reading thumbnails is still the model
    "detect": "owlv2", "segment": "sam2", "track": "sam2", "ocr": "paddleocr",
    "search": "siglip",
    "depth": "depth_anything", "pointmap": "vggt",
    "python": "code", "measure": "code", "geometry": "code",
}
UNCLASSIFIED = "unclassified"  # one bucket, so two vague methods are never two models
AMBIGUOUS = "ambiguous"
ARTIFACT_REF = re.compile(r"\ba_\d{4,}\b")


def method_family(method: str) -> str:
    """Which model a method ran on, found by naming a tool anywhere in the string.

    Agents describe methods in prose ("view crop a_0008 (pair on lane markings)"), so
    this looks for a tool name rather than expecting a bare token. Everything it
    cannot place lands in ONE bucket: giving each unrecognised string its own family
    would make any two differently-worded methods look independent, which is how a
    real session got four `verified` claims out of the same VLM reading two crops.
    """
    hits = {family for name, family in MODEL_FAMILY.items()
            if re.search(rf"(?<![a-z]){name}(?![a-z])", method.lower())}
    if len(hits) == 1:
        return hits.pop()
    return AMBIGUOUS if hits else UNCLASSIFIED


@dataclass(frozen=True)
class Check:
    kind: str
    methods: list[str]
    result: str
    artifacts: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    harness: bool = False  # computed by cross_check, not asserted by the model

    @property
    def families(self) -> set[str]:
        return {method_family(m) for m in self.methods}

    @property
    def cited_artifacts(self) -> list[str]:
        """Artifact IDs named anywhere in the check, including inside method text.

        Agents write methods like "view crop a_0008 (pair on lane markings)", so the
        IDs that matter for lineage are often in prose rather than in `artifacts`.
        """
        found = list(self.artifacts)
        for text in [*self.methods, self.kind]:
            found.extend(ARTIFACT_REF.findall(text))
        return list(dict.fromkeys(found))

    def passed(self) -> bool:
        return self.result.lower() in PASSING

    def failed(self) -> bool:
        return self.result.lower() in FAILING


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    status: str
    evidence: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    session: str = ""

    def as_checks(self) -> list[Check]:
        return [Check(kind=c.get("kind", "unspecified"),
                      methods=list(c.get("methods", [])),
                      result=str(c.get("result", "")),
                      artifacts=list(c.get("artifacts", [])),
                      detail={k: v for k, v in c.items()
                              if k not in {"kind", "methods", "result", "artifacts"}})
                for c in self.checks]


class RejectedReport(ValueError):
    """submit_report refused the claims. Every problem is listed, not just the first."""


class Ledger:
    def __init__(self, root: str | Path, session: str, artifacts: ArtifactStore):
        self.root = Path(root)
        self.session = session
        self.artifacts = artifacts
        self.claims: list[Claim] = []
        self.coverage: dict[str, Any] = {}

    @property
    def path(self) -> Path:
        return self.root / LEDGER_DIR / f"{self.session}.json"

    # --- independence ----------------------------------------------------
    def _ancestors(self, artifact_id: str, seen: set[str] | None = None) -> set[str]:
        """Artifact IDs this one was derived from, following parents and prompts."""
        seen = seen if seen is not None else set()
        try:
            art = self.artifacts.load(artifact_id)
        except KeyError:
            return seen
        parents = [art.parent, art.params.get("from_artifact")]
        for parent in parents:
            # Asset refs (im_/vid_) end the chain; only artifacts continue it.
            if isinstance(parent, str) and parent.startswith("a_") and parent not in seen:
                seen.add(parent)
                self._ancestors(parent, seen)
        return seen

    def _shares_lineage(self, check: Check) -> str | None:
        """Two cited artifacts, one derived from the other, are not two opinions."""
        cited = check.cited_artifacts
        for artifact_id in cited:
            overlap = self._ancestors(artifact_id).intersection(cited)
            if overlap:
                return (f"{artifact_id} is derived from {sorted(overlap)[0]}, so they "
                        "share a model's errors")
        return None

    def check_problems(self, check: Check) -> list[str]:
        """Why this check does not count as passing and independent."""
        problems = []
        vague = [m for m in check.methods if method_family(m) == UNCLASSIFIED]
        unclear = [m for m in check.methods if method_family(m) == AMBIGUOUS]
        if len(check.methods) < 2:
            problems.append(f"check {check.kind!r} lists {len(check.methods)} method(s); "
                            "an independent check needs at least two")
        elif vague:
            problems.append(
                f"check {check.kind!r}: cannot tell which model {vague} ran on; name the "
                f"tool in each method, one of {sorted(set(MODEL_FAMILY))}")
        elif unclear:
            problems.append(f"check {check.kind!r}: {unclear} names more than one tool; "
                            "give one method per tool so independence can be judged")
        elif len(check.families) < 2:
            shared = sorted(check.families)[0]
            problems.append(f"check {check.kind!r} methods {check.methods} all use "
                            f"{shared}; two methods sharing a model are not independent. "
                            "Reading the same thing twice with the same model repeats its "
                            "mistakes — use a different tool, or call the claim supported")
        lineage = self._shares_lineage(check)
        if lineage:
            problems.append(f"check {check.kind!r}: {lineage}")
        if not check.passed() and not check.failed():
            problems.append(f"check {check.kind!r} result {check.result!r} is not one of "
                            f"{sorted(PASSING | FAILING)}")
        return problems

    def checks(self, claim: Claim) -> list[Check]:
        """The claim's checks, with cited cross_check artifacts read from disk.

        A check dict carrying `check: a_0009` is replaced wholesale by that
        artifact's recorded verdict; anything the model wrote beside the ID is
        ignored, so it cannot upgrade a disagreement by retyping the result.
        """
        out = []
        for raw, parsed in zip(claim.checks, claim.as_checks()):
            ref = raw.get("check")
            if ref is None:
                out.append(parsed)
                continue
            try:
                art = self.artifacts.load(ref)
            except KeyError:
                art = None
            if art is None or art.kind != "check":
                out.append(Check(kind=f"check {ref}", methods=[], result="missing"))
                continue
            p = art.params
            out.append(Check(kind=f"cross_check:{p['kind']}", methods=list(p["methods"]),
                             result=p["result"], artifacts=list(p["artifacts"]),
                             detail={"check": ref, **p.get("detail", {})}, harness=True))
        return out

    def independent_passes(self, claim: Claim) -> list[Check]:
        return [c for c in self.checks(claim)
                if c.harness and c.passed() and not self.check_problems(c)]

    def warranted_status(self, claim: Claim) -> str:
        checks = self.checks(claim)
        if any(c.failed() for c in checks):
            return CONTESTED
        if not claim.evidence:
            return HYPOTHESIS
        return VERIFIED if self.independent_passes(claim) else SUPPORTED

    # --- validation ------------------------------------------------------
    def problems(self, claims: list[Claim]) -> list[str]:
        found: list[str] = []
        seen_ids: set[str] = set()
        if not claims:
            found.append("no claims submitted; a report needs at least one claim")

        for claim in claims:
            tag = claim.claim_id or "<missing claim_id>"
            if not claim.claim_id:
                found.append("a claim has no claim_id")
            elif claim.claim_id in seen_ids:
                found.append(f"duplicate claim_id {claim.claim_id!r}")
            seen_ids.add(claim.claim_id)

            if not claim.text.strip():
                found.append(f"{tag}: claim text is empty")
            if claim.status not in STATUSES:
                found.append(f"{tag}: status {claim.status!r} is not one of {list(STATUSES)}")
                continue
            found.extend(self._reference_problems(tag, claim))
            status = self._status_problem(tag, claim)
            if status:
                found.append(status)
        return found

    def _kind(self, artifact_id: str) -> str | None:
        try:
            return self.artifacts.load(artifact_id).kind
        except KeyError:
            return None

    def _reference_problems(self, tag: str, claim: Claim) -> list[str]:
        """Cited checks that are not cross_check artifacts, evidence that does not exist."""
        found = []
        for raw in claim.checks:
            ref = raw.get("check") if isinstance(raw, dict) else None
            if ref is not None and self._kind(ref) != "check":
                found.append(f"{tag}: check {ref!r} is not a cross_check artifact")
        for artifact_id in claim.evidence:
            if self._kind(artifact_id) is None:
                found.append(f"{tag}: evidence {artifact_id!r} is not an artifact in "
                             "this workspace")
        return found

    def _status_problem(self, tag: str, claim: Claim) -> str | None:
        """Why the claimed status is stronger than the evidence warrants, if it is."""
        warranted = self.warranted_status(claim)
        if claim.status == HYPOTHESIS and claim.evidence:
            return (f"{tag}: a hypothesis carries no evidence, but {claim.evidence} "
                    "was cited — call it supported instead")
        if claim.status != HYPOTHESIS and not claim.evidence:
            return (f"{tag}: status {claim.status!r} needs at least one evidence "
                    "artifact; a claim with none is a hypothesis")
        if warranted == CONTESTED and claim.status != CONTESTED:
            return (f"{tag}: a check disagreed, so this claim is contested, "
                    f"not {claim.status!r}")
        if claim.status != VERIFIED or warranted == VERIFIED:
            return None
        checks = self.checks(claim)
        detail = [p for check in checks for p in self.check_problems(check)]
        if detail:
            reason = "; ".join(detail)
        elif checks:
            reason = ("checks you write yourself do not count; run cross_check "
                      "and cite the artifact it returns as {\"check\": \"a_…\"}")
        else:
            reason = "no check was run against it"
        return f"{tag}: verified needs a passing independent check — {reason}"

    # --- submission ------------------------------------------------------
    def submit(self, claims: list[Claim], coverage: dict[str, Any] | None = None,
               ) -> "Report":
        problems = self.problems(claims)
        if problems:
            raise RejectedReport("\n".join(f"- {p}" for p in problems))
        # Cited checks are written out in full, so the saved ledger (and the HTML
        # report built from it) reads without the artifact store.
        self.claims = [Claim(**{**asdict(c), "session": self.session,
                                "checks": [self._expand(raw, chk) for raw, chk
                                           in zip(c.checks, self.checks(c))]})
                       for c in claims]
        self.coverage = coverage or {}
        self.save()
        return Report(self.session, self.claims, self.path)

    @staticmethod
    def _expand(raw: dict[str, Any], check: Check) -> dict[str, Any]:
        if not check.harness:
            return raw
        return {"check": raw["check"], "kind": check.kind, "methods": check.methods,
                "result": check.result, "artifacts": check.artifacts,
                **{k: v for k, v in check.detail.items() if k != "check"}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "session": self.session,
                   "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "claims": [asdict(c) for c in self.claims],
                   "coverage": self.coverage}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(self.path)

    @classmethod
    def load(cls, root: str | Path, session: str, artifacts: ArtifactStore) -> "Ledger":
        led = cls(root, session, artifacts)
        if led.path.is_file():
            raw = json.loads(led.path.read_text())
            led.claims = [Claim(**c) for c in raw["claims"]]
            led.coverage = raw.get("coverage", {})
        return led


@dataclass(frozen=True)
class Report:
    session: str
    claims: list[Claim]
    path: Path

    def text(self) -> str:
        lines = [f"Report {self.session} — {len(self.claims)} claims, saved to {self.path}"]
        for status in (VERIFIED, SUPPORTED, CONTESTED):
            group = [c for c in self.claims if c.status == status]
            if not group:
                continue
            lines.append(f"\n{status.upper()} ({len(group)})")
            for claim in group:
                lines.append(f"  {claim.claim_id} {claim.text}")
                lines.append(f"      evidence: {', '.join(claim.evidence)}")
                for check in claim.as_checks():
                    lines.append(f"      check: {check.kind} {check.methods} "
                                 f"-> {check.result}")
        # Hypotheses are reported apart, so an unevidenced guess is never read as a
        # finding just because it sits in the same list.
        guesses = [c for c in self.claims if c.status == HYPOTHESIS]
        if guesses:
            lines.append(f"\nHYPOTHESES — no evidence ({len(guesses)})")
            lines.extend(f"  {c.claim_id} {c.text}" for c in guesses)
        return "\n".join(lines)
