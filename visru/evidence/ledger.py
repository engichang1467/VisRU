"""Claims, the evidence behind them, and the checks run against them.

The ledger is the only place a status is decided. The model proposes one; the ledger
computes what the evidence actually warrants and refuses anything stronger. That
refusal is the whole mechanism — a status the model can assert at will measures
nothing, and Phase 2 exists to test whether `verified` predicts correctness.
"""

import json
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
    "view": "vlm", "vlm": "vlm", "read": "vlm", "count": "vlm",
    "detect": "owlv2", "segment": "sam2", "ocr": "paddleocr",
    "depth": "depth_anything", "pointmap": "vggt",
    "python": "code", "measure": "code", "geometry": "code",
}


@dataclass(frozen=True)
class Check:
    kind: str
    methods: list[str]
    result: str
    artifacts: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def families(self) -> set[str]:
        # An unknown method is its own family: assuming independence would be the
        # generous reading, and the ledger's job is the ungenerous one.
        return {MODEL_FAMILY.get(m.lower(), f"other:{m.lower()}") for m in self.methods}

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
        for artifact_id in check.artifacts:
            ancestors = self._ancestors(artifact_id)
            overlap = ancestors.intersection(check.artifacts)
            if overlap:
                return (f"{artifact_id} is derived from {sorted(overlap)[0]}, so they "
                        "share a model's errors")
        return None

    def check_problems(self, check: Check) -> list[str]:
        """Why this check does not count as passing and independent."""
        problems = []
        if len(check.methods) < 2:
            problems.append(f"check {check.kind!r} lists {len(check.methods)} method(s); "
                            "an independent check needs at least two")
        elif len(check.families) < 2:
            shared = sorted(check.families)[0]
            problems.append(f"check {check.kind!r} methods {check.methods} all use "
                            f"{shared}; two methods sharing a model are not independent")
        lineage = self._shares_lineage(check)
        if lineage:
            problems.append(f"check {check.kind!r}: {lineage}")
        if not check.passed() and not check.failed():
            problems.append(f"check {check.kind!r} result {check.result!r} is not one of "
                            f"{sorted(PASSING | FAILING)}")
        return problems

    def independent_passes(self, claim: Claim) -> list[Check]:
        return [c for c in claim.as_checks()
                if c.passed() and not self.check_problems(c)]

    def warranted_status(self, claim: Claim) -> str:
        checks = claim.as_checks()
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

            for artifact_id in claim.evidence:
                try:
                    self.artifacts.load(artifact_id)
                except KeyError:
                    found.append(f"{tag}: evidence {artifact_id!r} is not an artifact in "
                                 "this workspace")

            warranted = self.warranted_status(claim)
            if claim.status == HYPOTHESIS and claim.evidence:
                found.append(f"{tag}: a hypothesis carries no evidence, but {claim.evidence} "
                             "was cited — call it supported instead")
            elif claim.status != HYPOTHESIS and not claim.evidence:
                found.append(f"{tag}: status {claim.status!r} needs at least one evidence "
                             "artifact; a claim with none is a hypothesis")
            elif warranted == CONTESTED and claim.status != CONTESTED:
                found.append(f"{tag}: a check disagreed, so this claim is contested, "
                             f"not {claim.status!r}")
            elif claim.status == VERIFIED and warranted != VERIFIED:
                detail = []
                for check in claim.as_checks():
                    detail.extend(self.check_problems(check))
                reason = ("; ".join(detail) if detail
                          else "no check was run against it")
                found.append(f"{tag}: verified needs a passing independent check — {reason}")
        return found

    # --- submission ------------------------------------------------------
    def submit(self, claims: list[Claim]) -> "Report":
        problems = self.problems(claims)
        if problems:
            raise RejectedReport("\n".join(f"- {p}" for p in problems))
        self.claims = [Claim(**{**asdict(c), "session": self.session}) for c in claims]
        self.save()
        return Report(self.session, self.claims, self.path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "session": self.session,
                   "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "claims": [asdict(c) for c in self.claims]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(self.path)

    @classmethod
    def load(cls, root: str | Path, session: str, artifacts: ArtifactStore) -> "Ledger":
        led = cls(root, session, artifacts)
        if led.path.is_file():
            raw = json.loads(led.path.read_text())
            led.claims = [Claim(**c) for c in raw["claims"]]
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
