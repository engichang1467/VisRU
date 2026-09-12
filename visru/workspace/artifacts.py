"""Derived images with provenance.

Every image the model is shown gets an ID, a PNG, and a JSON sidecar naming the
parent asset, the tool and parameters that made it, and its coordinate transform.
The sidecar is the contract for Phase 1: given the log, any view can be remade.
"""

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from visru.geometry import ViewTransform

ARTIFACT_DIR = ".visru/artifacts"


@dataclass(frozen=True)
class Mark:
    """A numbered detection or mask, so the model refers to "mark 3", not raw pixels."""

    id: int
    label: str
    box: tuple[float, float, float, float] | None = None  # source pixels
    score: float | None = None
    mask_path: str | None = None  # PNG next to the artifact, in its view frame


@dataclass(frozen=True)
class Artifact:
    id: str
    kind: str  # view | contact_sheet | detect | segment | overlay | plot
    path: str  # workspace-relative
    parent: str | None  # asset ref or another artifact ID
    params: dict[str, Any] = field(default_factory=dict)
    transform: dict[str, Any] | None = None
    marks: list[dict[str, Any]] = field(default_factory=list)
    created: str = ""

    @property
    def view_transform(self) -> ViewTransform | None:
        return ViewTransform(**self.transform) if self.transform else None

    def mark(self, mark_id: int) -> Mark:
        for m in self.marks:
            if m["id"] == mark_id:
                # JSON has no tuples; a box must come back comparable to what went in.
                box = m.get("box")
                return Mark(**{**m, "box": tuple(box) if box is not None else None})
        have = ", ".join(str(m["id"]) for m in self.marks) or "none"
        raise KeyError(f"{self.id} has no mark {mark_id} (marks: {have})")


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.dir = self.root / ARTIFACT_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self._next = 1 + max((int(p.stem[2:]) for p in self.dir.glob("a_*.json")), default=0)

    def _new_id(self) -> str:
        # The counter is a boot-time snapshot. If a second process has written
        # artifacts since, skip past them rather than overwrite cited evidence.
        while (self.dir / f"a_{self._next:04d}.json").exists():
            self._next += 1
        artifact_id = f"a_{self._next:04d}"
        self._next += 1
        return artifact_id

    def save(self, image: Image.Image, kind: str, parent: str | None = None,
             params: dict[str, Any] | None = None,
             transform: ViewTransform | None = None,
             marks: list[Mark] | None = None,
             mask_images: dict[int, Image.Image] | None = None) -> Artifact:
        artifact_id = self._new_id()
        png = self.dir / f"{artifact_id}.png"
        image.save(png, format="PNG")

        # Masks are written here, not by the caller, because only the store knows the
        # artifact's ID — and a mask whose path never lands in its Mark is unfindable.
        stored = []
        for mark in marks or []:
            img = (mask_images or {}).get(mark.id)
            if img is None:
                stored.append(mark)
                continue
            path = self.dir / f"{artifact_id}_mark{mark.id}_mask.png"
            img.convert("L").save(path, format="PNG")
            stored.append(replace(mark, mask_path=path.relative_to(self.root).as_posix()))

        art = Artifact(
            id=artifact_id,
            kind=kind,
            path=png.relative_to(self.root).as_posix(),
            parent=parent,
            params=params or {},
            transform=asdict(transform) if transform else None,
            marks=[asdict(m) for m in stored],
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        (self.dir / f"{artifact_id}.json").write_text(json.dumps(asdict(art), indent=1))
        return art

    def load_mask(self, artifact_id: str, mark_id: int) -> Image.Image:
        """The mask for one mark, in the artifact's own view frame.

        Not source resolution: the artifact's transform is what places it, and
        storing a 12 MP mask per mark of a 4K image would be wasteful.
        """
        mark = self.load(artifact_id).mark(mark_id)
        if not mark.mask_path:
            raise KeyError(f"{artifact_id} mark {mark_id} has no mask")
        with Image.open(self.root / mark.mask_path) as img:
            return img.convert("L")

    def load(self, artifact_id: str) -> Artifact:
        sidecar = self.dir / f"{artifact_id}.json"
        if not sidecar.is_file():
            raise KeyError(f"unknown artifact {artifact_id!r}")
        return Artifact(**json.loads(sidecar.read_text()))

    def load_image(self, artifact_id: str) -> Image.Image:
        with Image.open(self.root / self.load(artifact_id).path) as img:
            return img.convert("RGB")

    def to_source(self, artifact_id: str, u: float, v: float) -> tuple[float, float]:
        """Map view pixels in an artifact back to source-asset pixels."""
        tf = self.load(artifact_id).view_transform
        if tf is None:
            raise ValueError(f"{artifact_id} carries no transform")
        return tf.to_source(u, v)
