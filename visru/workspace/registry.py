"""Asset registry: content-hash IDs and metadata for a workspace folder.

IDs are `im_<sha12>` / `vid_<sha12>`, so the same bytes always get the same ID and
a re-index is stable. Video frames are addressed as `vid_<sha12>#<frame_index>`.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from visru.workspace import video

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
STATE_DIR = ".visru"
REGISTRY_FILE = "registry.json"


@dataclass(frozen=True)
class Asset:
    id: str
    kind: str  # "image" | "video"
    paths: tuple[str, ...]  # workspace-relative, sorted; identical bytes share an ID
    width: int
    height: int
    size_bytes: int
    sha256: str
    fps: float | None = None
    frames: int | None = None
    duration: float | None = None

    @property
    def path(self) -> str:
        return self.paths[0]

    @property
    def name(self) -> str:
        return Path(self.paths[0]).name

    def summary(self) -> str:
        extra = "" if len(self.paths) == 1 else f" (+{len(self.paths) - 1} duplicate paths)"
        if self.kind == "video":
            return (f"{self.id} {self.name} video {self.width}x{self.height} "
                    f"{self.frames} frames @ {self.fps:.3g} fps "
                    f"({self.duration:.1f}s){extra}")
        return f"{self.id} {self.name} image {self.width}x{self.height}{extra}"


@dataclass
class ScanStats:
    new: int = 0  # assets not in the registry before
    hashed: int = 0  # files read from disk this scan
    cached: int = 0  # files skipped on size+mtime
    removed: int = 0  # assets whose files are all gone
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        line = (f"{self.new} new, {self.hashed} hashed, {self.cached} cached, "
                f"{self.removed} removed")
        return line + (f", {len(self.failed)} failed" if self.failed else "")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _kind(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    return None


def parse_ref(ref: str) -> tuple[str, int | None]:
    """Split `vid_abc123#1164` into ("vid_abc123", 1164)."""
    asset_id, _, frame = ref.partition("#")
    if not frame:
        return asset_id, None
    if not frame.isdigit():
        raise ValueError(f"bad frame index in {ref!r}; expected <asset_id>#<frame_index>")
    return asset_id, int(frame)


class Registry:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.assets: dict[str, Asset] = {}
        # workspace-relative path -> (size, mtime_ns, asset_id); lets a re-index skip
        # hashing unchanged files, which is the whole cost of indexing a 4K folder.
        self.files: dict[str, tuple[int, int, str]] = {}

    @property
    def state_path(self) -> Path:
        return self.root / STATE_DIR / REGISTRY_FILE

    @classmethod
    def load(cls, root: str | Path) -> "Registry":
        reg = cls(root)
        if reg.state_path.is_file():
            raw = json.loads(reg.state_path.read_text())
            reg.assets = {a["id"]: Asset(**{**a, "paths": tuple(a["paths"])})
                          for a in raw["assets"]}
            reg.files = {p: (v[0], v[1], v[2]) for p, v in raw.get("files", {}).items()}
        return reg

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1,
                   "assets": [asdict(a) for a in self.list()],
                   "files": {p: list(v) for p, v in sorted(self.files.items())}}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(self.state_path)  # atomic: an interrupted index keeps the old one

    def scan(self) -> ScanStats:
        """Walk the workspace and refresh the registry. Re-hashes only changed files."""
        stats = ScanStats()
        old_ids = set(self.assets)
        assets: dict[str, Asset] = {}
        files: dict[str, tuple[int, int, str]] = {}

        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or STATE_DIR in path.parts:
                continue
            kind = _kind(path)
            if kind is None:
                continue
            rel = path.relative_to(self.root).as_posix()
            st = path.stat()

            cached = self.files.get(rel)
            reuse = cached and cached[0] == st.st_size and cached[1] == st.st_mtime_ns
            asset = self.assets.get(cached[2]) if reuse else None
            if asset is None:
                try:
                    asset = self._build(path, rel, kind, st.st_size)
                except Exception as exc:  # unreadable or corrupt; keep indexing the rest
                    stats.failed.append(f"{rel}: {type(exc).__name__}: {exc}")
                    continue
                stats.hashed += 1
            else:
                stats.cached += 1

            files[rel] = (st.st_size, st.st_mtime_ns, asset.id)
            prev = assets.get(asset.id)
            paths = tuple(sorted(set(prev.paths) | {rel})) if prev else (rel,)
            assets[asset.id] = Asset(**{**asdict(asset), "paths": paths})

        stats.new = len(set(assets) - old_ids)
        stats.removed = len(old_ids - set(assets))
        self.assets, self.files = assets, files
        return stats

    def _build(self, path: Path, rel: str, kind: str, size: int) -> Asset:
        sha = _sha256(path)
        asset_id = f"{'im' if kind == 'image' else 'vid'}_{sha[:12]}"
        if kind == "image":
            with Image.open(path) as img:
                w, h = img.size
            return Asset(asset_id, kind, (rel,), w, h, size, sha)
        meta = video.probe(path)
        return Asset(asset_id, kind, (rel,), meta.width, meta.height, size, sha,
                     fps=meta.fps, frames=meta.frames, duration=meta.duration)

    def get(self, ref: str) -> Asset:
        asset_id, _ = parse_ref(ref)
        try:
            return self.assets[asset_id]
        except KeyError:
            raise KeyError(f"unknown asset {asset_id!r}; run `visru index` first") from None

    def abspath(self, ref: str) -> Path:
        return self.root / self.get(ref).path

    def list(self, prefix: str | None = None, kind: str | None = None,
             contains: str | None = None) -> list[Asset]:
        """Filter by filename prefix, asset kind, or substring of the path."""
        out = []
        for asset in self.assets.values():
            if kind and asset.kind != kind:
                continue
            if prefix and not any(Path(p).name.startswith(prefix) for p in asset.paths):
                continue
            if contains and not any(contains in p for p in asset.paths):
                continue
            out.append(asset)
        return sorted(out, key=lambda a: a.path)

    def video_meta(self, ref: str) -> video.VideoMeta:
        asset = self.get(ref)
        if asset.kind != "video":
            raise ValueError(f"{asset.id} is an image, not a video")
        return video.VideoMeta(asset.width, asset.height, asset.fps or 0.0,
                               asset.frames or 0, asset.duration or 0.0)

    def load_image(self, ref: str) -> Image.Image:
        """Load an image asset, or one frame of a video (`vid_abc123#1164`)."""
        asset_id, frame = parse_ref(ref)
        asset = self.get(asset_id)
        path = self.root / asset.path
        if asset.kind == "video":
            if frame is None:
                raise ValueError(
                    f"{asset_id} is a video; address a frame as {asset_id}#<frame_index>")
            return video.decode_index(path, frame, self.video_meta(asset_id)).image
        if frame is not None:
            raise ValueError(f"{asset_id} is an image; it has no frames")
        with Image.open(path) as img:
            return img.convert("RGB")
