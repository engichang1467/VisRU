"""Embedding index: one SigLIP vector per image and per sampled video frame.

What `search` reads. A video is sampled at a fixed rate (midpoints of equal
intervals, as `video.frame_times` does for contact sheets), so rows of one video
are in time order and neighbouring rows are neighbouring moments — which is what
lets search merge adjacent hits into a time window.

ponytail: brute-force numpy cosine over one .npz. Fine to ~1e6 vectors; FAISS
when an index outgrows that, or memory.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from visru import models
from visru.workspace import video
from visru.workspace.registry import Registry, parse_ref

INDEX_FILE = ".visru/index/embeddings.npz"
BATCH = 32  # frames decoded and embedded at once; bounds memory on long videos


@dataclass
class Index:
    refs: list[str]  # `im_x` or `vid_x#<frame_index>`
    times: np.ndarray  # seconds; NaN for images
    vectors: np.ndarray  # (n, d) float16, unit length
    model: str

    def __len__(self) -> int:
        return len(self.refs)

    def assets(self) -> set[str]:
        return {parse_ref(r)[0] for r in self.refs}


def path(root: str | Path) -> Path:
    return Path(root) / INDEX_FILE


def load(root: str | Path) -> Index | None:
    p = path(root)
    if not p.is_file():
        return None
    with np.load(p) as raw:
        return Index(refs=[str(r) for r in raw["refs"]], times=raw["times"],
                     vectors=raw["vectors"], model=str(raw["model"]))


def save(root: str | Path, index: Index) -> None:
    p = path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name("embeddings.tmp.npz")
    np.savez(tmp, refs=np.array(index.refs), times=index.times,
             vectors=index.vectors, model=np.array(index.model))
    tmp.replace(p)  # atomic, like the registry


def _video_rows(registry: Registry, asset_id: str, fps: float):
    """(ref, time, image) for frames sampled at `fps`, in batches."""
    meta = registry.video_meta(asset_id)
    n = max(1, math.ceil(meta.duration * fps))
    times = video.frame_times(meta, n)
    seen: set[int] = set()
    for i in range(0, len(times), BATCH):
        frames = video.decode_at(registry.abspath(asset_id), times[i:i + BATCH], meta)
        # Sampling near the frame rate lands two midpoints on one frame (the last,
        # usually); a duplicate row would count one moment twice in a window.
        fresh = []
        for f in frames:
            if f.index not in seen:
                seen.add(f.index)
                fresh.append(f)
        yield [(f"{asset_id}#{f.index}", f.time, f.image) for f in fresh]


def build(root: str | Path, registry: Registry, fps: float = 1.0,
          progress=None) -> tuple[Index, int]:
    """Embed every asset not already indexed. Returns (index, assets added).

    Asset IDs are content hashes, so an asset already in the index is unchanged
    and skipped; one gone from the registry is dropped. A different embedding
    model means the old vectors are incomparable, so the index starts over.
    """
    emb = models.embedder()
    old = load(root)
    if old is not None and old.model != emb.name:
        old = None
    live = set(registry.assets)
    keep = [i for i, r in enumerate(old.refs) if parse_ref(r)[0] in live] if old else []
    refs = [old.refs[i] for i in keep] if old else []
    times = [float(old.times[i]) for i in keep] if old else []
    vectors = [old.vectors[keep]] if old and keep else []
    have = {parse_ref(r)[0] for r in refs}

    added = 0
    for asset in registry.list():
        if asset.id in have:
            continue
        if asset.kind == "image":
            batches = [[(asset.id, math.nan, registry.load_image(asset.id))]]
        else:
            batches = _video_rows(registry, asset.id, fps)
        for batch in batches:
            if not batch:
                continue
            vectors.append(emb.images([im for _, _, im in batch]).astype(np.float16))
            refs.extend(r for r, _, _ in batch)
            times.extend(t for _, t, _ in batch)
        added += 1
        if progress:
            progress(asset)

    dim = vectors[0].shape[1] if vectors else 0
    index = Index(refs=refs, times=np.array(times, dtype=float),
                  vectors=np.concatenate(vectors) if vectors else np.zeros((0, dim),
                                                                           np.float16),
                  model=emb.name)
    save(root, index)
    return index, added


@dataclass(frozen=True)
class Hit:
    ref: str
    time: float | None
    score: float
    row: int  # position in the index; adjacent rows of one video are adjacent samples


def query(index: Index, text: str, asset: str | None = None, t0: float | None = None,
          t1: float | None = None, k: int = 8) -> list[Hit]:
    """Top-k rows by cosine similarity to `text`, optionally within one asset and
    a time range."""
    if not len(index):
        return []
    q = models.embedder().text(text).astype(np.float32)
    scores = index.vectors.astype(np.float32) @ q

    mask = np.ones(len(index), bool)
    if asset:
        mask &= np.array([parse_ref(r)[0] == asset for r in index.refs])
    if t0 is not None:
        mask &= ~(index.times < t0)  # NaN (images) never falls outside a range
    if t1 is not None:
        mask &= ~(index.times > t1)
    rows = np.flatnonzero(mask)
    top = rows[np.argsort(-scores[rows], kind="stable")[:k]]
    return [Hit(index.refs[i], None if math.isnan(index.times[i]) else float(index.times[i]),
                float(scores[i]), int(i)) for i in top]


def windows(index: Index, hits: list[Hit], duration: dict[str, float]) -> list[dict]:
    """Merge hits on neighbouring samples of one video into time windows.

    Two hits merge when at most one unhit sample lies between them. Each window is
    padded by half a sample step on both sides, since a sample stands for the
    interval around it. Ordered by best score in the window.
    """
    by_asset: dict[str, list[Hit]] = {}
    for h in hits:
        if h.time is not None:
            by_asset.setdefault(parse_ref(h.ref)[0], []).append(h)

    out = []
    for asset_id, group in by_asset.items():
        group.sort(key=lambda h: h.row)
        rows = [i for i, r in enumerate(index.refs) if parse_ref(r)[0] == asset_id]
        step = (index.times[rows[1]] - index.times[rows[0]]) if len(rows) > 1 else 0.0
        run = [group[0]]
        for h in group[1:] + [None]:
            if h is not None and h.row - run[-1].row <= 2:
                run.append(h)
                continue
            t0 = max(0.0, run[0].time - step / 2)
            t1 = min(duration.get(asset_id, math.inf), run[-1].time + step / 2)
            out.append({"asset": asset_id, "t0": round(float(t0), 2),
                        "t1": round(float(t1), 2),
                        "score": round(max(x.score for x in run), 4),
                        "frames": [x.ref for x in run]})
            run = [h] if h is not None else []
    return sorted(out, key=lambda w: -w["score"])
