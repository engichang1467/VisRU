"""Video decoding. Frames are addressed by index; time is derived from fps.

ponytail: assumes constant frame rate. A VFR source will report an index that
does not match its own timestamps. Store per-frame pts instead if that turns up.
"""

from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frames: int
    duration: float


@dataclass(frozen=True)
class DecodedFrame:
    index: int
    time: float
    image: Image.Image


def probe(path: str | Path) -> VideoMeta:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        duration = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        frames = stream.frames or (round(duration * fps) if fps else 0)
        return VideoMeta(stream.codec_context.width, stream.codec_context.height,
                         fps, frames, duration)


def frame_times(meta: VideoMeta, n: int, t0: float | None = None,
                t1: float | None = None) -> list[float]:
    """n timestamps spread over [t0, t1], sampled at interval midpoints.

    Midpoints, not endpoints: the last frame of a video is often a cut or a black
    frame, and sampling it wastes one of the n slots the model paid for.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    start = 0.0 if t0 is None else max(0.0, t0)
    end = meta.duration if t1 is None else min(meta.duration, t1)
    if end <= start:
        return [start]
    step = (end - start) / n
    return [start + step * (i + 0.5) for i in range(n)]


def decode_at(path: str | Path, times: list[float], meta: VideoMeta | None = None,
              ) -> list[DecodedFrame]:
    """Decode the frame at or just after each timestamp. Seeks per target."""
    meta = meta or probe(path)
    out: list[DecodedFrame] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base
        for t in sorted(times):
            container.seek(int(t / tb), stream=stream)  # lands on the keyframe before t
            picked = None
            for frame in container.decode(stream):
                picked = frame
                if frame.time is not None and frame.time >= t - 1e-6:
                    break
            if picked is None:
                continue
            ftime = float(picked.time or 0.0)
            out.append(DecodedFrame(
                index=round(ftime * meta.fps) if meta.fps else 0,
                time=ftime,
                image=picked.to_image(),
            ))
    return out


def decode_index(path: str | Path, index: int, meta: VideoMeta | None = None,
                 ) -> DecodedFrame:
    """Decode one frame by index."""
    meta = meta or probe(path)
    if not meta.fps:
        raise ValueError(f"{path} reports no frame rate; cannot address frames by index")
    if index < 0 or (meta.frames and index >= meta.frames):
        raise IndexError(f"frame {index} out of range (0..{meta.frames - 1})")
    frames = decode_at(path, [index / meta.fps], meta)
    if not frames:
        raise IndexError(f"frame {index} could not be decoded from {path}")
    return frames[0]
