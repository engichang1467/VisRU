"""Coordinate handling. Every tool input and output uses source-asset pixels."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ViewTransform:
    """Maps pixels in a rendered view back to the source asset."""

    asset_id: str
    x0: float  # crop origin, source pixels
    y0: float
    scale: float  # rendered px per source px (aspect ratio preserved)

    def to_source(self, u: float, v: float) -> tuple[float, float]:
        return self.x0 + u / self.scale, self.y0 + v / self.scale

    def to_view(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x0) * self.scale, (y - self.y0) * self.scale


def fit_render_size(
    w: int,
    h: int,
    max_edge: int,
    max_visual_tokens: int,
    patch: int = 28,
    max_upscale: float = 2.0,
) -> tuple[int, int, float]:
    """Largest size within the model's limits, so the API never resizes again.

    Returns (render_w, render_h, scale) where scale is rendered px per source px.
    """
    if w < 1 or h < 1:
        raise ValueError(f"source size must be positive, got {w}x{h}")

    s = min(max_upscale, max_edge / max(w, h))
    while True:
        rw, rh = max(1, round(w * s)), max(1, round(h * s))
        if math.ceil(rw / patch) * math.ceil(rh / patch) <= max_visual_tokens:
            return rw, rh, s
        s *= 0.98
