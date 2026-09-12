"""Coordinate round trips. The most important unit tests in the repo.

A crop of a crop of a crop must still report source-asset pixels, or every claim
the agent makes about "where" is quietly wrong.
"""

import math
import random

import pytest

from visru.config import ModelConfig
from visru.geometry import ViewTransform, fit_render_size

LIMITS = ModelConfig().limits
TOL = 1e-6


def _view_of(parent: ViewTransform, box_view: tuple[float, float, float, float]) -> ViewTransform:
    """Crop a region given in the parent's *view* pixels, render it, return its transform.

    Mirrors what the `view` tool does: the box arrives in source pixels (here,
    converted from view pixels by the caller, as the model would), the crop is
    rendered to fit the model limits, and the transform carries the cumulative
    scale back to the source asset.
    """
    u0, v0, u1, v1 = box_view
    x0, y0 = parent.to_source(u0, v0)
    x1, y1 = parent.to_source(u1, v1)
    cw, ch = max(1, round(x1 - x0)), max(1, round(y1 - y0))
    _, _, scale = fit_render_size(cw, ch, **LIMITS)
    return ViewTransform(parent.asset_id, x0, y0, scale)


def _root(w: int, h: int) -> ViewTransform:
    _, _, scale = fit_render_size(w, h, **LIMITS)
    return ViewTransform("im_test", 0.0, 0.0, scale)


def test_round_trip_identity():
    tf = ViewTransform("im_test", 812.0, 402.0, 1.7)
    for x, y in [(0.0, 0.0), (812.0, 402.0), (1240.5, 690.25)]:
        u, v = tf.to_view(x, y)
        rx, ry = tf.to_source(u, v)
        assert rx == pytest.approx(x, abs=TOL)
        assert ry == pytest.approx(y, abs=TOL)


def test_round_trip_nested_crops():
    """Three levels of zoom, random boxes, random points."""
    rng = random.Random(0)
    for _ in range(200):
        w, h = rng.randint(64, 8000), rng.randint(64, 8000)
        tf = _root(w, h)
        chain = [tf]
        for _ in range(3):
            vw, vh = w * tf.scale, h * tf.scale
            u0, v0 = rng.uniform(0, vw * 0.5), rng.uniform(0, vh * 0.5)
            u1, v1 = rng.uniform(u0 + 8, vw), rng.uniform(v0 + 8, vh)
            tf = _view_of(tf, (u0, v0, u1, v1))
            chain.append(tf)

        for level in chain:
            for _ in range(5):
                x, y = rng.uniform(0, w), rng.uniform(0, h)
                rx, ry = level.to_source(*level.to_view(x, y))
                assert rx == pytest.approx(x, rel=1e-9, abs=1e-6)
                assert ry == pytest.approx(y, rel=1e-9, abs=1e-6)


def test_levels_agree_on_the_same_source_point():
    """A point named in a deep view must map to the same source pixel the
    shallower views report, otherwise marks drift as the agent zooms."""
    rng = random.Random(1)
    w, h = 4032, 3024
    parent = _root(w, h)
    child = _view_of(parent, (100.0, 80.0, 900.0, 700.0))
    grandchild = _view_of(child, (10.0, 10.0, 400.0, 300.0))

    for _ in range(100):
        # Pick a point inside the deepest crop, in its own view pixels.
        u, v = rng.uniform(0, 380), rng.uniform(0, 290)
        src = grandchild.to_source(u, v)
        # Hand it up the chain and back down: every level agrees on the source pixel.
        via_child = child.to_source(*child.to_view(*src))
        via_parent = parent.to_source(*parent.to_view(*src))
        assert via_child == pytest.approx(src, abs=1e-6)
        assert via_parent == pytest.approx(src, abs=1e-6)


def test_fit_render_size_respects_limits():
    rng = random.Random(2)
    sizes = [(rng.randint(1, 12000), rng.randint(1, 12000)) for _ in range(500)]
    sizes += [(1, 1), (10000, 8), (8, 10000), (1, 12000), (4032, 3024), (28, 28)]
    for w, h in sizes:
        rw, rh, scale = fit_render_size(w, h, **LIMITS)
        tokens = math.ceil(rw / LIMITS["patch"]) * math.ceil(rh / LIMITS["patch"])
        assert tokens <= LIMITS["max_visual_tokens"], (w, h, rw, rh)
        assert max(rw, rh) <= LIMITS["max_edge"], (w, h, rw, rh)
        assert scale <= LIMITS["max_upscale"] + TOL, (w, h, scale)
        assert rw >= 1 and rh >= 1
        assert rw == max(1, round(w * scale)) and rh == max(1, round(h * scale))


def test_far_corner_round_trips_within_half_a_rendered_pixel():
    """The far corner is where a wrong scale shows up first. One scale serves both
    axes, so each must land within half a rendered pixel of the render size."""
    rng = random.Random(3)
    sizes = [(rng.randint(16, 9000), rng.randint(16, 9000)) for _ in range(500)]
    sizes += [(1055, 7913), (7913, 1055)]  # extreme aspect: caught scale = rw / w
    for w, h in sizes:
        rw, rh, scale = fit_render_size(w, h, **LIMITS)
        tf = ViewTransform("im_test", 0.0, 0.0, scale)
        u, v = tf.to_view(w, h)
        assert abs(u - rw) <= 0.5, (w, h, rw, rh, u)
        assert abs(v - rh) <= 0.5, (w, h, rw, rh, v)


def test_fit_render_size_upscales_small_crops_but_not_past_the_cap():
    rw, rh, scale = fit_render_size(40, 30, **LIMITS)
    assert (rw, rh, scale) == (80, 60, 2.0)


def test_fit_render_size_rejects_empty_crops():
    for w, h in [(0, 10), (10, 0), (-1, 5)]:
        with pytest.raises(ValueError):
            fit_render_size(w, h, **LIMITS)
