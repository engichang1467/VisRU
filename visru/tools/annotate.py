"""Drawing on rendered views. Labels are always in source-asset pixels."""

import math

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from visru.geometry import ViewTransform

RULE = (255, 214, 0)  # amber: rare in photographs, legible on both dark and light
INK = (0, 0, 0)
PAPER = (255, 255, 255)


def font(size: int = 15) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _nice_step(raw: float) -> int:
    """Round up to the next 1/2/5 x 10^k, so tick labels stay readable numbers."""
    raw = max(raw, 1.0)
    power = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 5, 10):
        if mult * power >= raw:
            return int(mult * power)
    return int(10 * power)


def label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
          fnt: ImageFont.ImageFont | None = None, fill=RULE, anchor: str = "la") -> None:
    """Text with an outline, so it survives whatever is underneath it."""
    draw.text(xy, text, font=fnt or font(), fill=fill, anchor=anchor,
              stroke_width=2, stroke_fill=INK)


def draw_grid(img: Image.Image, tf: ViewTransform, target_spacing: int = 140,
              line_alpha: int = 60) -> Image.Image:
    """Overlay a ruler labelled in source-asset pixels.

    Lines are faint and ticks are opaque: the model zoomed in to see detail, and a
    solid grid would cover the thing it came to look at.
    """
    step = _nice_step(target_spacing / tf.scale)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fnt = font()
    faint = (*RULE, line_alpha)
    solid = (*RULE, 255)

    x = math.ceil(tf.x0 / step) * step
    while True:
        u, _ = tf.to_view(x, 0)
        if u > img.width:
            break
        if u >= 0:
            draw.line([(u, 0), (u, img.height)], fill=faint)
            draw.line([(u, 0), (u, 12)], fill=solid, width=2)
            if u <= img.width - 44:  # a label clipped by the edge is worse than none
                label(draw, (u + 4, 3), f"{int(x)}", fnt, fill=solid)
        x += step

    y = math.ceil(tf.y0 / step) * step
    while True:
        _, v = tf.to_view(0, y)
        if v > img.height:
            break
        if v >= 0:
            draw.line([(0, v), (img.width, v)], fill=faint)
            draw.line([(0, v), (12, v)], fill=solid, width=2)
            if v <= img.height - 22:
                # Clear the x-label band, or the first row's labels stack on top of
                # each other in the corner and neither is readable.
                label(draw, (4, max(v + 4, 24)), f"{int(y)}", fnt, fill=solid)
        y += step

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


# Distinct at a glance and distinguishable when two marks overlap.
MARK_COLORS = [(255, 59, 48), (0, 200, 255), (255, 214, 0), (120, 255, 80),
               (255, 120, 220), (255, 140, 0), (150, 150, 255), (0, 255, 190)]


def mark_color(mark_id: int) -> tuple[int, int, int]:
    return MARK_COLORS[(mark_id - 1) % len(MARK_COLORS)]


def draw_marks(img: Image.Image, marks, tf: ViewTransform,
               masks: dict[int, Image.Image] | None = None) -> Image.Image:
    """Number every detection on the image (Set-of-Mark).

    The model then says "mark 3" instead of emitting pixel coordinates, which is the
    one thing it reliably gets wrong. Mark boxes arrive in source pixels.
    """
    out = img.convert("RGBA")
    # Numbers must stay readable whether the render is 400 px tall or 2000.
    size = max(16, min(44, round(out.height / 34)))
    fnt = font(size)
    pad, stroke = max(4, size // 3), max(2, size // 8)

    if masks:
        overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
        for mark_id, mask in masks.items():
            colour = mark_color(mark_id)
            flat = mask.convert("L").resize(out.size)
            overlay.paste(Image.new("RGBA", out.size, (*colour, 80)), (0, 0), flat)
            # Outline the mask too: a tint the same colour as the object it covers
            # is invisible, and masks land on objects of every colour.
            edge = flat.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(5))
            overlay.paste(Image.new("RGBA", out.size, (*INK, 255)), (0, 0), edge)
            inner = flat.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
            overlay.paste(Image.new("RGBA", out.size, (*colour, 255)), (0, 0), inner)
        out = Image.alpha_composite(out, overlay)

    draw = ImageDraw.Draw(out)
    for mark in marks:
        colour = mark_color(mark.id)
        if mark.box is None:
            continue
        u0, v0 = tf.to_view(mark.box[0], mark.box[1])
        u1, v1 = tf.to_view(mark.box[2], mark.box[3])
        draw.rectangle([u0, v0, u1, v1], outline=colour, width=stroke)

        # Chip inside the box: a label hanging outside gets clipped at the edge.
        text = str(mark.id)
        tw = draw.textlength(text, font=fnt)
        draw.rectangle([u0, v0, u0 + tw + 2 * pad, v0 + size + pad], fill=(*INK, 255))
        draw.text((u0 + pad, v0 + (size + pad) / 2), text, font=fnt, fill=colour,
                  anchor="lm")
    return out.convert("RGB")


def caption(img: Image.Image, text: str, height: int = 22) -> Image.Image:
    """Add a caption bar under an image (contact-sheet cells)."""
    out = Image.new("RGB", (img.width, img.height + height), INK)
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.text((4, img.height + height / 2), text, font=font(14), fill=PAPER, anchor="lm")
    return out
