"""Generate a 1280×640 social preview image for GitHub using Pillow."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    """Find a suitable font on the system, fall back to default."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/SFNSText.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for path in font_paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: str,
) -> None:
    """Draw a rounded rectangle."""
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def generate_social_preview(output_path: str = "assets/social-preview.png") -> None:
    """Generate a 1280×640 social preview image."""
    W, H = 1280, 640

    # Create image with dark background
    img = Image.new("RGBA", (W, H), _rgb("#0d1117"))
    draw = ImageDraw.Draw(img)

    # --- Background gradient bands ---
    for i in range(H):
        ratio = i / H
        r = int(13 + (22 - 13) * ratio)
        g = int(17 + (27 - 17) * ratio)
        b = int(23 + (39 - 23) * ratio)
        draw.line([(0, i), (W, i)], fill=(r, g, b))

    # --- Accent diagonal stripe ---
    accent = _rgb("#58a6ff")
    accent_dark = _rgb("#1f6feb")
    for i in range(H):
        t = i / H
        r = int(88 * (1 - t) + 31 * t)
        g = int(166 * (1 - t) + 111 * t)
        b = int(255 * (1 - t) + 235 * t)
        draw.line([(int(W * 0.75), i), (W, i)], fill=(r, g, b), width=1)

    # --- Graph node visual (left side) ---
    # We'll draw a stylized graph: 3 nodes + connectors
    node_color = _rgb("#58a6ff")
    node_color_2 = _rgb("#3fb950")
    node_color_3 = _rgb("#f0883e")
    edge_color = _rgb("#30363d")

    nodes = [
        (150, 200, 100, 60, "Route", node_color),
        (150, 310, 100, 60, "Retrieve", node_color_2),
        (150, 420, 100, 60, "Generate", node_color_3),
        (320, 255, 100, 60, "Grade", node_color),
        (320, 365, 100, 60, "Transform", node_color_2),
        (490, 310, 100, 60, "Hallucination\nCheck", node_color_3),
    ]

    font_small = _find_font(14)

    for x, y, w, h, label, color in nodes:
        _draw_rounded_rect(draw, (x, y, x + w, y + h), 10, fill=color)
        # Draw label
        lines = label.split("\n")
        for li, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font_small)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x + (w - tw) / 2, y + (h - th * len(lines)) / 2 + li * (th + 2)),
                line,
                fill="#ffffff",
                font=font_small,
            )

    # --- Edges between nodes ---
    edges = [
        ((200, 230), (200, 280)),
        ((200, 340), (200, 390)),
        ((250, 230), (320, 240)),
        ((250, 340), (320, 340)),
        ((420, 285), (490, 300)),
        ((420, 395), (490, 390)),
        ((320, 285), (320, 240)),
        ((320, 395), (320, 340)),
    ]

    for start, end in edges:
        draw.line([start, end], fill=edge_color, width=2)

    # --- Arrow heads (simple triangles) ---
    arrows = [
        (320, 285, 320, 240),
        (320, 395, 320, 340),
        (490, 300, 420, 285),
        (490, 390, 420, 395),
    ]

    for ex, ey, sx, sy in arrows:
        draw.polygon(
            [(ex, ey), (ex - 6, ey - 8), (ex + 6, ey - 8)],
            fill=edge_color,
        )

    # --- Right side: Title & subtitle ---
    font_title = _find_font(48)
    font_subtitle = _find_font(22)
    font_tag = _find_font(16)

    title = "Certilab Adaptive RAG"
    subtitle = "Self-correcting RAG pipeline with LangGraph & OpenAI"
    tagline = "7-node graph • Query rewriting • Hallucination detection"

    # Title
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = title_bbox[2] - title_bbox[0]
    draw.text((W - tw - 80, 160), title, fill="#ffffff", font=font_title)

    # Subtitle
    sub_bbox = draw.textbbox((0, 0), subtitle, font=font_subtitle)
    sw = sub_bbox[2] - sub_bbox[0]
    draw.text((W - sw - 80, 230), subtitle, fill="#8b949e", font=font_subtitle)

    # Tagline
    tag_bbox = draw.textbbox((0, 0), tagline, font=font_tag)
    tw2 = tag_bbox[2] - tag_bbox[0]
    draw.text((W - tw2 - 80, 275), tagline, fill="#58a6ff", font=font_tag)

    # --- Bottom tech stack pills ---
    pills = ["LangGraph", "OpenAI", "Qdrant", "Python 3.11+", "Tavily"]
    pill_y = 480
    pill_h = 30
    pill_x_start = W - 80

    for pill in reversed(pills):
        pill_bbox = draw.textbbox((0, 0), pill, font=font_tag)
        pw = pill_bbox[2] - pill_bbox[0]
        pill_x = pill_x_start - pw - 24
        _draw_rounded_rect(
            draw,
            (pill_x, pill_y, pill_x + pw + 24, pill_y + pill_h),
            8,
            fill=_rgb("#21262d"),
        )
        draw.text(
            (pill_x + 12, pill_y + (pill_h - (pill_bbox[3] - pill_bbox[1])) // 2),
            pill,
            fill="#c9d1d9",
            font=font_tag,
        )
        pill_x_start = pill_x - 12

    # Ensure assets directory exists
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    img.save(output, "PNG")
    print(f"✅ Social preview saved to {output} ({W}×{H})")


if __name__ == "__main__":
    generate_social_preview()
