import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "Arial.ttf", "arial.ttf",
]

def _find_font():
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            return fp
    return None

_FONT_CACHE = {}
def _get_font(path, size):
    key = (path, int(size))
    f = _FONT_CACHE.get(key)
    if f is not None:
        return f
    try:
        f = ImageFont.truetype(path, int(size)) if path else ImageFont.load_default()
    except (IOError, OSError):
        f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f

# Dark, OCR-friendly topic palette (strong contrast on white).
TOPIC_COLORS = [
    (170, 25, 25),    # dark red
    (25, 55, 160),    # dark blue
    (30, 110, 45),    # dark green
    (120, 35, 140),   # dark purple
    (170, 90, 10),    # dark orange
    (25, 110, 120),   # dark teal
    (140, 35, 100),   # dark magenta
    (90, 85, 20),     # dark olive
]

def assign_topic_ids(phis, threshold=0.55, max_gap=4):
    """
    Cluster high-saliency tokens into topics by positional proximity. Salient
    runs (phi >= threshold) separated by more than `max_gap` low-saliency
    tokens become distinct topics with distinct colors. Non-salient tokens
    receive topic_id == -1.
    """
    n = len(phis)
    topic_ids = [-1] * n
    current = -1
    last_hit = -10**9
    for i, p in enumerate(phis):
        if float(p) >= threshold:
            if i - last_hit > max_gap:
                current += 1
            topic_ids[i] = current
            last_hit = i
    return topic_ids

def _weight_to_color(phi, topic_id=-1):
    # phi in [0,1]. Topic tokens get a dark palette colour modulated by phi;
    # non-topic tokens stay dark grey (never washed out, so OCR still reads).
    phi = float(np.clip(phi, 0.0, 1.0))
    if topic_id >= 0:
        base = TOPIC_COLORS[topic_id % len(TOPIC_COLORS)]
        intensity = 0.75 + 0.25 * phi
        return tuple(max(0, min(255, int(c * intensity))) for c in base)
    darkness = int(130 - 70 * phi)
    darkness = max(45, min(150, darkness))
    return (darkness, darkness, darkness)

def _render_word_layout(word_weights, sizes, colors, question_text, image_width, s_max, margin):
    if (not isinstance(image_width, (int, np.integer)) or
            not isinstance(margin, (int, np.integer)) or margin < 0 or image_width <= 2 * margin):
        raise ValueError("image_width and margin must be integers with margin >= 0 and image_width > 2 * margin")
    if not word_weights:
        return Image.new("RGB", (image_width, 200), color=(255, 255, 255))
    usable_width = image_width - 2 * margin
    font_path = _find_font()
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    question_font = _get_font(font_path, 18) if question_text else None
    if question_text:
        question_bounds = measure.textbbox((0, 0), f"Q: {question_text}", font=question_font)
        if question_bounds[2] - question_bounds[0] > usable_width:
            raise ValueError("Question header exceeds the available width; increase image_width or shorten question_text")
    line_height = s_max + 10
    baseline = margin + s_max + (56 if question_text else 0)
    x = margin
    placements = []
    bottom = baseline
    for (word, _), size, color in zip(word_weights, sizes, colors):
        font = _get_font(font_path, int(size))
        bbox = measure.textbbox((0, 0), word, font=font)
        width = bbox[2] - bbox[0]
        if width > usable_width:
            raise ValueError("A word exceeds the available width; increase image_width or reduce font sizes")
        if x + width > image_width - margin:
            x = margin
            baseline += line_height
        position = (x, baseline)
        bounds = measure.textbbox(position, word, font=font, anchor="ls")
        bottom = max(bottom, bounds[3])
        placements.append((word, font, color, position))
        x += width + max(4, int(size * 0.25))

    height = max(baseline + 30, bottom + margin)
    image = Image.new("RGB", (image_width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    if question_text:
        draw.text((margin, margin), f"Q: {question_text}", font=question_font, fill=(40, 40, 120))
        draw.line([(margin, margin + 40), (image_width - margin, margin + 40)], fill=(200, 200, 200), width=1)
    for word, font, color, position in placements:
        draw.text(position, word, font=font, fill=color, anchor="ls")
    return image


def render_img(word_weights, question_text=None, image_width=900,
               s_min=18, s_max=44, beta=4.5, mu_pct=60.0, margin=24,
               topic_threshold=0.55, topic_max_gap=4):
    if not word_weights:
        return _render_word_layout([], [], [], None, image_width, s_max, margin)
    weights = np.array([weight for _, weight in word_weights], dtype=np.float32)
    mu = float(np.percentile(weights, mu_pct))
    phi = 1.0 / (1.0 + np.exp(-beta * (weights - mu)))
    sizes = (s_min + (s_max - s_min) * np.power(phi, 0.55)).astype(int)
    topic_ids = assign_topic_ids(phi, threshold=topic_threshold, max_gap=topic_max_gap)
    colors = [_weight_to_color(float(weight), topic_id=topic) for weight, topic in zip(phi, topic_ids)]
    return _render_word_layout(word_weights, sizes, colors, question_text, image_width, s_max, margin)


def render_img_tiered(word_weights, tiers, question_text=None,
                      image_width=900, s_min=16, s_mid=26, s_max=46, margin=24,
                      topic_max_gap=4):
    if not word_weights:
        return _render_word_layout([], [], [], None, image_width, s_max, margin)
    weights = np.array([weight for _, weight in word_weights], dtype=np.float32)
    tiers = np.asarray(tiers)
    if tiers.shape != weights.shape or not np.isin(tiers, (0, 1, 2)).all():
        raise ValueError("tiers must contain one class ID from 0 to 2 per word")
    tiers = tiers.astype(np.int8)
    secondary = tiers == 1
    if secondary.any():
        secondary_weights = weights[secondary]
        lo, hi = float(secondary_weights.min()), float(secondary_weights.max())
        secondary_normalized = (secondary_weights - lo) / (hi - lo + 1e-8)
    secondary_index = 0
    topic_ids = assign_topic_ids((tiers == 2).astype(np.float32), threshold=0.5, max_gap=topic_max_gap)
    sizes = np.empty(len(weights), dtype=np.int32)
    colors = []
    for index, tier in enumerate(tiers):
        if tier == 2:
            sizes[index] = s_max
            topic = topic_ids[index]
            colors.append(TOPIC_COLORS[topic % len(TOPIC_COLORS)] if topic >= 0 else (170, 25, 25))
        elif tier == 1:
            weight = float(secondary_normalized[secondary_index])
            secondary_index += 1
            sizes[index] = int(s_mid + (s_max - 2 - s_mid) * (weight ** 0.55))
            grey = int(75 - 30 * weight)
            colors.append((grey, grey, grey))
        else:
            sizes[index] = s_min
            colors.append((120, 120, 120))
    return _render_word_layout(word_weights, sizes, colors, question_text, image_width, s_max, margin)


# Legacy alias + bar chart helpers kept for backward compatibility
def get_render_attributes(weight, min_size=12, max_size=50):
    font_size = int(min_size + (max_size - min_size) * weight)
    if weight > 0.8:
        color = (220, 20, 20)
    else:
        gray_val = int(200 * (1.0 - (weight / 0.8)))
        color = (gray_val, gray_val, gray_val)
    return font_size, color

def render_tsvr_image(word_weights, image_width=800, max_font_size=50):
    return render_img(word_weights, image_width=image_width, s_max=max_font_size)


def render_tsvr_page(word_weights, n_words_budget, image_width=1024,
                     page_aspect=1.414, margin=24, s_min=14, s_max=40,
                     topic_threshold=0.55, topic_max_gap=4):
    """Saliency-budgeted single-page render for OCR compression.

    Selects the top-`n_words_budget` words by weight (preserving original
    reading order), then lays them out on a fixed-aspect canvas of
    `image_width x image_width*page_aspect`. Font size is auto-shrunk so
    all kept words fit — guaranteeing the output is a page-shaped image
    DeepSeek-OCR can actually read (vs. the 1:100 scrolls produced by
    `render_tsvr_image` on long contracts, which get crushed during the
    model's internal resize to its patch grid).

    The kept words are still coloured by topic/saliency exactly like
    `render_img`, so DeepSeek-OCR sees the same visual salience cues.
    """
    if not word_weights:
        return Image.new('RGB', (image_width, int(image_width * page_aspect)),
                         color=(255, 255, 255))

    n_total = len(word_weights)
    B = max(1, min(n_words_budget, n_total))

    weights = np.array([w for _, w in word_weights], dtype=np.float32)
    if B < n_total:
        # Keep top-B by weight, preserving reading order.
        kept_idx = np.argpartition(-weights, B - 1)[:B]
        kept_idx = np.sort(kept_idx)
    else:
        kept_idx = np.arange(n_total)
    kept = [word_weights[i] for i in kept_idx]
    kept_weights = weights[kept_idx]

    # Saliency colour/topic computed on the KEPT subset (high-contrast page).
    mu = float(np.percentile(kept_weights, 60.0)) if len(kept_weights) else 0.0
    phi = 1.0 / (1.0 + np.exp(-4.5 * (kept_weights - mu)))
    topic_ids = assign_topic_ids(phi, threshold=topic_threshold, max_gap=topic_max_gap)

    canvas_h = int(image_width * page_aspect)
    usable_w = image_width - 2 * margin
    usable_h = canvas_h - 2 * margin

    font_path = _find_font()

    # Pick the largest font that still fits all B words on the page.
    # Two-pass: measure at probe size then scale by sqrt(area_ratio).
    def _layout_heights(font_size):
        """Dry-run layout; return total height used. Returns None if any word
        wider than usable_w at this font (forces a shrink)."""
        font = _get_font(font_path, font_size)
        space_w = font_size * 0.30
        line_h = int(font_size * 1.30)
        x = 0.0
        y = 0
        for (word, _) in kept:
            bbox = font.getbbox(word)
            w = bbox[2] - bbox[0]
            if w > usable_w and font_size > s_min:
                return None
            if x + w > usable_w:
                x = 0.0
                y += line_h
            x += w + space_w
        return y + line_h

    lo, hi = s_min, s_max
    best = s_min
    while lo <= hi:
        mid = (lo + hi) // 2
        h = _layout_heights(mid)
        if h is not None and h <= usable_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    font_size = best

    img = Image.new('RGB', (image_width, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font(font_path, font_size)
    space_w = int(font_size * 0.30)
    line_h = int(font_size * 1.30)

    x = margin
    baseline = margin + font_size
    for (word, _), p, tid in zip(kept, phi, topic_ids):
        bbox = draw.textbbox((0, 0), word, font=font)
        w = bbox[2] - bbox[0]
        if x + w > image_width - margin:
            x = margin
            baseline += line_h
            if baseline + 4 > canvas_h - margin:
                break  # ran out of page (shouldn't happen after binary search)
        color = _weight_to_color(float(p), topic_id=tid)
        draw.text((x, baseline), word, font=font, fill=color, anchor='ls')
        x += w + space_w
    return img

def visualize_single_attention(word_weights, output_path):
    fig = plt.figure(figsize=(14, 5))
    from render.utils import group_by_stem_and_sort
    grouped = group_by_stem_and_sort(word_weights)
    display_limit = min(60, len(grouped))
    words = [w[0] for w in grouped[:display_limit]]
    weights = [w[1] for w in grouped[:display_limit]]

    # Reuse the topic palette so bar chart and distortion image share identity.
    full_weights = np.array([w for _, w in word_weights], dtype=np.float32)
    topic_ids_full = assign_topic_ids(full_weights, threshold=0.55)
    word_to_topic = {}
    for (w, _), tid in zip(word_weights, topic_ids_full):
        if tid >= 0 and w not in word_to_topic:
            word_to_topic[w] = tid

    def bar_color(word, weight):
        tid = word_to_topic.get(word, -1)
        if tid >= 0:
            r, g, b = TOPIC_COLORS[tid % len(TOPIC_COLORS)]
            return (r / 255.0, g / 255.0, b / 255.0)
        gray = max(0.25, 0.65 - 0.35 * weight)
        return (gray, gray, gray)

    colors = [bar_color(w, wt) for w, wt in zip(words, weights)]
    plt.bar(range(len(words)), weights, color=colors)
    plt.xticks(range(len(words)), words, rotation=45, ha='right', fontsize=10)
    plt.title("Saliency Attention", fontsize=14, fontweight='bold')
    plt.ylabel("Attention Prob")
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
