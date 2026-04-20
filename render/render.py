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

def render_img(word_weights, question_text=None, image_width=900,
               s_min=18, s_max=44, beta=4.5, mu_pct=60.0, margin=24,
               topic_threshold=0.55, topic_max_gap=4):
    if not word_weights:
        return Image.new('RGB', (image_width, 200), color=(255, 255, 255))

    weights = np.array([w for _, w in word_weights], dtype=np.float32)
    # Use a high percentile (default 60th) instead of median: legal docs are
    # heavy-tailed, so median centring pushes the majority into the low bucket.
    mu = float(np.percentile(weights, mu_pct))
    phi = 1.0 / (1.0 + np.exp(-beta * (weights - mu)))
    # Power curve (<1) lifts low-phi tokens so the small->large transition is
    # gradual rather than stepped — easier on the eye and on OCR.
    size_scale = np.power(phi, 0.55)
    sizes = (s_min + (s_max - s_min) * size_scale).astype(int)

    topic_ids = assign_topic_ids(phi, threshold=topic_threshold, max_gap=topic_max_gap)

    font_path = _find_font()
    line_height = s_max + 10

    # Provisional tall canvas; crop at the end
    est_lines = len(word_weights) // max(1, image_width // 40) + 4
    canvas_h = max(600, (est_lines + 2) * line_height + 200)
    img = Image.new('RGB', (image_width, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = margin
    if question_text:
        qf = _get_font(font_path, 18)
        draw.text((margin, y), f"Q: {question_text}", font=qf, fill=(40, 40, 120))
        y += 40
        draw.line([(margin, y), (image_width - margin, y)], fill=(200, 200, 200), width=1)
        y += 16

    # Baseline-aligned layout: every word on the current line sits on the same baseline
    x = margin
    baseline = y + s_max  # baseline y-coord for first content line
    for idx, ((word, _), size, p) in enumerate(zip(word_weights, sizes, phi)):
        font = _get_font(font_path, size)
        bbox = draw.textbbox((0, 0), word, font=font)
        w_width = bbox[2] - bbox[0]
        if x + w_width > image_width - margin:
            x = margin
            baseline += line_height
        color = _weight_to_color(float(p), topic_id=topic_ids[idx])
        draw.text((x, baseline), word, font=font, fill=color, anchor="ls")
        x += w_width + max(4, int(size * 0.25))

    final_h = baseline + 30
    return img.crop((0, 0, image_width, min(final_h, canvas_h)))

def render_img_tiered(word_weights, tiers, question_text=None,
                      image_width=900, s_min=16, s_mid=26, s_max=46, margin=24,
                      topic_max_gap=4):
    """
    Three-tier rendering driven by composite saliency phi and tier ids
    (2=primary, 1=secondary, 0=tertiary).

    Visual mapping:
        tier 2 -> s_max font, dark red
        tier 1 -> interpolated [s_mid, s_max-2] font, dark grey scaling with phi
        tier 0 -> s_min font, light grey
    """
    if not word_weights:
        return Image.new("RGB", (image_width, 200), color=(255, 255, 255))

    weights = np.array([w for _, w in word_weights], dtype=np.float32)
    tiers = np.asarray(tiers, dtype=np.int8)

    # Normalise within secondary tier for smooth interpolation
    sec_mask = tiers == 1
    if sec_mask.any():
        sec_w = weights[sec_mask]
        lo, hi = float(sec_w.min()), float(sec_w.max())
        sec_norm = (sec_w - lo) / (hi - lo + 1e-8)
    sec_idx = 0

    # Cluster primary-tier tokens into topics by positional proximity so each
    # distinct salient region gets a distinct dark palette colour.
    primary_mask = (tiers == 2).astype(np.float32)
    topic_ids = assign_topic_ids(primary_mask, threshold=0.5, max_gap=topic_max_gap)

    sizes = np.empty(len(weights), dtype=np.int32)
    colors = [None] * len(weights)
    for i, (_, w) in enumerate(word_weights):
        t = int(tiers[i])
        if t == 2:
            sizes[i] = s_max
            tid = topic_ids[i]
            base = TOPIC_COLORS[tid % len(TOPIC_COLORS)] if tid >= 0 else (170, 25, 25)
            colors[i] = base                  # primary: topic colour
        elif t == 1:
            phi = float(sec_norm[sec_idx]); sec_idx += 1
            # Smooth power-curve size ramp so the jump to primary is gradual.
            sizes[i] = int(s_mid + (s_max - 2 - s_mid) * (phi ** 0.55))
            grey = int(75 - 30 * phi)
            colors[i] = (grey, grey, grey)    # secondary: dark grey
        else:
            sizes[i] = s_min
            colors[i] = (120, 120, 120)       # tertiary: mid-grey (OCR-safe)

    font_path = _find_font()
    line_height = s_max + 10

    est_lines = len(word_weights) // max(1, image_width // 40) + 4
    canvas_h = max(600, (est_lines + 2) * line_height + 200)
    img = Image.new("RGB", (image_width, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = margin
    if question_text:
        qf = _get_font(font_path, 18)
        draw.text((margin, y), f"Q: {question_text}", font=qf, fill=(40, 40, 120))
        y += 40
        draw.line([(margin, y), (image_width - margin, y)], fill=(200, 200, 200), width=1)
        y += 16

    x = margin
    baseline = y + s_max
    for (word, _), size, color in zip(word_weights, sizes, colors):
        font = _get_font(font_path, int(size))
        bbox = draw.textbbox((0, 0), word, font=font)
        w_width = bbox[2] - bbox[0]
        if x + w_width > image_width - margin:
            x = margin
            baseline += line_height
        draw.text((x, baseline), word, font=font, fill=color, anchor="ls")
        x += w_width + max(4, int(size * 0.25))

    final_h = baseline + 30
    return img.crop((0, 0, image_width, min(final_h, canvas_h)))


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
