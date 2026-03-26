import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def get_render_attributes(weight, min_size=12, max_size=50):
    font_size = int(min_size + (max_size - min_size) * weight)
    
    if weight > 0.8:
        color = (220, 20, 20)
    else:
        gray_val = int(200 * (1.0 - (weight / 0.8)))
        color = (gray_val, gray_val, gray_val)
    return font_size, color

def render_tsvr_image(word_weights, image_width=800, max_font_size=50):
    # Dynamically allocate image height because full contracts can be infinitely long
    est_lines = len(word_weights) // (image_width // 40) + 1 
    image_height = max(800, est_lines * (max_font_size + 15) + 500)
    img = Image.new('RGB', (image_width, image_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    font_paths = ["arial.ttf", "Arial.ttf", "/Library/Fonts/Arial.ttf"]
    font_path = None
    for fp in font_paths:
        if os.path.exists(fp) or fp == "arial.ttf":
            font_path = fp
            break

    x_cursor = 20
    y_cursor = 50
    line_height = max_font_size + 15
    
    for word, weight in word_weights:
        size, color = get_render_attributes(weight, max_size=max_font_size)
        
        try:
            font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        except IOError:
            font = ImageFont.load_default()
            
        bbox = draw.textbbox((0, 0), word, font=font)
        word_width = bbox[2] - bbox[0]
        
        if x_cursor + word_width > image_width - 20:
            x_cursor = 20
            y_cursor += line_height
            
        # VERY CRITICAL: By invoking anchor="ls" (left-ascender baseline), the coordinate passed in
        # represents exactly where the character's physical baseline should strike.
        draw.text((x_cursor, y_cursor), word, font=font, fill=color, anchor="ls")
        x_cursor += word_width + int(size * 0.4)
        
    final_height = y_cursor + 50
    img = img.crop((0, 0, image_width, final_height))
    return img

def visualize_single_attention(word_weights, output_path):
    fig = plt.figure(figsize=(14, 5))
    
    from render.utils import filter_for_barchart, group_by_stem_and_sort
    
    # We group and sort descending seamlessly deduplicating similar stems natively
    grouped_ww = group_by_stem_and_sort(word_weights)
    
    display_limit = min(60, len(grouped_ww))
    words = [w[0] for w in grouped_ww[:display_limit]]
    weights = [w[1] for w in grouped_ww[:display_limit]]
    
    colors = ['red' if w > 0.8 else ('#333333' if w > 0.4 else '#cccccc') for w in weights]
    plt.bar(range(len(words)), weights, color=colors)
    plt.xticks(range(len(words)), words, rotation=45, ha='right', fontsize=10)
    plt.title("Saliency Attention (Stem-Merged & Sorted by Max Weight exclusively for plotting)", fontsize=14, fontweight='bold')
    plt.ylabel("Attention Prob")
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)