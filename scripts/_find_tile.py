"""Find the tile image within the Elevation content region by analyzing column texture."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import numpy as np
from PIL import Image
from pipeline.composite_segmenter import render_page, analyze_page_layout
from pipeline.clip_classifier import CLIPClassifier

clf = CLIPClassifier()
pdf = "catalogues/18x12 Elevation.pdf"
out = "_elev_page2"

info = analyze_page_layout(pdf, 1, clf)
region = info["content_regions"][0]["image"]
rw, rh = region.size
print(f"Region: {rw}x{rh}")

# Try CLIP on left/right halves
mid = rw // 2
left_half = region.crop((0, 0, mid, rh))
right_half = region.crop((mid, 0, rw, rh))

ls = clf.classify(left_half)
rs = clf.classify(right_half)
print(f"Left half CLIP:  {ls}")
print(f"Right half CLIP: {rs}")
left_half.save(f"{out}/left_half.png")
right_half.save(f"{out}/right_half.png")

# Try splitting at different x positions
gray = np.array(region.convert("L"), dtype=np.float32)
# Column variance (high for texture, low for flat/text areas)
col_var = np.var(gray, axis=0)

# Smooth the column variance
k = 50
padded = np.pad(col_var, k//2, mode='edge')
smoothed = np.convolve(padded, np.ones(k)/k, mode='valid')[:rw]

print(f"\nColumn variance (smoothed) at key positions:")
for x in range(0, rw, 100):
    print(f"  x={x}: var={smoothed[x]:.1f}")

# Find where variance jumps significantly (text→tile transition)
# Look for a big step up in the middle portion
mid_start = int(rw * 0.3)
mid_end = int(rw * 0.7)
max_jump = 0
jump_x = mid
for x in range(mid_start, mid_end):
    jump = smoothed[x] - smoothed[x - 50] if x > 50 else 0
    if jump > max_jump:
        max_jump = jump
        jump_x = x

print(f"\nBiggest variance jump at x={jump_x}, jump={max_jump:.1f}")

# Try CLIP on multiple split points
for split_pct in [0.35, 0.40, 0.45, 0.50, 0.55]:
    sx = int(rw * split_pct)
    right = region.crop((sx, 0, rw, rh))
    scores = clf.classify(right)
    tex = scores.get("texture", 0) + scores.get("preview_tile", 0)
    print(f"  Split at {split_pct:.0%} (x={sx}): texture+preview={tex:.3f}, scores={scores}")
    right.save(f"{out}/split_{int(split_pct*100)}.png")

# Also try edge density approach
edges = cv2.Canny(np.array(region.convert("L")), 50, 150)
col_edge = np.mean(edges > 0, axis=0)
padded_e = np.pad(col_edge, k//2, mode='edge')
smoothed_e = np.convolve(padded_e, np.ones(k)/k, mode='valid')[:rw]

print(f"\nEdge density by column:")
for x in range(0, rw, 100):
    print(f"  x={x}: edge_density={smoothed_e[x]:.4f}")
