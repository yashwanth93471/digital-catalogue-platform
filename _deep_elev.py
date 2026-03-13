"""Deep analysis of the single content region on Elevation page 2.
Goal: Find the tile image within the region (right portion)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import numpy as np
from PIL import Image
from pipeline.composite_segmenter import render_page, analyze_page_layout, _find_large_rects, _remove_nested_rects
from pipeline.clip_classifier import CLIPClassifier

clf = CLIPClassifier()
pdf = "catalogues/18x12 Elevation.pdf"
out = "_elev_page2"
os.makedirs(out, exist_ok=True)

info = analyze_page_layout(pdf, 1, clf)
content_img = info["content_regions"][0]["image"]
cw, ch = content_img.size
print(f"Content region size: {cw}x{ch}")

# Try finding sub-rectangles within the content region with lower min_area threshold
img_np = np.array(content_img)
for min_area in [0.10, 0.05, 0.03, 0.02]:
    rects = _find_large_rects(img_np, min_area_ratio=min_area)
    rects = _remove_nested_rects(rects)
    print(f"\n  min_area={min_area}: {len(rects)} rects")
    for i, (rx, ry, rw, rh) in enumerate(rects):
        area_pct = (rw * rh) / (cw * ch)
        print(f"    Rect {i}: x={rx}, y={ry}, w={rw}, h={rh}, area={area_pct:.1%}")
        crop = content_img.crop((rx, ry, rx + rw, ry + rh))
        crop.save(f"{out}/subrect_area{min_area}_r{i}.png")

# Also try on the full page image
page_img = info["page_img"]
pw, ph = page_img.size
print(f"\nFull page: {pw}x{ph}")
page_np = np.array(page_img)
for min_area in [0.05, 0.03, 0.02]:
    rects = _find_large_rects(page_np, min_area_ratio=min_area)
    rects = _remove_nested_rects(rects)
    print(f"\n  Page min_area={min_area}: {len(rects)} rects")
    for i, (rx, ry, rw, rh) in enumerate(rects):
        area_pct = (rw * rh) / (pw * ph)
        print(f"    Rect {i}: x={rx}, y={ry}, w={rw}, h={rh}, area={area_pct:.1%}")
        crop = page_img.crop((rx, ry, rx + rw, ry + rh))
        scores = clf.classify(crop)
        print(f"      CLIP: {scores}")
        crop.save(f"{out}/page_rect_area{min_area}_r{i}.png")
