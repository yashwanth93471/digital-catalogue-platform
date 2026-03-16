"""Test non-white density approach to find tile within composite region."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import numpy as np
from PIL import Image
from pipeline.composite_segmenter import render_page, analyze_page_layout, crop_to_content
from pipeline.clip_classifier import CLIPClassifier

clf = CLIPClassifier()
pdf = "catalogues/18x12 Elevation.pdf"
out = "_elev_page2"

# Test on pages 2, 4, 6, 8 (product_info pages)
for pn in [1, 3, 5, 7, 9, 11]:
    info = analyze_page_layout(pdf, pn, clf)
    if not info["content_regions"]:
        continue
    region = info["content_regions"][0]["image"]
    rw, rh = region.size
    aspect = rw / rh

    gray = np.array(region.convert("L"), dtype=np.float32)
    h, w = gray.shape

    # Non-white density per column
    non_white = (gray < 235).astype(np.float32)
    col_density = np.mean(non_white, axis=0)

    # Heavy smoothing
    k = max(w // 15, 10)
    padded = np.pad(col_density, k // 2, mode='edge')
    smoothed = np.convolve(padded, np.ones(k) / k, mode='valid')[:w]

    print(f"\nPage {pn+1}: {rw}x{rh}, aspect={aspect:.2f}, role={info['role']}")
    print(f"  Col density at key positions:")
    for x in range(0, w, 100):
        print(f"    x={x}: density={smoothed[x]:.3f}")

    # Find tile start (first column where density > 0.40)
    tile_start = w
    for x in range(w):
        if smoothed[x] > 0.40:
            tile_start = x
            break

    # Find tile end
    tile_end = 0
    for x in range(w - 1, -1, -1):
        if smoothed[x] > 0.40:
            tile_end = x
            break

    print(f"  Tile bounds: x={tile_start} to x={tile_end}")

    if tile_start < w and tile_start > w * 0.15:
        tile_crop = region.crop((max(0, tile_start - 5), 0, min(w, tile_end + 5), rh))
        tile_final = crop_to_content(tile_crop)
        print(f"  Tile crop: {tile_crop.size} -> after crop_to_content: {tile_final.size}")
        tile_final.save(f"{out}/tile_p{pn+1}.png")

        # Also save the text portion
        text_crop = region.crop((0, 0, tile_start, rh))
        text_crop.save(f"{out}/text_p{pn+1}.png")
    else:
        print(f"  No composite detected (tile_start={tile_start})")
