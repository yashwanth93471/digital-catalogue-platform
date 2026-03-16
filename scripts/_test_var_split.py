"""Test column variance approach for tile isolation."""
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

# Test on product_info pages (even pages: 2,4,6 = indices 1,3,5)
for pn in [1, 3, 5, 13, 17, 21, 25, 29]:
    info = analyze_page_layout(pdf, pn, clf)
    if not info["content_regions"]:
        continue
    region = info["content_regions"][0]["image"]
    rw, rh = region.size
    aspect = rw / rh

    gray = np.array(region.convert("L"), dtype=np.float32)
    h, w = gray.shape

    # Column variance with smoothing
    col_var = np.var(gray, axis=0)
    k = max(w // 30, 5)
    padded = np.pad(col_var, k // 2, mode='edge')
    smoothed = np.convolve(padded, np.ones(k) / k, mode='valid')[:w]

    # Find minimum in [15%, 55%] range
    start = int(w * 0.15)
    end = int(w * 0.55)
    search = smoothed[start:end]
    min_idx = start + int(np.argmin(search))
    min_val = smoothed[min_idx]
    
    # Find 75th percentile of all column variance
    p75 = np.percentile(smoothed, 75)
    ratio = p75 / max(min_val, 1)

    # Find tile start: from min_idx rightward, first column where var > min_val * 3
    threshold = min_val * 3
    tile_start = min_idx
    for x in range(min_idx, w):
        if smoothed[x] > threshold:
            tile_start = x
            break

    print(f"Page {pn+1}: {rw}x{rh}, aspect={aspect:.2f}, role={info['role']}")
    print(f"  Var min at x={min_idx}: {min_val:.0f}, p75={p75:.0f}, ratio={ratio:.1f}")
    print(f"  Threshold={threshold:.0f}, tile_start={tile_start}")

    if ratio > 2.0 and tile_start > w * 0.15:
        tile_crop = region.crop((max(0, tile_start - 3), 0, w, rh))
        tile_final = crop_to_content(tile_crop)
        print(f"  Tile: ({tile_start}, 0, {w}, {rh}) -> {tile_crop.size} -> {tile_final.size}")
        tile_final.save(f"{out}/var_tile_p{pn+1}.png")
        
        # Also get CLIP score
        scores = clf.classify(tile_final)
        print(f"  CLIP: texture+preview={scores.get('texture',0)+scores.get('preview_tile',0):.3f}")
        
        # Save text portion
        text_crop = region.crop((0, 0, min_idx, rh))
        text_crop.save(f"{out}/var_text_p{pn+1}.png")
    else:
        print(f"  Not composite (ratio={ratio:.1f})")
