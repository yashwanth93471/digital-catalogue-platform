"""Test CLIP-guided split on ALL Elevation product_info pages."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from PIL import Image
from pipeline.composite_segmenter import render_page, analyze_page_layout, crop_to_content
from pipeline.clip_classifier import CLIPClassifier

clf = CLIPClassifier()
pdf = "catalogues/18x12 Elevation.pdf"
out = "_elev_page2"

# All even pages (product_info pages: indices 1,3,5,...,29)
for pn in range(1, 31, 2):
    info = analyze_page_layout(pdf, pn, clf)
    if info["role"] != "product_info" or not info["content_regions"]:
        continue
    region = info["content_regions"][0]["image"]
    rw, rh = region.size
    aspect = rw / rh

    if aspect < 1.35 or info["content_regions"][0]["area_pct"] < 0.80:
        print(f"Page {pn+1}: SKIP (aspect={aspect:.2f}, area={info['content_regions'][0]['area_pct']:.2f})")
        continue

    # Test CLIP on different right-portion splits
    full_scores = clf.classify(region)
    full_tile = full_scores.get("texture", 0) + full_scores.get("preview_tile", 0)

    results = []
    for pct in [0.30, 0.35, 0.40, 0.45, 0.50]:
        x = int(rw * pct)
        right = region.crop((x, 0, rw, rh))
        scores = clf.classify(right)
        tile_score = scores.get("texture", 0) + scores.get("preview_tile", 0)
        results.append((pct, x, tile_score))

    best = max(results, key=lambda r: r[2])
    print(f"Page {pn+1}: full_tile={full_tile:.3f}, best_split={best[0]:.0%} (x={best[1]}), score={best[2]:.3f}")
    
    for pct, x, score in results:
        marker = " <-- BEST" if pct == best[0] else ""
        print(f"    {pct:.0%}: {score:.3f}{marker}")
