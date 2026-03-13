"""Quick full pattern check for MOROCON."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from pipeline.composite_segmenter import render_page, _find_large_rects, _remove_nested_rects
from pipeline.clip_classifier import CLIPClassifier

clf = CLIPClassifier()

pdf = "catalogues/MOROCON GVT.pdf"
print("MOROCON GVT - all 30 pages at 200dpi:")
for pn in range(30):
    img = render_page(pdf, pn, dpi=200)
    w, h = img.size
    img_np = np.array(img)
    rects = _find_large_rects(img_np, min_area_ratio=0.01)
    rects = _remove_nested_rects(rects)
    
    # Count square-ish rects (possible thumbnails)
    sq = []
    full = []
    for rx, ry, rw, rh in rects:
        area = (rw * rh) / (w * h)
        aspect = rw / rh if rh > 0 else 0
        if 0.08 < area < 0.25 and 0.8 < aspect < 1.3:
            sq.append(f"{area:.0%}")
        elif area > 0.35:
            full.append(f"{area:.0%}")
    
    scores = clf.classify(img)
    top = max(scores, key=scores.get)
    
    if len(sq) >= 2:
        ptype = "THUMB"
    elif len(full) >= 1:
        ptype = "APP  "
    elif len(sq) == 1:
        ptype = "INTRO"
    else:
        ptype = "OTHER"
    
    print(f"  P{pn+1:>2} {ptype}: {len(rects)}rects, sq={sq}, full={full} [{top}]")
