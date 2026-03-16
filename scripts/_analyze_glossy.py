"""Analyze 800x800 Glossy Catalogue structure - all pages."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import fitz
import numpy as np
from PIL import Image, ImageDraw
from pipeline.composite_segmenter import render_page, _find_large_rects, _remove_nested_rects
from pipeline.clip_classifier import CLIPClassifier

clf = CLIPClassifier()

pdf = "catalogues/800x800 Glossy Catalogue 2022 - Copy.pdf"
doc = fitz.open(pdf)
n = len(doc)
print(f"Total pages: {n}")

os.makedirs("_debug_glossy", exist_ok=True)

for pn in range(n):
    page = doc[pn]
    text = page.get_text().strip()
    
    img = render_page(pdf, pn, dpi=150)
    w, h = img.size
    img_np = np.array(img)
    
    rects = _find_large_rects(img_np, min_area_ratio=0.01)
    rects = _remove_nested_rects(rects)
    
    scores = clf.classify(img)
    top = max(scores, key=scores.get)
    
    print(f"\nP{pn+1:>3}: {w}x{h}, {len(rects)} rects, [{top} {scores[top]:.2f}]")
    if text:
        print(f"     txt: {text[:120].replace(chr(10), ' | ')}")
    
    for i, (rx, ry, rw, rh) in enumerate(rects):
        area = (rw * rh) / (w * h)
        aspect = rw / rh if rh > 0 else 0
        crop = img.crop((rx, ry, rx+rw, ry+rh))
        s = clf.classify(crop)
        c = max(s, key=s.get)
        print(f"     r{i}: ({rx},{ry}) {rw}x{rh}  area={area:.1%}  asp={aspect:.2f}  [{c}]")
    
    # Save first 20 pages annotated
    if pn < 20:
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        for i, (rx, ry, rw, rh) in enumerate(rects):
            draw.rectangle([(rx, ry), (rx+rw, ry+rh)], outline="red", width=3)
            draw.text((rx+5, ry+5), f"R{i}", fill="red")
        draw_img.save(f"_debug_glossy/P{pn+1:02d}.jpg")

doc.close()
