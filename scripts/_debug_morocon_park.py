"""Debug Morocon page pairing and Park names."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pytesseract as pyt
from PIL import Image
from pipeline.composite_segmenter import render_page, _find_large_rects, _remove_nested_rects
from pipeline.clip_classifier import CLIPClassifier
from pipeline.name_extractor import _ocr_with_variants, _best_ocr_line

pyt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
clf = CLIPClassifier()

# ── MOROCON: check which pages have thumbnails vs applications ──
print("MOROCON GVT - Page classification:")
pdf = "catalogues/MOROCON GVT.pdf"
for pn in range(30):
    img = render_page(pdf, pn, dpi=100)
    w, h = img.size
    img_np = np.array(img)
    rects = _find_large_rects(img_np, min_area_ratio=0.03)
    rects = _remove_nested_rects(rects)
    
    # Count square-ish rects in middle
    sq_rects = []
    for rx, ry, rw, rh in rects:
        area = (rw * rh) / (w * h)
        aspect = rw / rh if rh > 0 else 0
        if 0.05 < area < 0.30 and 0.7 < aspect < 1.4 and ry > h * 0.10:
            sq_rects.append((rx, ry, rw, rh))
    
    scores = clf.classify(img)
    top = max(scores, key=scores.get)
    
    if len(sq_rects) >= 2:
        ptype = "THUMB_PAGE"
    elif len(sq_rects) == 1:
        ptype = "THUMB_1"
    elif scores.get("room_scene", 0) > 0.25:
        ptype = "APP"
    else:
        ptype = f"OTHER({top})"
    
    print(f"  P{pn+1:>2}: {len(rects)} rects, {len(sq_rects)} sq -> {ptype}")

# ── PARK: render pages and check for text ──
print("\n\nPARK COLLECTION - OCR per page:")
pdf2 = "catalogues/PARK COLLECTION.pdf"
out = "_park_debug"
os.makedirs(out, exist_ok=True)
for pn in range(24):
    img = render_page(pdf2, pn, dpi=150)
    w, h = img.size
    
    # OCR bottom 20%
    bottom = img.crop((0, int(h*0.80), w, h))
    text = pyt.image_to_string(bottom).strip()
    
    # OCR top 20%
    top_area = img.crop((0, 0, w, int(h*0.20)))
    text_top = pyt.image_to_string(top_area).strip()
    
    # Full OCR
    full = pyt.image_to_string(img).strip()
    
    scores = clf.classify(img)
    top_cat = max(scores, key=scores.get)
    
    print(f"  P{pn+1:>2} [{top_cat}]:")
    if text:
        print(f"    bottom: {text[:80]}")
    if text_top:
        print(f"    top:    {text_top[:80]}")
    # Save to check
    img.save(f"{out}/p{pn+1}.png")
