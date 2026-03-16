"""Analyze all 4 new PDFs: page counts, image counts, text, and render sample pages."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import fitz
import numpy as np
from PIL import Image
from pipeline.composite_segmenter import render_page, _find_large_rects, _remove_nested_rects
from pipeline.clip_classifier import CLIPClassifier

clf = CLIPClassifier()

pdfs = {
    "DREAM_HYD": "catalogues/DREAM TILES -HYDERABAD JPG With Preview 30x30.pdf",
    "MOROCON": "catalogues/MOROCON GVT.pdf",
    "MILLANO": "catalogues/MILANO CERAMICA.pdf",
    "PARK": "catalogues/PARK COLLECTION.pdf",
}

for label, path in pdfs.items():
    print(f"\n{'='*60}")
    print(f"  {label}: {path}")
    print(f"{'='*60}")
    
    doc = fitz.open(path)
    n_pages = len(doc)
    print(f"  Pages: {n_pages}")
    
    # Check text blocks on first few pages
    for pn in range(min(5, n_pages)):
        page = doc[pn]
        text = page.get_text().strip()
        imgs = page.get_images(full=True)
        print(f"  Page {pn+1}: {len(imgs)} images, text={repr(text[:80]) if text else 'none'}")
    
    out = f"_analysis_{label}"
    os.makedirs(out, exist_ok=True)
    
    # Render and analyze sample pages
    sample_pages = list(range(min(8, n_pages)))  # First 8 pages
    for pn in sample_pages:
        img = render_page(path, pn, dpi=100)
        img.save(f"{out}/p{pn+1}.png")
        w, h = img.size
        
        # Find rectangles
        img_np = np.array(img)
        rects = _find_large_rects(img_np, min_area_ratio=0.03)
        rects = _remove_nested_rects(rects)
        
        # CLIP on full page
        scores = clf.classify(img)
        top_cat = max(scores, key=scores.get)
        
        print(f"  Page {pn+1}: {w}x{h}, {len(rects)} rects, CLIP={top_cat}({scores[top_cat]:.2f})")
        
        # CLIP on each rect
        for i, (rx, ry, rw, rh) in enumerate(rects):
            crop = img.crop((rx, ry, rx+rw, ry+rh))
            s = clf.classify(crop)
            cat = max(s, key=s.get)
            area_pct = (rw * rh) / (w * h)
            print(f"    Rect {i}: ({rx},{ry},{rw},{rh}) area={area_pct:.1%} -> {cat}({s[cat]:.2f})")
            crop.save(f"{out}/p{pn+1}_r{i}_{cat}.png")
    
    doc.close()
    print()
