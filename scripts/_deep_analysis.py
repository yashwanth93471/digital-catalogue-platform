"""Deep analysis of each new PDF - all pages."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import fitz
import numpy as np
from PIL import Image
from pipeline.composite_segmenter import render_page, _find_large_rects, _remove_nested_rects
from pipeline.clip_classifier import CLIPClassifier

clf = CLIPClassifier()

def analyze_all_pages(label, path):
    doc = fitz.open(path)
    n = len(doc)
    print(f"\n{'='*70}")
    print(f"  {label} ({n} pages)")
    print(f"{'='*70}")
    
    for pn in range(n):
        page = doc[pn]
        imgs = page.get_images(full=True)
        text = page.get_text().strip()
        
        img = render_page(path, pn, dpi=100)
        w, h = img.size
        img_np = np.array(img)
        rects = _find_large_rects(img_np, min_area_ratio=0.03)
        rects = _remove_nested_rects(rects)
        
        scores = clf.classify(img)
        top = max(scores, key=scores.get)
        
        rect_info = []
        for rx, ry, rw, rh in rects:
            crop = img.crop((rx, ry, rx+rw, ry+rh))
            s = clf.classify(crop)
            c = max(s, key=s.get)
            area = (rw*rh)/(w*h)
            rect_info.append(f"{c}({area:.0%}@{rx},{ry},{rw}x{rh})")
        
        text_short = text[:50].replace('\n', ' ') if text else '-'
        rinfo = " | ".join(rect_info) if rect_info else "none"
        print(f"  P{pn+1:>2}: {len(rects)}rects [{top}] -> {rinfo}")
        if text_short != '-':
            print(f"       txt: {text_short}")
    
    doc.close()

analyze_all_pages("DREAM HYDERABAD", "catalogues/DREAM TILES -HYDERABAD JPG With Preview 30x30.pdf")
analyze_all_pages("MOROCON GVT", "catalogues/MOROCON GVT.pdf")
analyze_all_pages("MILLANO CERAMICA", "catalogues/MILANO CERAMICA.pdf")
analyze_all_pages("PARK COLLECTION", "catalogues/PARK COLLECTION.pdf")
