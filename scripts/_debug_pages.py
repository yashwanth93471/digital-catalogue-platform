"""Debug MOROCON and MILLANO page structure in detail."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import fitz
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pipeline.composite_segmenter import render_page, _find_large_rects, _remove_nested_rects
from pipeline.clip_classifier import CLIPClassifier

clf = CLIPClassifier()
os.makedirs("_debug_pages", exist_ok=True)

def debug_pdf(label, path, page_range, out_prefix):
    doc = fitz.open(path)
    print(f"\n{'='*70}")
    print(f"  {label} - pages {page_range[0]+1} to {page_range[-1]+1}")
    print(f"{'='*70}")
    
    for pn in page_range:
        if pn >= len(doc):
            break
        page = doc[pn]
        text = page.get_text().strip()
        imgs = page.get_images(full=True)
        
        # Render at 200dpi
        img = render_page(path, pn, dpi=200)
        w, h = img.size
        img_np = np.array(img)
        
        # Find rects with VERY relaxed thresholds
        rects = _find_large_rects(img_np, min_area_ratio=0.01)
        rects = _remove_nested_rects(rects)
        
        # Classify full page
        scores = clf.classify(img)
        top = max(scores, key=scores.get)
        
        print(f"\n  P{pn+1}: size={w}x{h}, {len(imgs)} embedded images, {len(rects)} rects, [{top}]")
        if text:
            print(f"    text: {text[:100].replace(chr(10), ' ')}")
        else:
            print(f"    text: (none)")
        
        for i, (rx, ry, rw, rh) in enumerate(rects):
            area = (rw * rh) / (w * h)
            aspect = rw / rh if rh > 0 else 0
            crop = img.crop((rx, ry, rx+rw, ry+rh))
            s = clf.classify(crop)
            c = max(s, key=s.get)
            print(f"    rect{i}: ({rx},{ry}) {rw}x{rh}  area={area:.1%}  aspect={aspect:.2f}  [{c}]")
        
        # Save annotated page
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        for i, (rx, ry, rw, rh) in enumerate(rects):
            draw.rectangle([(rx, ry), (rx+rw, ry+rh)], outline="red", width=3)
            draw.text((rx+5, ry+5), f"R{i}", fill="red")
        draw_img.save(f"_debug_pages/{out_prefix}_P{pn+1}.jpg")
    
    doc.close()

# MOROCON: check P1-P8
debug_pdf("MOROCON GVT", "catalogues/MOROCON GVT.pdf", range(0, 8), "morocon")

# MILLANO: check P5-P18
debug_pdf("MILLANO CERAMICA", "catalogues/MILANO CERAMICA.pdf", range(4, 18), "millano")
