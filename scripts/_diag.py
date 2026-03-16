"""Quick diagnostic: test CLIP scoring on horizontal splits for flat pages."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.clip_classifier import CLIPClassifier
from pipeline.composite_segmenter import render_page, _segment_by_layout, _find_large_rects
import numpy as np

clf = CLIPClassifier()

pdfs = [
    ("catalogues/16x16 PARKING.pdf", [2, 5, 10, 20]),
    ("catalogues/18x12 Elevation.pdf", [2, 5, 10, 20]),
    ("catalogues/4X2 Hight Glosy.pdf", [2, 4]),
]

for pdf_path, pages in pdfs:
    name = os.path.basename(pdf_path)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    for pn in pages:
        print(f"\n  Page {pn}:")
        img = render_page(pdf_path, pn, dpi=200)
        
        # Full image classification
        scores = clf.classify(img)
        cat, conf = clf.get_category(img)
        print(f"    Full image: {cat} ({conf:.2f})  tex={scores['texture']:.2f} scene={scores['room_scene']:.2f} prev={scores['preview_tile']:.2f}")
        
        # Test horizontal splits at 35%, 40%, 45%, 50%
        w, h = img.size
        for frac in [0.35, 0.40, 0.45, 0.50]:
            split_y = int(h * frac)
            top = img.crop((0, 0, w, split_y))
            bot = img.crop((0, split_y, w, h))
            
            ts = clf.classify(top)
            bs = clf.classify(bot)
            
            t_cat = max(ts, key=ts.get)
            b_cat = max(bs, key=bs.get)
            
            # Combined scores
            score_tex_scene = ts.get("texture", 0) + bs.get("room_scene", 0)
            score_scene_tex = ts.get("room_scene", 0) + bs.get("texture", 0)
            # With preview_tile as texture proxy
            score_prev_scene = (ts.get("texture", 0) + ts.get("preview_tile", 0)) + bs.get("room_scene", 0)
            
            best = max(score_tex_scene, score_scene_tex, score_prev_scene)
            
            print(f"    Split {frac:.0%}: top={t_cat}({ts[t_cat]:.2f}) bot={b_cat}({bs[b_cat]:.2f})  "
                  f"T+S={score_tex_scene:.2f} S+T={score_scene_tex:.2f} P+S={score_prev_scene:.2f} best={best:.2f}")
        
        # Also test contour detection
        img_np = np.array(img)
        rects = _find_large_rects(img_np, min_area_ratio=0.05)
        print(f"    Contours found: {len(rects)} (min_area=5%)")
        for i, (x, y, rw, rh) in enumerate(rects[:4]):
            area_pct = (rw * rh) / (w * h) * 100
            crop = img.crop((x, y, x+rw, y+rh))
            cs = clf.classify(crop)
            c_cat = max(cs, key=cs.get)
            print(f"      rect{i}: ({x},{y}) {rw}x{rh} ({area_pct:.1f}%) -> {c_cat}({cs[c_cat]:.2f}) tex={cs['texture']:.2f} scene={cs['room_scene']:.2f}")

print("\nDone.")
