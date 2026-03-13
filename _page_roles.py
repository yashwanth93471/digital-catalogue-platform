"""Deep analysis: classify every page in each flat-page PDF.

For each page:
  - Render at 200 DPI
  - CLIP classify full image
  - Find contour regions, classify each
  - Determine page role: product_info / application / cover / mixed
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image
from pipeline.clip_classifier import CLIPClassifier
from pipeline.composite_segmenter import render_page, _find_large_rects, _edge_density

clf = CLIPClassifier()

pdfs = [
    ("catalogues/4X2 Hight Glosy.pdf", 6),
    ("catalogues/18x12 Elevation.pdf", 32),
    ("catalogues/16x16 PARKING.pdf", 31),
]

for pdf_path, n_pages in pdfs:
    name = os.path.basename(pdf_path)
    print(f"\n{'='*70}")
    print(f"  {name}  ({n_pages} pages)")
    print(f"{'='*70}")

    for pn in range(1, n_pages + 1):
        img = render_page(pdf_path, pn, dpi=150)  # lower DPI for speed
        w, h = img.size
        scores = clf.classify(img)
        cat = max(scores, key=scores.get)

        # Find contour regions
        img_np = np.array(img)
        rects = _find_large_rects(img_np, min_area_ratio=0.04)

        # Classify regions
        region_info = []
        for i, (x, y, rw, rh) in enumerate(rects[:5]):
            area_pct = (rw * rh) / (w * h) * 100
            crop = img.crop((x, y, x + rw, y + rh))
            rs = clf.classify(crop)
            rc = max(rs, key=rs.get)
            ed = _edge_density(crop)
            region_info.append({
                "i": i, "x": x, "y": y, "w": rw, "h": rh,
                "area_pct": area_pct, "cat": rc, "conf": rs[rc],
                "tex": rs["texture"], "scene": rs["room_scene"],
                "prev": rs["preview_tile"], "edge_d": ed,
            })

        # Determine page role
        scene_score = scores.get("room_scene", 0)
        tex_score = scores.get("texture", 0)
        prev_score = scores.get("preview_tile", 0)
        logo_score = scores.get("logo", 0)

        # Check for room scene regions
        has_scene_region = any(r["scene"] > 0.3 for r in region_info)
        has_tex_region = any(r["tex"] > 0.3 for r in region_info)

        if scene_score > 0.4 or has_scene_region:
            role = "APPLICATION"
        elif logo_score > 0.5:
            role = "COVER"
        elif tex_score > 0.3 or prev_score > 0.3 or has_tex_region:
            role = "PRODUCT_INFO"
        else:
            role = "UNKNOWN"

        # Count large vs small regions (variant detection)
        if len(region_info) >= 2:
            areas = sorted([r["area_pct"] for r in region_info], reverse=True)
            largest = areas[0]
            others = areas[1:]
            # If largest is >2x the next, it's the main; others are variants
            if others and largest > others[0] * 1.8:
                variant_note = f"MAIN({largest:.0f}%) + {len(others)} variant(s)"
            elif len(areas) >= 2 and abs(areas[0] - areas[1]) < 5:
                variant_note = f"{len(areas)} similar-sized regions"
            else:
                variant_note = f"{len(areas)} regions"
        elif len(region_info) == 1:
            variant_note = f"1 region ({region_info[0]['area_pct']:.0f}%)"
        else:
            variant_note = "no clear regions"

        print(f"  P{pn:>2d}  {role:<14s}  full={cat}({scores[cat]:.2f})  "
              f"tex={tex_score:.2f} scene={scene_score:.2f} prev={prev_score:.2f}  "
              f"rects={len(rects)}  {variant_note}")

        # Print region details for pages with interesting structure
        if len(region_info) >= 2:
            for r in region_info[:4]:
                print(f"        rect{r['i']}: ({r['x']},{r['y']}) {r['w']}x{r['h']} "
                      f"({r['area_pct']:.1f}%) {r['cat']}({r['conf']:.2f}) "
                      f"tex={r['tex']:.2f} scene={r['scene']:.2f} ed={r['edge_d']:.3f}")

print("\nDone.")
