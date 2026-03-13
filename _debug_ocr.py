"""Debug what _ocr_near_regions sees for PARKING pages."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image
from pipeline.composite_segmenter import analyze_page_layout
from pipeline.clip_classifier import CLIPClassifier
from pipeline.name_extractor import _ocr_with_variants, _best_ocr_line
from config.settings import TESSERACT_PATH
import pytesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

clf = CLIPClassifier()
pdf = "catalogues/16x16 PARKING.pdf"

# Test pages: 1 (works=FOG-27), 3 (fails), 5 (fails), 12 (works=SP128)
for pn in [1, 3, 5, 12]:
    info = analyze_page_layout(pdf, pn, clf, dpi=150)
    content = info["content_regions"]
    page_img = info["page_img"]
    w, h = page_img.size
    
    print(f"\n=== Page {pn}: {len(content)} content regions ===")
    for i, r in enumerate(content):
        print(f"  Region {i}: bbox={r['bbox']}, area={r['area_pct']:.1f}%, "
              f"cat={r['category']}, edge_d={r.get('edge_density', 0):.3f}")
    
    if len(content) >= 2:
        by_area = sorted(content, key=lambda r: r["area_pct"], reverse=True)
        a1, a2 = by_area[0]["area_pct"], by_area[1]["area_pct"]
        similar = a2 > a1 * 0.65
        print(f"  Similar size: {similar} (a1={a1:.1f}%, a2={a2:.1f}%)")
        
        if similar:
            by_x = sorted(by_area[:2], key=lambda r: r["bbox"][0])
            lx, ly, lw, lh = by_x[0]["bbox"]
            print(f"  Left tile bbox: x={lx}, y={ly}, w={lw}, h={lh}")
            
            below_top = ly + lh
            below_bot = min(h, below_top + int(h * 0.12))
            print(f"  Below-left crop: y={below_top}-{below_bot}, x={lx}-{lx+lw}")
            
            if below_bot - below_top > 10:
                crop = page_img.crop((lx, below_top, lx + lw, below_bot))
                rw, rh = crop.size
                if rh < 200:
                    scale = max(2, 200 // max(rh, 1))
                    crop = crop.resize((rw * scale, rh * scale), Image.LANCZOS)
                
                # Save the crop for visual inspection
                crop.save(f"_debug_below_left_p{pn}.png")
                
                # OCR it
                gray = crop.convert("L")
                lines = _ocr_with_variants(gray)
                name = _best_ocr_line(lines)
                print(f"  Below-left OCR result: {name}")
                print(f"  Raw lines (first 10):")
                for ln in [l.strip() for l in lines if l.strip()][:10]:
                    print(f"    '{ln}'")

print("\nDone. Check _debug_below_left_p*.png files.")
