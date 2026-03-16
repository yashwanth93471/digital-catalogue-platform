"""Check what OCR sees for Carving Series product pages."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.composite_segmenter import analyze_page_layout
from pipeline.clip_classifier import CLIPClassifier
from pipeline.name_extractor import _ocr_with_variants, _best_ocr_line
from config.settings import TESSERACT_PATH
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

clf = CLIPClassifier()
pdf = "catalogues/Carving Series.pdf"

for pn in [4, 5, 6, 9, 16]:
    info = analyze_page_layout(pdf, pn, clf, dpi=150)
    page_img = info["page_img"]
    w, h = page_img.size
    content = info["content_regions"]
    
    print(f"\n=== Page {pn}: {len(content)} regions, img={w}x{h} ===")
    
    # Check what text is around the regions
    # Left side (where text/name might be for pages with content on right)
    if content:
        all_x = [r["bbox"][0] for r in content]
        min_x = min(all_x)
        if min_x > 10:
            left = page_img.crop((0, 0, min_x, h))
            left_up = left.resize((left.width * 2, left.height * 2), Image.LANCZOS)
            gray = left_up.convert("L")
            lines = _ocr_with_variants(gray)
            name = _best_ocr_line(lines)
            print(f"  Left-side OCR: {name}")
            raw = [l.strip() for l in lines if l.strip()][:8]
            for ln in raw:
                print(f"    '{ln}'")
        
        # Between regions (right side between thumbnail and variant)
        for i, r in enumerate(content):
            bx, by, bw, bh = r["bbox"]
            print(f"  Region[{i}]: {bw}x{bh} at ({bx},{by}), cat={r['category']}")
        
        # Text below the smallest region (variant area) — might have product name
        by_area = sorted(content, key=lambda r: r["area_pct"], reverse=True)
        if len(by_area) >= 3:
            smallest = by_area[2]
            sx, sy, sw, sh = smallest["bbox"]
            below_top = sy + sh
            below_bot = min(h, below_top + int(h * 0.15))
            right_of = page_img.crop((sx, below_top, w, below_bot))
            right_up = right_of.resize((right_of.width * 2, right_of.height * 2), Image.LANCZOS)
            gray = right_up.convert("L")
            lines = _ocr_with_variants(gray)
            name = _best_ocr_line(lines)
            print(f"  Below-variant OCR: {name}")
            raw = [l.strip() for l in lines if l.strip()][:5]
            for ln in raw:
                print(f"    '{ln}'")
            
            # Right side of upper half 
            mid = by_area[1]  # medium region (thumbnail)
            mx, my, mw, mh = mid["bbox"]
            right_x = mx + mw
            gap = page_img.crop((right_x, my, w, my + mh))
            gap_up = gap.resize((gap.width * 2, gap.height * 2), Image.LANCZOS)
            gray = gap_up.convert("L")
            lines = _ocr_with_variants(gray)
            name = _best_ocr_line(lines)
            print(f"  Right-of-thumb OCR: {name}")
            raw = [l.strip() for l in lines if l.strip()][:8]
            for ln in raw:
                print(f"    '{ln}'")
