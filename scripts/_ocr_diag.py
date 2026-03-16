"""Diagnostic: OCR the area below tiles on PARKING pages."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.composite_segmenter import render_page, _find_large_rects, _remove_nested_rects
from pipeline.name_extractor import extract_name_ocr_from_image, _ocr_with_variants, _best_ocr_line
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pdf = "catalogues/16x16 PARKING.pdf"
pages = [1, 2, 5, 10]

for pn in pages:
    print(f"\n=== Page {pn} ===")
    img = render_page(pdf, pn, dpi=200)
    w, h = img.size
    
    # Find content rects
    img_np = np.array(img)
    rects = _remove_nested_rects(_find_large_rects(img_np, min_area_ratio=0.04))
    
    # Below tiles region
    max_y = max(ry + rh for rx, ry, rw, rh in rects) if rects else int(h * 0.6)
    below = img.crop((0, max_y, w, min(h, max_y + int(h * 0.2))))
    
    print(f"  Below-tile region: y={max_y} to {min(h, max_y + int(h * 0.2))}")
    
    # Try OCR with variants
    gray = below.convert("L")
    lines = _ocr_with_variants(gray)
    print(f"  OCR lines from below ({len(lines)}):")
    for line in lines[:15]:
        line = line.strip()
        if line:
            print(f"    '{line}'")
    
    # Also try with higher DPI - render just the bottom region
    import fitz, io
    doc = fitz.open(pdf)
    page = doc[pn - 1]
    # Bottom 30% at 400 DPI 
    clip = fitz.Rect(page.rect.x0, page.rect.y0 + page.rect.height * 0.70,
                     page.rect.x1, page.rect.y1)
    pix = page.get_pixmap(dpi=400, clip=clip)
    hi_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    doc.close()
    
    print(f"\n  High-DPI bottom 30%:")
    hi_lines = _ocr_with_variants(hi_img)
    for line in hi_lines[:15]:
        line = line.strip()
        if line:
            print(f"    '{line}'")
    
    # Try the targeted region more specifically - below left tile only
    if len(rects) >= 2:
        by_x = sorted(rects[:2], key=lambda r: r[0])
        lx, ly, lw, lh = by_x[0]  # left tile
        # Below left tile
        below_left = img.crop((lx, ly + lh, lx + lw, min(h, ly + lh + int(h * 0.12))))
        gray_bl = below_left.convert("L")
        bl_lines = _ocr_with_variants(gray_bl)
        print(f"\n  Below-LEFT-tile region:")
        for line in bl_lines[:10]:
            line = line.strip()
            if line:
                print(f"    '{line}'")

print("\nDone.")
