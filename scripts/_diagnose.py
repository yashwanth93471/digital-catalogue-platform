"""Diagnose problem pages: try PDF text extraction + region-cropped OCR."""
import sys, os, io, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from config.settings import CATALOGUES_DIR, TESSERACT_PATH

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

pdf_path = os.path.join(CATALOGUES_DIR, "DREAM TILE WORLD.pdf")
doc = fitz.open(pdf_path)

# Problem pages (0-indexed): 0,1,2,8,9,13,22,23,27,32,33,34,35
PROBLEM_PAGES = [0, 1, 2, 8, 9, 13, 22, 23, 27, 32, 33, 34, 35]

for p in PROBLEM_PAGES:
    page = doc[p]
    page_key = f"page_{p+1:03d}"
    print(f"\n{'='*70}")
    print(f"  PAGE {p+1} ({page_key})")
    print(f"{'='*70}")

    # 1) PyMuPDF embedded text
    text = page.get_text("text").strip()
    print(f"\n--- PyMuPDF text extraction ---")
    if text:
        for line in text.split("\n"):
            if line.strip():
                print(f"  |{line.strip()}|")
    else:
        print("  [no embedded text]")

    # 2) Full-page high-DPI OCR
    pix = page.get_pixmap(dpi=400)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    ocr_full = pytesseract.image_to_string(img).strip()
    print(f"\n--- Full-page OCR (400 DPI) ---")
    for line in ocr_full.split("\n"):
        if line.strip():
            print(f"  |{line.strip()}|")

    # 3) Top 30% region OCR (where product names usually are)
    w, h = img.size
    top_region = img.crop((0, 0, w, int(h * 0.3)))
    ocr_top = pytesseract.image_to_string(top_region).strip()
    print(f"\n--- Top 30% OCR (400 DPI) ---")
    for line in ocr_top.split("\n"):
        if line.strip():
            print(f"  |{line.strip()}|")

    # 4) Center region OCR (20%-60% vertically, 10%-90% horizontally)
    center_region = img.crop((int(w * 0.1), int(h * 0.2), int(w * 0.9), int(h * 0.6)))
    ocr_center = pytesseract.image_to_string(center_region).strip()
    print(f"\n--- Center region OCR (400 DPI) ---")
    for line in ocr_center.split("\n"):
        if line.strip():
            print(f"  |{line.strip()}|")

    # 5) Inverted image OCR (for light text on dark bg)
    inv_img = ImageOps.invert(img.convert("RGB"))
    inv_gray = inv_img.convert("L")
    inv_contrast = ImageEnhance.Contrast(inv_gray).enhance(2.5)
    ocr_inv = pytesseract.image_to_string(inv_contrast).strip()
    print(f"\n--- Inverted + contrast OCR (400 DPI) ---")
    for line in ocr_inv.split("\n"):
        if line.strip():
            print(f"  |{line.strip()}|")

    # 6) Inverted center region
    center_inv = ImageOps.invert(img.crop((int(w*0.1), int(h*0.2), int(w*0.9), int(h*0.6))).convert("RGB"))
    center_inv_gray = ImageEnhance.Contrast(center_inv.convert("L")).enhance(2.5)
    ocr_center_inv = pytesseract.image_to_string(center_inv_gray).strip()
    print(f"\n--- Inverted center region OCR ---")
    for line in ocr_center_inv.split("\n"):
        if line.strip():
            print(f"  |{line.strip()}|")

    # 7) Morphological + OTSU on center
    pix2 = page.get_pixmap(dpi=400)
    img_np = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.h, pix2.w, pix2.n)
    if pix2.n == 4:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Center crop
    ch, cw = gray.shape
    center_gray = gray[int(ch*0.15):int(ch*0.55), int(cw*0.05):int(cw*0.95)]
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(center_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    ocr_adaptive = pytesseract.image_to_string(adaptive).strip()
    print(f"\n--- Adaptive threshold center OCR ---")
    for line in ocr_adaptive.split("\n"):
        if line.strip():
            print(f"  |{line.strip()}|")

doc.close()
