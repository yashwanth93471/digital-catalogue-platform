"""
Advanced ML diagnostics for problematic catalogue pages.

Uses:
• layout detection
• multi-OCR engines
• text-region detection
• OCR confidence scoring

pip install easyocr paddleocr scikit-image
"""

import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
import pytesseract
import easyocr
import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola

from config.settings import CATALOGUES_DIR, TESSERACT_PATH

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

reader = easyocr.Reader(['en'], gpu=False)

pdf_path = os.path.join(CATALOGUES_DIR, "DREAM TILE WORLD.pdf")

doc = fitz.open(pdf_path)

PROBLEM_PAGES = [0,1,2,8,9,13,22,23,27,32,33,34,35]


# -------------------------------------------------
# Detect text regions using contours
# -------------------------------------------------

def detect_text_regions(image):

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,50,150)

    contours,_ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []

    for cnt in contours:

        x,y,w,h = cv2.boundingRect(cnt)

        if w*h > 20000:
            regions.append((x,y,w,h))

    return regions


# -------------------------------------------------
# OCR with confidence
# -------------------------------------------------

def tesseract_conf(image):

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT
    )

    results = []

    for i,text in enumerate(data["text"]):

        if text.strip():

            conf = int(data["conf"][i])

            results.append((text,conf))

    return results


# -------------------------------------------------
# EasyOCR fallback
# -------------------------------------------------

def easy_ocr(image):

    results = reader.readtext(np.array(image))

    out = []

    for bbox,text,conf in results:

        out.append((text,conf))

    return out


# -------------------------------------------------
# Adaptive preprocessing
# -------------------------------------------------

def preprocess(image):

    gray = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2GRAY)

    thresh = threshold_sauvola(gray)

    binary = gray > thresh

    return (binary*255).astype(np.uint8)


# -------------------------------------------------
# Main diagnosis
# -------------------------------------------------

for p in PROBLEM_PAGES:

    page = doc[p]

    page_key = f"page_{p+1:03d}"

    print("\n"+"="*70)
    print(f"PAGE {p+1}")
    print("="*70)

    pix = page.get_pixmap(dpi=400)

    img = Image.open(io.BytesIO(pix.tobytes("png")))

    # ------------------------------------
    # Embedded text check
    # ------------------------------------

    text = page.get_text("text")

    if text.strip():

        print("\nEmbedded text detected")

        print(text[:500])

    else:

        print("\nNo embedded text")

    # ------------------------------------
    # Detect text regions
    # ------------------------------------

    regions = detect_text_regions(img)

    print(f"\nDetected {len(regions)} candidate text regions")

    for i,(x,y,w,h) in enumerate(regions[:5]):

        region = img.crop((x,y,x+w,y+h))

        print(f"\nRegion {i+1}")

        # Tesseract

        t_res = tesseract_conf(region)

        if t_res:

            best = sorted(t_res,key=lambda x:x[1],reverse=True)[0]

            print("Tesseract:",best)

        # EasyOCR

        e_res = easy_ocr(region)

        if e_res:

            best = sorted(e_res,key=lambda x:x[1],reverse=True)[0]

            print("EasyOCR:",best)

        # Preprocessed OCR

        prep = preprocess(region)

        prep_img = Image.fromarray(prep)

        p_res = pytesseract.image_to_string(prep_img)

        if p_res.strip():

            print("Preprocessed OCR:",p_res.strip())

doc.close()
