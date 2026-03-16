
pip install easyocr opencv-python scikit-image

import easyocr
import cv2
from skimage import exposure

reader = easyocr.Reader(['en'], gpu=False)


# -------------------------------------------------
# Advanced image preprocessing
# -------------------------------------------------

def preprocess_catalogue_image(img):

    img_np = np.array(img)

    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    # CLAHE improves contrast in textured catalogues
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)

    # gamma correction improves dark scans
    gamma = exposure.adjust_gamma(contrast, 1.2)

    # adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gamma,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    return thresh


# -------------------------------------------------
# OCR engine 1 — Tesseract with confidence
# -------------------------------------------------

def ocr_tesseract(img):

    data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT
    )

    words = []
    confidences = []

    for i, text in enumerate(data["text"]):

        text = text.strip()

        if not text:
            continue

        conf = int(data["conf"][i])

        if conf > 50:
            words.append(text)
            confidences.append(conf)

    return words


# -------------------------------------------------
# OCR engine 2 — EasyOCR fallback
# -------------------------------------------------

def ocr_easy(img):

    img_np = np.array(img)

    results = reader.readtext(img_np)

    words = []

    for bbox, text, conf in results:

        if conf > 0.5:
            words.append(text)

    return words


# -------------------------------------------------
# Detect title region using contours
# -------------------------------------------------

def detect_title_region(img):

    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = gray.shape

    title_candidates = []

    for cnt in contours:

        x,y,wc,hc = cv2.boundingRect(cnt)

        # titles usually appear near top of page
        if y < h * 0.30 and wc > w * 0.20:

            title_candidates.append((x,y,wc,hc))

    if not title_candidates:
        return img

    # choose largest region
    title_candidates.sort(key=lambda c: c[2]*c[3], reverse=True)

    x,y,wc,hc = title_candidates[0]

    return img.crop((x,y,x+wc,y+hc))


# -------------------------------------------------
# Multi-OCR pipeline
# -------------------------------------------------

def run_advanced_ocr(page):

    img = _render_header(page,400)

    title_region = detect_title_region(img)

    processed = preprocess_catalogue_image(title_region)

    candidates = []

    # Tesseract
    candidates.extend(ocr_tesseract(processed))

    # fallback
    if len(candidates) < 2:
        candidates.extend(ocr_easy(title_region))

    # remove duplicates
    candidates = list(set(candidates))

    return candidates
