"""
Extract product names from catalogue pages using multi-strategy OCR.

Uses multiple preprocessing strategies (default, high-DPI, grayscale+contrast,
OTSU threshold) and picks the best result via confidence scoring.
Auto-detects intro/non-product pages.

Usage:
    python scripts/extract_product_names.py
    python scripts/extract_product_names.py path/to/catalogue.pdf

Output:
    products/product_names.json
"""

import sys
import os
import io
import re
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from config.settings import CATALOGUES_DIR, PRODUCTS_DIR, TESSERACT_PATH

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

DEFAULT_PDF = os.path.join(CATALOGUES_DIR, "sample.pdf")
OUTPUT_JSON = os.path.join(PRODUCTS_DIR, "product_names.json")

# ─── Words/phrases to ignore (case-insensitive) ────────────────
IGNORE_WORDS = {
    # Sintered stone catalogues
    "surface", "series", "available sizes", "available", "sizes",
    "available size", "sintered", "stone", "sintered stone", "stones",
    "colour body", "color body", "fullbody", "full body",
    "thickness", "surfaces", "gloss", "matt", "finish",
    "mm", "15mm", "15 mm",
    "index", "applications", "application",
    # Tile catalogues
    "details", "collection", "type", "random", "endless",
    "tap to", "tap", "details collection",
}

# Lines matching these patterns are not product names
IGNORE_PATTERNS = [
    re.compile(r"^\d+\s*[xX×]\s*\d+", re.IGNORECASE),       # 800x2400x15mm
    re.compile(r"^\d+\s*$"),                                   # standalone numbers
    re.compile(r"^[^a-zA-Z]*$"),                               # no letters at all
    re.compile(r"^\w{1,2}$"),                                  # 1-2 char junk
    re.compile(r"gloss\s*/\s*matt", re.IGNORECASE),           # Gloss/Matt header
    re.compile(r"^stone\s+\w+$", re.IGNORECASE),              # "Stone galaxy"
    re.compile(r"^sintered\s+\w+$", re.IGNORECASE),           # "sintered galaxy"
    re.compile(r"^stones\s+\w+$", re.IGNORECASE),             # "Stones SPARKLE"
    re.compile(r"^Sssintered\b", re.IGNORECASE),              # OCR misread
    re.compile(r"^eee\b", re.IGNORECASE),                     # OCR noise
    re.compile(r"^sintered\s+i\b", re.IGNORECASE),            # OCR garble
    re.compile(r"^stone\s+colors$", re.IGNORECASE),           # header
    re.compile(r"^\w+body$", re.IGNORECASE),                   # "FULLBODY"
    re.compile(r"www\.", re.IGNORECASE),                       # URLs
    re.compile(r"random\s*:\s*\d", re.IGNORECASE),            # "Random : 4"
    re.compile(r"type\s*:", re.IGNORECASE),                    # "Type : Endless"
    re.compile(r"^\w+\s+granito", re.IGNORECASE),             # company names
    re.compile(r"pvt|ltd", re.IGNORECASE),                     # company suffixes
]

# Auto-detect non-product pages by these keywords
NON_PRODUCT_KEYWORDS = [
    "index", "content", "about us", "contact", "pvt. ltd", "pvt ltd",
    "granito pvt", "sunpark", "company", "copyright",
]

# Known OCR corrections
OCR_CORRECTIONS = {
    "Asacia White": "Acacia White",
}


# ─── Multi-strategy OCR ────────────────────────────────────────

def ocr_default(page, dpi=200):
    """Strategy 1: Default rendering."""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def ocr_high_dpi(page, dpi=400):
    """Strategy 2: High DPI for small/decorative text."""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def ocr_grayscale_contrast(page, dpi=300):
    """Strategy 3: Grayscale + contrast boost + sharpen."""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    return pytesseract.image_to_string(img)


def ocr_otsu(page, dpi=300):
    """Strategy 4: OTSU threshold binarization."""
    pix = page.get_pixmap(dpi=dpi)
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(otsu)


OCR_STRATEGIES = [
    ("default",    ocr_default),
    ("high_dpi",   ocr_high_dpi),
    ("contrast",   ocr_grayscale_contrast),
    ("otsu",       ocr_otsu),
]


# ─── Filtering & scoring ───────────────────────────────────────

def is_noise(line: str) -> bool:
    """Return True if the line is boilerplate, not a product name."""
    normalized = line.lower().strip()
    if normalized in IGNORE_WORDS:
        return True
    for pattern in IGNORE_PATTERNS:
        if pattern.match(line):
            return True
    return False


def clean_product_name(name: str) -> str:
    """Remove trailing OCR artifacts from a product name."""
    # Remove trailing non-alpha junk: symbols, single chars, numbers
    name = re.sub(r"\s+[^a-zA-Z\s]{1,5}\s*$", "", name)           # trailing "w»" etc.
    name = re.sub(r"\s*[\(\)\[\]{}|]+\s*", " ", name)              # brackets
    name = re.sub(r"\s*\d+[.,]\s*Crys\w+", "", name, flags=re.I)  # "2. Crystallo"
    name = re.sub(r"\s*Crys\w+", "", name, flags=re.I)             # "Crystallo"
    name = re.sub(r"\s*Crysi\w+", "", name, flags=re.I)            # "Crysiallo"
    name = re.sub(r"\s*Collection\b", "", name, flags=re.I)        # "Collection"
    name = re.sub(r"\s*DETAILS\b", "", name, flags=re.I)           # "DETAILS"
    name = re.sub(r"\s*[^\w\s]+$", "", name)                       # trailing symbols
    name = re.sub(r"\s+", " ", name).strip()                       # collapse spaces
    return name


def score_candidate(name: str) -> float:
    """Score a product name candidate. Higher = more likely a real product name."""
    score = 0.0
    words = name.split()
    word_count = len(words)

    # 2-3 words is ideal for a product name
    if 2 <= word_count <= 4:
        score += 10
    elif word_count == 1:
        score += 5

    # All-caps or title-case is good
    if name.isupper() or name.istitle():
        score += 5

    # Penalize very short names
    if len(name) < 3:
        score -= 20

    # Penalize names with lots of non-alpha chars
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in name) / max(len(name), 1)
    score += alpha_ratio * 10

    # Penalize if still has junk
    if re.search(r"[»«\[\]\(\){}|]", name):
        score -= 10

    return score


def extract_product_name(lines: list[str]) -> str | None:
    """Find the product name from OCR lines of a single page."""
    candidates = []
    for line in lines:
        cleaned = clean_product_name(line)
        if not cleaned or is_noise(cleaned):
            continue
        candidates.append(cleaned)

    if not candidates:
        return None

    # Score each candidate, pick the best
    scored = [(c, score_candidate(c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored[0][1] > 0 else None


def is_non_product_page(all_text: str) -> bool:
    """Auto-detect if a page is an intro/index/company page."""
    text_lower = all_text.lower()
    for keyword in NON_PRODUCT_KEYWORDS:
        if keyword in text_lower:
            return True
    # Very little readable text = likely a cover/image-only page
    alpha_chars = sum(c.isalpha() for c in all_text)
    if alpha_chars < 20:
        return True
    return False


def extract_all_names(pdf_path: str, output_json: str = None) -> dict:
    """Extract product names from every page using multi-strategy OCR."""
    if not os.path.isfile(pdf_path):
        print(f"[ERROR] PDF not found: {pdf_path}")
        return {}

    # Use provided output path or default
    if output_json is None:
        output_json = OUTPUT_JSON

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    product_names = {}

    print(f"[INFO] OCR scanning: {os.path.basename(pdf_path)}")
    print(f"[INFO] Pages: {total_pages}")
    print(f"[INFO] Strategies: {len(OCR_STRATEGIES)}")
    print("-" * 60)

    for page_num in range(total_pages):
        page_key = f"page_{page_num + 1:03d}"
        page = doc[page_num]

        # Run all OCR strategies and collect candidates
        best_name = None
        best_score = -999
        best_strategy = None

        for strategy_name, strategy_fn in OCR_STRATEGIES:
            try:
                text = strategy_fn(page)
            except Exception:
                continue

            # Auto-detect non-product pages on first strategy
            if strategy_name == "default" and is_non_product_page(text):
                best_name = None
                best_strategy = "skip"
                break

            lines = [l.strip() for l in text.split("\n") if l.strip()]
            name = extract_product_name(lines)

            if name:
                s = score_candidate(name)
                if s > best_score:
                    best_score = s
                    best_name = name
                    best_strategy = strategy_name

        if best_strategy == "skip":
            print(f"  {page_key} → [skipped — non-product page]")
            continue

        if best_name:
            best_name = OCR_CORRECTIONS.get(best_name, best_name)
            product_names[page_key] = best_name
            print(f"  {page_key} → {best_name}  ({best_strategy})")
        else:
            print(f"  {page_key} → [no product name detected]")

    doc.close()

    # Save JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(product_names, f, indent=2, ensure_ascii=False)

    print("-" * 60)
    print(f"[DONE] {len(product_names)} product names extracted")
    print(f"[DONE] Saved to: {output_json}")

    return product_names


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    extract_all_names(pdf_path)
