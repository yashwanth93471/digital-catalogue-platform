"""
Extract product names from catalogue pages using multi-strategy OCR.

Uses multiple preprocessing strategies (default, high-DPI, grayscale+contrast,
OTSU threshold) and picks the best result via confidence scoring.
Auto-detects intro/non-product pages.

Improvements:
  - Focuses OCR on the top header region (top 20%) where product names appear
  - Filters brand names (Mozilla, Exquisite Surfaces, Crystallo, etc.)
  - Accepts a set of skip pages (e.g., application pages merged with thumbnails)
  - Better noise filtering for catalogue boilerplate text

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

# ---- Brand names to strip from OCR results (case-insensitive) ----
BRAND_NAMES = {
    "mozilla", "mozilla granito", "mozillagranito",
    "exquisite surfaces", "exquisite",
    "crystallo", "crystallo collection", "crysiallo",
    "sunpark", "sunpark ceramics",
}

# ---- Words/phrases to ignore (case-insensitive) ----
IGNORE_WORDS = {
    # Sintered stone catalogues
    "surface", "series", "available sizes", "available", "sizes",
    "available size", "sintered", "stone", "sintered stone", "stones",
    "colour body", "color body", "fullbody", "full body",
    "thickness", "surfaces", "gloss", "matt", "finish",
    "mm", "15mm", "15 mm", "9mm", "9mm thickness",
    "index", "applications", "application",
    # Tile/slab catalogues
    "details", "collection", "type", "random", "endless",
    "tap to", "tap", "details collection", "tap to details",
    "back", "ceramic", "ceramics", "porcelain",
    "granito", "granito pvt",
}

# Lines matching these patterns are not product names
IGNORE_PATTERNS = [
    re.compile(r"^\d+\s*[xX\xd7]\s*\d+", re.IGNORECASE),     # 800x2400x15mm
    re.compile(r"^\d+\s*$"),                                    # standalone numbers
    re.compile(r"^[^a-zA-Z]*$"),                                # no letters at all
    re.compile(r"^\w{1,2}$"),                                   # 1-2 char junk
    re.compile(r"gloss\s*/\s*matt", re.IGNORECASE),            # Gloss/Matt header
    re.compile(r"^stone\s+\w+$", re.IGNORECASE),               # "Stone galaxy"
    re.compile(r"^sintered\s+\w+$", re.IGNORECASE),            # "sintered galaxy"
    re.compile(r"^stones\s+\w+$", re.IGNORECASE),              # "Stones SPARKLE"
    re.compile(r"^Sssintered\b", re.IGNORECASE),               # OCR misread
    re.compile(r"^eee\b", re.IGNORECASE),                      # OCR noise
    re.compile(r"^sintered\s+i\b", re.IGNORECASE),             # OCR garble
    re.compile(r"^stone\s+colors$", re.IGNORECASE),            # header
    re.compile(r"^\w+body$", re.IGNORECASE),                    # "FULLBODY"
    re.compile(r"www\.", re.IGNORECASE),                        # URLs
    re.compile(r"random\s*:\s*\d", re.IGNORECASE),             # "Random : 4"
    re.compile(r"^random$", re.IGNORECASE),                     # standalone "Random"
    re.compile(r"type\s*:", re.IGNORECASE),                     # "Type : Endless"
    re.compile(r"^\w+\s+granito", re.IGNORECASE),              # company names
    re.compile(r"pvt|ltd", re.IGNORECASE),                      # company suffixes
    re.compile(r"^\d+mm\b", re.IGNORECASE),                     # "9MM THICKNESS"
    re.compile(r"thickness", re.IGNORECASE),                    # thickness specs
    re.compile(r"^\.\s*\.?\s*\w+", re.IGNORECASE),             # ". . Random" ". Random"
    re.compile(r"^tap\s+to\b", re.IGNORECASE),                 # "Tap To Details"
    re.compile(r"^endless$", re.IGNORECASE),                    # standalone "Endless"
    re.compile(r"^back$", re.IGNORECASE),                       # "BACK" button text
    re.compile(r"ceramics|porcelain", re.IGNORECASE),          # material types
]

# Auto-detect non-product pages by these keywords
NON_PRODUCT_KEYWORDS = [
    "index", "content", "about us", "contact", "pvt. ltd", "pvt ltd",
    "granito pvt", "sunpark", "company", "copyright",
    "table of contents", "disclaimer",
]

# Known OCR corrections
OCR_CORRECTIONS = {
    "Asacia White": "Acacia White",
}


# ---- Multi-strategy OCR (header-focused) ----

def _render_header(page, dpi, header_fraction=0.20):
    """Render only the top portion of a page where product names appear."""
    full_rect = page.rect
    header_rect = fitz.Rect(
        full_rect.x0, full_rect.y0,
        full_rect.x1, full_rect.y0 + full_rect.height * header_fraction,
    )
    pix = page.get_pixmap(dpi=dpi, clip=header_rect)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def _render_full(page, dpi):
    """Render the full page."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def ocr_header_default(page, dpi=300):
    """Strategy 1: Header region at 300 DPI."""
    img = _render_header(page, dpi)
    return pytesseract.image_to_string(img)


def ocr_header_high_dpi(page, dpi=500):
    """Strategy 2: Header region at high DPI for small text."""
    img = _render_header(page, dpi)
    return pytesseract.image_to_string(img)


def ocr_header_contrast(page, dpi=400):
    """Strategy 3: Header region with grayscale + contrast boost."""
    img = _render_header(page, dpi)
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    return pytesseract.image_to_string(img)


def ocr_header_otsu(page, dpi=400):
    """Strategy 4: Header region with OTSU binarization."""
    img = _render_header(page, dpi)
    img_np = np.array(img)
    if len(img_np.shape) == 3 and img_np.shape[2] == 4:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
    elif len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(otsu)


def ocr_full_default(page, dpi=200):
    """Strategy 5: Full page fallback at standard DPI."""
    img = _render_full(page, dpi)
    return pytesseract.image_to_string(img)


OCR_STRATEGIES = [
    ("header_300",     ocr_header_default),
    ("header_500",     ocr_header_high_dpi),
    ("header_contrast", ocr_header_contrast),
    ("header_otsu",    ocr_header_otsu),
    ("full_200",       ocr_full_default),
]


# ---- Filtering & scoring ----

def _strip_brand_names(name: str) -> str:
    """Remove known brand names from OCR text."""
    for brand in sorted(BRAND_NAMES, key=len, reverse=True):
        name = re.sub(re.escape(brand), "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip()


def is_noise(line: str) -> bool:
    """Return True if the line is boilerplate, not a product name."""
    normalized = line.lower().strip()
    if normalized in IGNORE_WORDS:
        return True
    if normalized in BRAND_NAMES:
        return True
    for pattern in IGNORE_PATTERNS:
        if pattern.match(line):
            return True
    return False


def clean_product_name(name: str) -> str:
    """Remove trailing OCR artifacts and brand names from a product name."""
    # Strip brand names first
    name = _strip_brand_names(name)
    # Remove trailing non-alpha junk: symbols, single chars, numbers
    name = re.sub(r"\s+[^a-zA-Z\s]{1,5}\s*$", "", name)           # trailing "w--" etc.
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
    if re.search(r"[--\xab\[\]\(\){}|]", name):
        score -= 10

    # Bonus for names that look like product names (WORD WORD pattern)
    if re.match(r"^[A-Z][A-Za-z]+(\s+[A-Z][A-Za-z]+){0,3}$", name):
        score += 5

    return score


def extract_product_name(lines: list[str]) -> str | None:
    """Find the product name from OCR lines of a single page."""
    candidates = []
    for line in lines:
        cleaned = clean_product_name(line)
        if not cleaned or is_noise(cleaned):
            continue
        # Skip if the cleaned result is just a brand name
        if cleaned.lower().strip() in BRAND_NAMES:
            continue
        # Skip very short results (single char after cleaning)
        if len(cleaned) < 3:
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


def extract_all_names(pdf_path: str, output_json: str = None,
                      skip_pages: set = None) -> dict:
    """Extract product names from every page using multi-strategy OCR.

    Args:
        pdf_path: Path to the PDF file.
        output_json: Path to save the product_names.json.
        skip_pages: Set of page keys (e.g., {"page_006", "page_008"}) to skip.
                    Used to skip application-only pages that were merged.
    """
    if not os.path.isfile(pdf_path):
        print(f"[ERROR] PDF not found: {pdf_path}")
        return {}

    # Use provided output path or default
    if output_json is None:
        output_json = OUTPUT_JSON

    if skip_pages is None:
        skip_pages = set()

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    product_names = {}

    print(f"[INFO] OCR scanning: {os.path.basename(pdf_path)}")
    print(f"[INFO] Pages: {total_pages}")
    print(f"[INFO] Strategies: {len(OCR_STRATEGIES)} (header-focused)")
    if skip_pages:
        print(f"[INFO] Skipping {len(skip_pages)} merged application pages")
    print("-" * 60)

    for page_num in range(total_pages):
        page_key = f"page_{page_num + 1:03d}"
        page = doc[page_num]

        # Skip pages that were merged as application-only pages
        if page_key in skip_pages:
            print(f"  {page_key} -> [skipped -- merged application page]")
            continue

        # Run all OCR strategies and collect candidates
        best_name = None
        best_score = -999
        best_strategy = None

        for strategy_name, strategy_fn in OCR_STRATEGIES:
            try:
                text = strategy_fn(page)
            except Exception:
                continue

            # Auto-detect non-product pages (only check on full-page strategy)
            if strategy_name == "full_200" and is_non_product_page(text):
                if best_name is None:
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
            print(f"  {page_key} -> [skipped -- non-product page]")
            continue

        if best_name:
            best_name = OCR_CORRECTIONS.get(best_name, best_name)
            product_names[page_key] = best_name
            print(f"  {page_key} -> {best_name}  ({best_strategy})")
        else:
            print(f"  {page_key} -> [no product name detected]")

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
