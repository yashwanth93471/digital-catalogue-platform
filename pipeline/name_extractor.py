"""
Product-name extraction from PDF text blocks with OCR fallback.

Strategy
--------
  1. **PyMuPDF text blocks** -- the PDF already contains selectable text.
     Filter to the top 25 % of the page, pick the block with the largest
     font size, strip brand names.

  2. **LayoutParser** (optional) -- when installed, render the page and
     run deep-learning layout detection to locate *title* regions.  Gives
     more reliable region segmentation on complex layouts.

  3. **Tesseract OCR fallback** -- if no usable text is found, render the
     header region at 400 DPI, enhance contrast, and OCR.

All strategies filter out known brand names (Mozilla, Crystallo,
Exquisite Surfaces ...) and boilerplate catalogue text.
"""

import io
import os
import re

import fitz
import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

# ── Optional LayoutParser import ───────────────────────────────
try:
    import layoutparser as lp
    HAS_LAYOUTPARSER = True
except ImportError:
    HAS_LAYOUTPARSER = False


# ── Brand names (case-insensitive) ─────────────────────────────
BRAND_NAMES = {
    "mozilla", "mozilla granito", "mozillagranito",
    "exquisite surfaces", "exquisite",
    "crystallo", "crystallo collection", "crysiallo",
    "sunpark", "sunpark ceramics",
    "dream tile world",
}

# Words / phrases to reject
IGNORE_WORDS = {
    "surface", "series", "available", "sizes", "sintered", "stone",
    "colour body", "color body", "fullbody", "thickness", "surfaces",
    "gloss", "matt", "finish", "mm", "index", "application",
    "details", "collection", "type", "random", "endless",
    "tap to", "tap", "ceramic", "ceramics", "porcelain", "back",
    "granito", "pvt", "ltd", "www", "9mm", "15mm", "tap to details",
    "available sizes", "available size", "9mm thickness",
}

IGNORE_PATTERNS = [
    re.compile(r"^\d+\s*[xX\u00d7]\s*\d+", re.I),   # 800x2400
    re.compile(r"^\d+\s*$"),                           # lone numbers
    re.compile(r"^[^a-zA-Z]*$"),                       # no letters
    re.compile(r"^\w{1,2}$"),                          # 1-2 char junk
    re.compile(r"www\.", re.I),
    re.compile(r"pvt|ltd", re.I),
    re.compile(r"^\d+mm", re.I),
    re.compile(r"thickness", re.I),
    re.compile(r"type\s*:", re.I),
    re.compile(r"random\s*:", re.I),
    re.compile(r"^\.\s*", re.I),                       # ". Random"
    re.compile(r"^tap\s+to", re.I),
    re.compile(r"ceramics|porcelain", re.I),
    re.compile(r"^endless$", re.I),
    re.compile(r"^random$", re.I),
    re.compile(r"^back$", re.I),
    re.compile(r"@.*@", re.I),                         # OCR artefact on decorative fonts
    re.compile(r"special\s*series", re.I),
    re.compile(r"punch\s*series", re.I),
]

NON_PRODUCT_KEYWORDS = [
    "index", "content", "about us", "contact", "pvt. ltd", "pvt ltd",
    "company", "copyright", "disclaimer", "table of contents",
]


# ── internal helpers ───────────────────────────────────────────

def _strip_brands(text: str) -> str:
    for b in sorted(BRAND_NAMES, key=len, reverse=True):
        text = re.sub(re.escape(b), "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _clean(name: str) -> str:
    name = _strip_brands(name)
    # Strip stray quotes and punctuation that OCR often produces
    name = re.sub(r'["\'\u201c\u201d\u2018\u2019`]+', '', name)
    name = re.sub(r"\s*(?:collection|details|crystallo|crysiallo)\b", "", name, flags=re.I)
    name = re.sub(r"\s*[\(\)\[\]{}|]+\s*", " ", name)
    # Strip dimension patterns: "400X400 MM", "800x1600"
    name = re.sub(r"\b\d{2,4}\s*[xX\u00d7]\s*\d{2,4}\s*(?:mm)?\b", "", name, flags=re.I)
    # Strip series / type descriptions
    name = re.sub(r"\b[BM]{0,2}\s*(?:PUNCH|SPECIAL|SPCIAL|RANDOM)\s*SERIES\b", "", name, flags=re.I)
    name = re.sub(r"\b(?:PUNCH|SPECIAL|SPCIAL)\s+SERIES\b", "", name, flags=re.I)
    # Strip "SIZE/SIRE/SUE SERIES RANDOM" (OCR variants of "SIZE SERIES")
    name = re.sub(r"\s+(?:SIRE|SIZE|SUE)\s+SERIES\b.*", "", name, flags=re.I)
    name = re.sub(r"\s+SERIES\s+(?:RANDOM|ENDLESS)\b.*", "", name, flags=re.I)
    name = re.sub(r"\bSERIES\b", "", name, flags=re.I)
    # Strip stray trailing "MM", "M", "MB", "BM" that were part of size/series text
    name = re.sub(r"\s+[BM]{1,2}\s*$", "", name)
    name = re.sub(r"\s+MM\s*$", "", name, flags=re.I)
    name = re.sub(r"\s*[^\w\s]+$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _is_noise(text: str) -> bool:
    t = text.lower().strip()
    if t in IGNORE_WORDS or t in BRAND_NAMES:
        return True
    for pat in IGNORE_PATTERNS:
        if pat.match(text):
            return True
    return False


def _starts_with_boilerplate(text: str) -> bool:
    """Check if the first word of text is a known boilerplate word.

    Catches lines like 'Sintered ...', 'Stone BLE', 'SERIES ...'
    without falsely filtering product names like 'Saga Stone Himalaya'.
    """
    words = text.split()
    return bool(words) and words[0].lower() in IGNORE_WORDS


def _is_non_product_page(text: str) -> bool:
    low = text.lower()
    for kw in NON_PRODUCT_KEYWORDS:
        if kw in low:
            return True
    if sum(c.isalpha() for c in text) < 20:
        return True
    return False


# ── primary: PyMuPDF text blocks ───────────────────────────────

def extract_name_from_text_blocks(text_blocks, page_height: float) -> str | None:
    """Pick the most prominent product name from extracted text blocks.

    Focuses on the header region (top 25 %) and prefers the largest font.
    """
    header = [b for b in text_blocks if b.y_position < page_height * 0.25]
    if not header:
        header = [b for b in text_blocks if b.y_position < page_height * 0.40]
    if not header:
        return None

    header.sort(key=lambda b: (-b.font_size, b.y_position))

    for block in header:
        cleaned = _clean(block.text)
        if not cleaned or _is_noise(cleaned) or _starts_with_boilerplate(cleaned) or len(cleaned) < 3:
            continue
        return cleaned.upper()
    return None


# ── optional: LayoutParser title detection ─────────────────────

def extract_name_layoutparser(pdf_path: str, page_num: int) -> str | None:
    """Use LayoutParser to detect title regions (when installed)."""
    if not HAS_LAYOUTPARSER:
        return None

    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=150)
        import numpy as np
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            import cv2
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        doc.close()

        # Use the simple rule-based layout model (no detectron2 needed)
        layout = lp.Layout([])
        # Extract text with OCR on the rendered image
        ocr_text = pytesseract.image_to_data(
            Image.fromarray(img_np),
            output_type=pytesseract.Output.DICT,
        )

        # Find text in the top 20% with high confidence
        header_h = pix.h * 0.20
        candidates = []
        for i, txt in enumerate(ocr_text["text"]):
            txt = txt.strip()
            if not txt or int(ocr_text["conf"][i]) < 50:
                continue
            if ocr_text["top"][i] < header_h:
                cleaned = _clean(txt)
                if (cleaned and not _is_noise(cleaned)
                        and not _starts_with_boilerplate(cleaned)
                        and len(cleaned) >= 5):
                    candidates.append((cleaned, int(ocr_text["conf"][i])))

        if candidates:
            # Group adjacent words into a product name
            best = max(candidates, key=lambda c: c[1])
            return best[0].upper()
    except Exception:
        pass
    return None


# ── fallback: Tesseract OCR on header ─────────────────────────

def _ocr_quality(text: str) -> float:
    """Score OCR output quality: ratio of alphanumeric chars to total.

    High-quality OCR → mostly letters/digits/spaces/hyphens.
    Garbage OCR from decorative fonts → lots of @, #, |, etc.
    """
    if not text:
        return 0.0
    good = sum(c.isalnum() or c in (' ', '-') for c in text)
    return good / len(text)


def _best_ocr_line(lines: list[str]) -> str | None:
    """Pick the best product-name line from a list of OCR lines."""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cleaned = _clean(line)
        if (cleaned
                and not _is_noise(cleaned)
                and not _starts_with_boilerplate(cleaned)
                and len(cleaned) >= 3
                and _ocr_quality(cleaned) >= 0.70):
            return cleaned.upper()
    return None


def _ocr_with_variants(pil_gray: Image.Image) -> list[str]:
    """Run Tesseract with multiple preprocessing variants.

    Returns a list of all OCR text lines across all variants
    (best-quality variants first).
    """
    variants = []

    # Variant 1: standard — contrast + sharpen
    v1 = ImageEnhance.Contrast(pil_gray).enhance(2.0)
    v1 = v1.filter(ImageFilter.SHARPEN)
    variants.append(v1)

    # Variant 2: inverted (catches white text on dark backgrounds)
    import PIL.ImageOps as _iops
    v2 = _iops.invert(pil_gray)
    v2 = ImageEnhance.Contrast(v2).enhance(2.0)
    v2 = v2.filter(ImageFilter.SHARPEN)
    variants.append(v2)

    # Variant 3: Otsu binarisation
    import numpy as _np
    arr = _np.array(pil_gray)
    _, binarised = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v3 = Image.fromarray(binarised)
    variants.append(v3)

    all_lines: list[str] = []
    for v in variants:
        try:
            text = pytesseract.image_to_string(v)
        except Exception:
            continue
        all_lines.extend(text.split("\n"))
    return all_lines


def _ocr_region(pdf_path: str, page_num: int, top_frac: float,
                bottom_frac: float, dpi: int = 400,
                tesseract_path: str | None = None) -> str | None:
    """Render a vertical slice of the page and OCR it for product names.

    Tries multiple image preprocessing variants (standard, inverted,
    binarised) to handle decorative fonts and dark backgrounds.
    """
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]

    clip_rect = fitz.Rect(
        page.rect.x0, page.rect.y0 + page.rect.height * top_frac,
        page.rect.x1, page.rect.y0 + page.rect.height * bottom_frac,
    )
    pix = page.get_pixmap(dpi=dpi, clip=clip_rect)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()

    gray = img.convert("L")
    lines = _ocr_with_variants(gray)
    return _best_ocr_line(lines)


def extract_name_ocr(pdf_path: str, page_num: int,
                     tesseract_path: str | None = None) -> str | None:
    """Render multiple regions of the page at 400 DPI and OCR them.

    Scans: top 30%, bottom 25%, then middle 30-70%  — product names in
    composite catalogue pages can appear anywhere.
    """
    # Region 1: header (top 30%) — most common
    name = _ocr_region(pdf_path, page_num, 0.0, 0.30,
                       tesseract_path=tesseract_path)
    if name:
        return name

    # Region 2: footer area (bottom 25%) — some catalogues put names below
    name = _ocr_region(pdf_path, page_num, 0.75, 1.0,
                       tesseract_path=tesseract_path)
    if name:
        return name

    # Region 3: middle band — rarely, but covers remaining cases
    name = _ocr_region(pdf_path, page_num, 0.30, 0.70, dpi=300,
                       tesseract_path=tesseract_path)
    return name


def extract_name_ocr_from_image(pil_image: Image.Image,
                                tesseract_path: str | None = None) -> str | None:
    """OCR a PIL image directly for product name (used by flat-page pipeline).

    Scans top 20%, bottom 20%, then middle band, using multiple
    preprocessing variants per region.
    """
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    w, h = pil_image.size

    regions = [
        pil_image.crop((0, 0, w, int(h * 0.20))),              # top 20%
        pil_image.crop((0, int(h * 0.80), w, h)),              # bottom 20%
        pil_image.crop((0, int(h * 0.20), w, int(h * 0.50))), # middle band
    ]

    for region in regions:
        gray = region.convert("L")
        lines = _ocr_with_variants(gray)
        name = _best_ocr_line(lines)
        if name:
            return name
    return None


# ── public API (tries all strategies) ──────────────────────────

def extract_product_name(
    page,           # ParsedPage
    pdf_path: str,
    tesseract_path: str | None = None,
) -> str | None:
    """Try every strategy to extract the product name for *page*.

    Returns the name in UPPER CASE, or ``None``.
    """
    # 1. PyMuPDF text blocks (fast, no rendering)
    name = extract_name_from_text_blocks(page.text_blocks, page.height)
    if name:
        return name

    # 2. LayoutParser title detection (optional)
    name = extract_name_layoutparser(pdf_path, page.page_num)
    if name:
        return name

    # 3. Tesseract OCR on multiple regions (header, footer, middle)
    name = extract_name_ocr(pdf_path, page.page_num, tesseract_path)
    return name


def is_non_product_page_text(page) -> bool:
    """Quick check whether *page* looks like an intro / index page.

    Pages with NO text blocks are NOT flagged -- they may contain
    product names as images that need OCR.
    """
    if not page.text_blocks:
        return False
    all_text = " ".join(b.text for b in page.text_blocks)
    return _is_non_product_page(all_text)
