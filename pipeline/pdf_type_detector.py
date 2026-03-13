"""
Detect the structural type of a PDF catalogue.

Two fundamental layouts exist in tile catalogues:

  **Type A — Multi-image pages**
    Each page has multiple separate embedded image objects (e.g. 2–6).
    Text is stored as selectable PDF text blocks with font-size metadata.
    The existing pipeline handles these perfectly.

  **Type B — Flat-page / composite-image pages**
    Each page is exported as a single full-page image (designed in
    Photoshop / InDesign / Illustrator, then flattened).  The texture,
    room scene, product name, borders, and decorations are all baked
    into one image.  There are virtually no PDF text blocks.

This module analyses the first *N* pages and returns the dominant type.
It also supports **per-page** detection for mixed PDFs.
"""

import fitz  # PyMuPDF

# ── Thresholds ─────────────────────────────────────────────────
# A page is "flat" when it has just 1 large image covering most of the
# page area, and no meaningful text blocks.
MAX_IMAGES_FOR_FLAT = 1            # ≤ 1 large image on the page
MIN_AREA_COVERAGE = 0.60           # image covers ≥ 60 % of page area
MAX_TEXT_BLOCKS_FOR_FLAT = 2       # ≤ 2 text blocks (maybe a page #)
SAMPLE_PAGES = 8                   # check first N pages
FLAT_PAGE_MAJORITY = 0.60          # if ≥ 60 % pages are flat → Type B

MIN_IMAGE_AREA = 300 * 300         # same as pdf_parser.py


class PDFType:
    MULTI_IMAGE = "multi_image"    # Type A
    FLAT_PAGE = "flat_page"        # Type B
    MIXED = "mixed"                # some A, some B


def _page_is_flat(page, doc) -> bool:
    """Check whether a single page is a flat composite image page."""
    page_area = page.rect.width * page.rect.height
    if page_area == 0:
        return False

    # Count large images
    large_images = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            base = doc.extract_image(xref)
        except Exception:
            continue
        w, h = base["width"], base["height"]
        if w * h >= MIN_IMAGE_AREA:
            # Get image rect on the page
            try:
                rects = page.get_image_rects(xref)
                if rects:
                    r = rects[0]
                    img_page_area = r.width * r.height
                else:
                    img_page_area = w * h
            except Exception:
                img_page_area = w * h
            large_images.append({
                "w": w, "h": h,
                "pixel_area": w * h,
                "page_area_ratio": img_page_area / page_area if page_area else 0,
            })

    # Count text blocks
    try:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        text_blocks = [
            b for b in blocks
            if b.get("type") == 0
            and any(
                span["text"].strip()
                for line in b.get("lines", [])
                for span in line.get("spans", [])
            )
        ]
    except Exception:
        text_blocks = []

    # Decision: flat if exactly 1 large image covering most of the page,
    # and very few / no text blocks.
    if len(large_images) > MAX_IMAGES_FOR_FLAT:
        return False

    if len(large_images) == 1:
        coverage = large_images[0]["page_area_ratio"]
        few_text = len(text_blocks) <= MAX_TEXT_BLOCKS_FOR_FLAT
        return coverage >= MIN_AREA_COVERAGE and few_text

    # 0 large images but also no text — might be a blank page
    return False


def detect_page_types(pdf_path: str) -> list[str]:
    """Return a per-page list of 'flat_page' or 'multi_image'."""
    doc = fitz.open(pdf_path)
    types = []
    for i in range(len(doc)):
        page = doc[i]
        t = PDFType.FLAT_PAGE if _page_is_flat(page, doc) else PDFType.MULTI_IMAGE
        types.append(t)
    doc.close()
    return types


def detect_pdf_type(pdf_path: str) -> str:
    """Classify the overall PDF as multi_image, flat_page, or mixed.

    Samples the first ``SAMPLE_PAGES`` pages.
    """
    doc = fitz.open(pdf_path)
    n_pages = min(len(doc), SAMPLE_PAGES)
    flat_count = 0
    for i in range(n_pages):
        if _page_is_flat(doc[i], doc):
            flat_count += 1
    doc.close()

    ratio = flat_count / n_pages if n_pages else 0
    if ratio >= FLAT_PAGE_MAJORITY:
        return PDFType.FLAT_PAGE
    elif flat_count == 0:
        return PDFType.MULTI_IMAGE
    else:
        return PDFType.MIXED
