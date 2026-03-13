"""
Parse PDF catalogues with PyMuPDF.

Extracts every embedded image together with its page coordinates and
every text block with font-size metadata.  A lightweight perceptual hash
(dHash) is computed per image so repeated logos / watermarks can be
detected downstream.
"""

import io
import os

import fitz  # PyMuPDF
from PIL import Image

# Minimum pixel area to keep an image (skip tiny icons / QR codes)
MIN_IMAGE_AREA = 300 * 300


# ── Data containers ────────────────────────────────────────────

class PageImage:
    """An image extracted from a single PDF page."""

    __slots__ = (
        "image", "page_num", "xref", "rect",
        "width", "height", "area", "phash",
        "clip_category", "clip_confidence",
    )

    def __init__(self, pil_image, page_num, xref, rect, width, height):
        self.image: Image.Image = pil_image
        self.page_num: int = page_num
        self.xref: int = xref
        self.rect = rect
        self.width: int = width
        self.height: int = height
        self.area: int = width * height
        self.phash: str = self._dhash()
        self.clip_category: str = ""
        self.clip_confidence: float = 0.0

    # 64-bit difference hash -- robust to JPEG re-compression.
    def _dhash(self, size: int = 9) -> str:
        small = self.image.resize((size, size - 1), Image.LANCZOS).convert("L")
        pixels = list(small.getdata())
        bits = []
        for row in range(size - 1):
            for col in range(size - 1):
                idx = row * size + col
                bits.append("1" if pixels[idx] < pixels[idx + 1] else "0")
        return "".join(bits)


class TextBlock:
    """A text span extracted from a PDF page."""

    __slots__ = ("text", "bbox", "page_num", "font_size", "y_position")

    def __init__(self, text, bbox, page_num, font_size=0.0):
        self.text: str = text.strip()
        self.bbox = bbox  # (x0, y0, x1, y1)
        self.page_num: int = page_num
        self.font_size: float = font_size
        self.y_position: float = bbox[1]


class ParsedPage:
    """All data extracted from one PDF page."""

    def __init__(self, page_num, width, height):
        self.page_num: int = page_num
        self.width: float = width
        self.height: float = height
        self.images: list[PageImage] = []
        self.text_blocks: list[TextBlock] = []
        # Filled later by classifier / filter
        self.filtered_images: list[PageImage] = []
        self.textures: list[PageImage] = []
        self.room_scenes: list[PageImage] = []


# ── Main parser ────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> list[ParsedPage]:
    """Parse a PDF and return a list of ``ParsedPage`` objects.

    Each page carries its embedded images (as PIL RGB) and its text
    blocks with font-size metadata.
    """
    doc = fitz.open(pdf_path)
    pages: list[ParsedPage] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        parsed = ParsedPage(page_num + 1, page.rect.width, page.rect.height)

        # ---- images ----------------------------------------------------
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            w, h = base_image["width"], base_image["height"]
            if w * h < MIN_IMAGE_AREA:
                continue

            # Position on the page
            try:
                rects = page.get_image_rects(xref)
                rect = rects[0] if rects else fitz.Rect(0, 0, w, h)
            except Exception:
                rect = fitz.Rect(0, 0, w, h)

            try:
                pil_img = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
            except Exception:
                continue

            parsed.images.append(PageImage(pil_img, page_num + 1, xref, rect, w, h))

        # ---- text blocks -----------------------------------------------
        try:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        except Exception:
            blocks = []

        for block in blocks:
            if block.get("type") != 0:
                continue
            parts: list[str] = []
            max_fs = 0.0
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    parts.append(span["text"])
                    max_fs = max(max_fs, span.get("size", 0))
            text = " ".join(parts).strip()
            if text:
                parsed.text_blocks.append(
                    TextBlock(text, block["bbox"], page_num + 1, max_fs)
                )

        pages.append(parsed)

    doc.close()
    return pages
