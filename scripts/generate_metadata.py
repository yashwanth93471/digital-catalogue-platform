"""
Generate metadata.json for each product folder.

Reads product_names.json and OCR text from the PDF to populate:
  - name, category, tags, description

Usage:
    python scripts/generate_metadata.py
"""

import sys
import os
import io
import re
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
import pytesseract
from PIL import Image
from config.settings import (
    CATALOGUES_DIR, PRODUCTS_DIR, TESSERACT_PATH,
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

DEFAULT_PDF = os.path.join(CATALOGUES_DIR, "sample.pdf")
NAMES_JSON = os.path.join(PRODUCTS_DIR, "product_names.json")
RENDER_DPI = 200

# ─── Series detection ──────────────────────────────────────────
KNOWN_SERIES = [
    "sparkle", "galaxy", "pure", "surface", "colors", "saga",
]

# ─── Body type detection ───────────────────────────────────────
BODY_KEYWORDS = {
    "fullbody": "Full Body",
    "full body": "Full Body",
    "colour body": "Colour Body",
    "color body": "Colour Body",
}

SIZE_PATTERN = re.compile(r"\d+\s*[xX×]\s*\d+(?:\s*[xX×]\s*\d+)?\s*mm", re.IGNORECASE)


def sanitize_folder_name(name: str) -> str:
    """Convert product name to lowercase_with_underscores."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def detect_series(lines: list[str], product_name: str) -> str:
    """Detect the product series from OCR lines."""
    text_lower = " ".join(lines).lower()
    for series in KNOWN_SERIES:
        if series in text_lower:
            return series.capitalize()
    # Fallback: use the last word of the product name if multi-word
    words = product_name.split()
    if len(words) >= 2:
        return words[-1]
    return "Standard"


def detect_body_type(lines: list[str]) -> str:
    """Detect body type from OCR text."""
    text_lower = " ".join(lines).lower()
    for keyword, label in BODY_KEYWORDS.items():
        if keyword in text_lower:
            return label
    return "Standard"


def detect_sizes(lines: list[str]) -> list[str]:
    """Extract available sizes from OCR text."""
    sizes = []
    for line in lines:
        match = SIZE_PATTERN.search(line)
        if match:
            size = re.sub(r"\s+", "", match.group())
            if size not in sizes:
                sizes.append(size)
    return sizes


def detect_color(product_name: str) -> str:
    """Extract color hint from product name."""
    color_words = {
        "white", "black", "grey", "gray", "ivory", "ice",
        "steel", "sparkle", "graphite", "diamond", "crema",
    }
    words = product_name.lower().split()
    colors = [w.capitalize() for w in words if w in color_words]
    return ", ".join(colors) if colors else words[-1].capitalize()


def build_tags(product_name: str, series: str, body_type: str, sizes: list[str]) -> list[str]:
    """Build a tag list from product attributes."""
    tags = ["sintered stone", "slab", "tile"]
    tags.append(series.lower())
    tags.append(detect_color(product_name).lower())
    if body_type != "Standard":
        tags.append(body_type.lower())
    for size in sizes:
        tags.append(size.lower())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def build_description(product_name: str, series: str, body_type: str, sizes: list[str]) -> str:
    """Generate a product description."""
    size_str = " and ".join(sizes) if sizes else "multiple sizes"
    return (
        f"{product_name} is a premium sintered stone slab from the {series} series. "
        f"Available in {size_str}. {body_type} construction."
    )


def generate_metadata(pdf_path: str, names_json: str, products_dir: str) -> int:
    """Generate metadata.json for each product folder."""
    if not os.path.isfile(names_json):
        print(f"[ERROR] Product names not found: {names_json}")
        return 0

    with open(names_json, "r", encoding="utf-8") as f:
        product_names = json.load(f)

    doc = fitz.open(pdf_path)

    print(f"[INFO] Generating metadata for {len(product_names)} products")
    print("-" * 55)

    count = 0

    for page_key, product_name in sorted(product_names.items()):
        page_num = int(page_key.split("_")[1]) - 1
        folder_name = sanitize_folder_name(product_name)
        product_dir = os.path.join(products_dir, folder_name)

        if not os.path.isdir(product_dir):
            print(f"  {folder_name}/ -- [SKIP] folder not found")
            continue

        # OCR the page for details
        page = doc[page_num]
        pix = page.get_pixmap(dpi=RENDER_DPI)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        # Extract attributes
        series = detect_series(lines, product_name)
        body_type = detect_body_type(lines)
        sizes = detect_sizes(lines)
        tags = build_tags(product_name, series, body_type, sizes)
        description = build_description(product_name, series, body_type, sizes)

        # List images in the folder
        images = sorted(f for f in os.listdir(product_dir) if f.endswith((".jpg", ".jpeg", ".png")))

        metadata = {
            "name": product_name,
            "category": "Sintered Stone",
            "series": series,
            "sizes": sizes,
            "body_type": body_type,
            "tags": tags,
            "description": description,
            "images": images,
        }

        meta_path = os.path.join(product_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        count += 1
        print(f"  {folder_name}/ -> metadata.json  (series={series}, {len(sizes)} sizes, {len(tags)} tags)")

    doc.close()

    print("-" * 55)
    print(f"[DONE] {count} metadata files generated")

    return count


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    generate_metadata(pdf_path, NAMES_JSON, PRODUCTS_DIR)
