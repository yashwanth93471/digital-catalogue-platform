"""
Extract embedded images from PDF catalogues using PyMuPDF.

Usage:
    python scripts/extract_images.py
    python scripts/extract_images.py path/to/catalogue.pdf
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # PyMuPDF
from config.settings import CATALOGUES_DIR, IMAGES_RAW_DIR

# ---- Configuration ---------------------------------------------------------
MIN_WIDTH = 300
MIN_HEIGHT = 300
DEFAULT_PDF = os.path.join(CATALOGUES_DIR, "sample.pdf")


def extract_images(pdf_path: str, output_dir: str) -> int:
    """Extract all embedded images from a PDF, skipping small ones."""
    if not os.path.isfile(pdf_path):
        print(f"[ERROR] PDF not found: {pdf_path}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    total_saved = 0
    total_skipped = 0

    print(f"[INFO] Opened: {os.path.basename(pdf_path)}")
    print(f"[INFO] Pages : {total_pages}")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Min size: {MIN_WIDTH}x{MIN_HEIGHT}px")
    print("-" * 50)

    start = time.time()

    for page_num in range(total_pages):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        if not image_list:
            continue

        img_counter = 0

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception as e:
                print(f"  [WARN] Page {page_num + 1}, image {img_index + 1}: extraction failed ({e})")
                continue

            width = base_image["width"]
            height = base_image["height"]

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                total_skipped += 1
                continue

            img_counter += 1
            ext = base_image["ext"]
            filename = f"page_{page_num + 1:03d}_img_{img_counter:02d}.{ext}"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "wb") as f:
                f.write(base_image["image"])

            total_saved += 1

        if image_list:
            print(f"  Page {page_num + 1:>4d}/{total_pages} — {img_counter} image(s) saved")

    elapsed = time.time() - start
    doc.close()

    print("-" * 50)
    print(f"[DONE] Saved  : {total_saved} images")
    print(f"[DONE] Skipped: {total_skipped} (below {MIN_WIDTH}x{MIN_HEIGHT})")
    print(f"[DONE] Time   : {elapsed:.1f}s")

    return total_saved


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    extract_images(pdf_path, IMAGES_RAW_DIR)
