"""
Extract images from PDF catalogues using Unstructured layout parsing.

Pipeline stage:
catalogues → images_raw
"""

import os
import sys
import json
import time
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CATALOGUES_DIR, IMAGES_RAW_DIR

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

MIN_WIDTH = 300
MIN_HEIGHT = 300

# ---------------------------------------------------
# Helper function
# ---------------------------------------------------

def save_image(element, output_dir, index):

    image_bytes = element.metadata.image_base64

    if not image_bytes:
        return None

    import base64

    img_data = base64.b64decode(image_bytes)

    filename = f"img_{index:05d}.png"
    path = os.path.join(output_dir, filename)

    with open(path, "wb") as f:
        f.write(img_data)

    return filename


# ---------------------------------------------------
# Main extraction logic
# ---------------------------------------------------

def extract_images(pdf_path):

    os.makedirs(IMAGES_RAW_DIR, exist_ok=True)

    print(f"\n[INFO] Processing: {os.path.basename(pdf_path)}")

    start = time.time()

    elements = partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        strategy="hi_res"
    )

    saved = 0
    metadata = []

    for i, element in enumerate(tqdm(elements)):

        if isinstance(element, Image):

            filename = save_image(element, IMAGES_RAW_DIR, saved)

            if filename is None:
                continue

            metadata.append({
                "filename": filename,
                "page": element.metadata.page_number
            })

            saved += 1

    elapsed = time.time() - start

    print(f"[DONE] Saved {saved} images in {elapsed:.2f}s")

    return metadata


# ---------------------------------------------------
# Batch catalogue processing
# ---------------------------------------------------

def process_catalogues():

    pdfs = [
        f for f in os.listdir(CATALOGUES_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdfs:
        print("[WARN] No PDFs found in catalogues/")
        return

    all_metadata = []

    for pdf in pdfs:

        path = os.path.join(CATALOGUES_DIR, pdf)

        metadata = extract_images(path)

        all_metadata.extend(metadata)

    meta_path = os.path.join(IMAGES_RAW_DIR, "images_metadata.json")

    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"[INFO] Metadata written: {meta_path}")


# ---------------------------------------------------

if __name__ == "__main__":
    process_catalogues()
