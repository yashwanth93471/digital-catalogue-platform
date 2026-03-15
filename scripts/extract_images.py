"""
Robust Catalogue Image Extraction Pipeline

Handles:
- embedded PDF images
- full page rendering
- granite / marble / tile texture detection

Pipeline:
catalogues → images_raw → tile_crops
"""

# ===============================
# Install dependencies (Colab)
# ===============================

# !pip install pymupdf pypdfium2 opencv-python pillow tqdm


# ===============================
# Imports
# ===============================

import os
import json
import fitz
import cv2
import numpy as np
import pypdfium2 as pdfium
from tqdm import tqdm
from PIL import Image


# ===============================
# Config
# ===============================

CATALOGUES_DIR = "catalogues"
IMAGES_RAW_DIR = "images_raw"
TILES_DIR = "tiles"

MIN_TILE_WIDTH = 200
MIN_TILE_HEIGHT = 200

os.makedirs(IMAGES_RAW_DIR, exist_ok=True)
os.makedirs(TILES_DIR, exist_ok=True)


# ===============================
# Extract embedded images
# ===============================

def extract_embedded_images(pdf_path):

    doc = fitz.open(pdf_path)

    metadata = []
    count = 0

    for page_index in range(len(doc)):

        page = doc.load_page(page_index)

        image_list = page.get_images(full=True)

        for img in image_list:

            xref = img[0]

            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]
            ext = base_image["ext"]

            filename = f"embedded_{count:05d}.{ext}"

            path = os.path.join(IMAGES_RAW_DIR, filename)

            with open(path, "wb") as f:
                f.write(image_bytes)

            metadata.append({
                "file": filename,
                "page": page_index + 1,
                "type": "embedded"
            })

            count += 1

    return metadata


# ===============================
# Render PDF pages
# ===============================

def render_pdf_pages(pdf_path):

    pdf = pdfium.PdfDocument(pdf_path)

    pages = []

    for i in range(len(pdf)):

        page = pdf[i]

        bitmap = page.render(scale=3)

        pil_image = bitmap.to_pil()

        pages.append(pil_image)

    return pages


# ===============================
# Detect tile textures
# ===============================

def detect_tiles(page_image, page_number):

    img = np.array(page_image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    tiles_metadata = []
    tile_count = 0

    for c in contours:

        x,y,w,h = cv2.boundingRect(c)

        if w > MIN_TILE_WIDTH and h > MIN_TILE_HEIGHT:

            crop = img[y:y+h, x:x+w]

            filename = f"tile_p{page_number}_{tile_count}.png"

            path = os.path.join(TILES_DIR, filename)

            cv2.imwrite(path, crop)

            tiles_metadata.append({
                "file": filename,
                "page": page_number,
                "type": "tile_crop"
            })

            tile_count += 1

    return tiles_metadata


# ===============================
# Process a single PDF
# ===============================

def process_pdf(pdf_path):

    print(f"\nProcessing: {os.path.basename(pdf_path)}")

    metadata = []

    # 1️⃣ Embedded images
    embedded = extract_embedded_images(pdf_path)

    metadata.extend(embedded)

    # 2️⃣ Page rendering
    pages = render_pdf_pages(pdf_path)

    # 3️⃣ Tile detection
    for i,page in enumerate(pages):

        tiles = detect_tiles(page, i+1)

        metadata.extend(tiles)

    return metadata


# ===============================
# Batch processing
# ===============================

def process_catalogues():

    pdfs = [
        f for f in os.listdir(CATALOGUES_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdfs:

        print("No PDFs found.")
        return

    all_metadata = []

    for pdf in tqdm(pdfs):

        pdf_path = os.path.join(CATALOGUES_DIR, pdf)

        metadata = process_pdf(pdf_path)

        all_metadata.extend(metadata)

    meta_file = os.path.join(IMAGES_RAW_DIR, "images_metadata.json")

    with open(meta_file, "w") as f:

        json.dump(all_metadata, f, indent=2)

    print(f"\nDone. Metadata saved to {meta_file}")


# ===============================
# Run pipeline
# ===============================

if __name__ == "__main__":

    process_catalogues()
