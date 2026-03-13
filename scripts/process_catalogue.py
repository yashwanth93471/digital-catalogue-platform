"""
Catalogue PDF Automation Pipeline
==================================
End-to-end pipeline that extracts product images from a PDF catalogue
and organizes them into named product folders with metadata.

Steps:
  1. Extract images from PDF
  2. Group images by page
  2.5. Filter blank/background images
  3. Select thumbnails (classify texture vs room scene)
  3.5. Merge page pairs (thumbnail page + application page -> single product)
  4. Extract product names (OCR, header-focused, brand-filtered)
  5. Create product folders
  6. Generate metadata

Usage:
    python scripts/process_catalogue.py catalogue.pdf
    python scripts/process_catalogue.py                  (defaults to catalogues/sample.pdf)
"""

import sys
import os
import time
import shutil
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    CATALOGUES_DIR, IMAGES_RAW_DIR, IMAGES_GROUPED_DIR, PRODUCTS_DIR, LOGS_DIR,
)

# ---- Import pipeline modules ----
from scripts.extract_images import extract_images
from scripts.group_images import group_images
from scripts.filter_images import filter_grouped_images
from scripts.extract_product_names import extract_all_names
from scripts.select_thumbnail import process_all as select_thumbnails
from scripts.merge_page_pairs import merge_page_pairs
from scripts.create_product_folders import create_product_folders, NAMES_JSON
from scripts.generate_metadata import generate_metadata

DEFAULT_PDF = os.path.join(CATALOGUES_DIR, "sample.pdf")


def setup_logging():
    """Configure logging to file and console."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"pipeline_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_file


def clean_output_dirs():
    """Keep all previous data - no cleaning performed."""
    # All folders (images_raw, images_grouped, products) are kept intact
    # Data accumulates across multiple PDF runs
    pass


def run_pipeline(pdf_path: str):
    """Execute the full catalogue processing pipeline."""
    log_file = setup_logging()
    logger = logging.getLogger(__name__)

    # Extract PDF name (without extension) to organize outputs
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Create PDF-specific subdirectories
    pdf_images_raw = os.path.join(IMAGES_RAW_DIR, pdf_name)
    pdf_images_grouped = os.path.join(IMAGES_GROUPED_DIR, pdf_name)
    pdf_products = os.path.join(PRODUCTS_DIR, pdf_name)
    pdf_names_json = os.path.join(pdf_products, "product_names.json")

    logger.info("=" * 60)
    logger.info("CATALOGUE PDF AUTOMATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Input PDF : {pdf_path}")
    logger.info(f"PDF name  : {pdf_name}")
    logger.info(f"Output    : {pdf_products}")
    logger.info(f"Log file  : {log_file}")

    if not os.path.isfile(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        sys.exit(1)

    pipeline_start = time.time()

    # ── Clean previous outputs ──────────────────────────────────
    logger.info("")
    logger.info("[0/7] Cleaning previous outputs...")
    clean_output_dirs()

    # ── Step 1: Extract images ──────────────────────────────────
    logger.info("")
    logger.info("[1/7] Extracting images from PDF...")
    step_start = time.time()
    image_count = extract_images(pdf_path, pdf_images_raw)
    logger.info(f"  -> {image_count} images extracted ({time.time() - step_start:.1f}s)")

    if image_count == 0:
        logger.error("No images extracted. Aborting.")
        sys.exit(1)

    # ── Step 2: Group images by page ────────────────────────────
    logger.info("")
    logger.info("[2/7] Grouping images by page...")
    step_start = time.time()
    groups = group_images(pdf_images_raw, pdf_images_grouped)
    logger.info(f"  -> {len(groups)} page groups ({time.time() - step_start:.1f}s)")

    # ── Step 2.5: Filter blank / background images ─────────────
    logger.info("")
    logger.info("[2.5/7] Filtering blank / background images...")
    step_start = time.time()
    filter_result = filter_grouped_images(pdf_images_grouped, delete=True)
    removed_count = filter_result.get("total_removed", 0)
    logger.info(f"  -> {removed_count} blank images removed ({time.time() - step_start:.1f}s)")

    # ── Step 3: Select thumbnails ───────────────────────────────
    logger.info("")
    logger.info("[3/7] Selecting thumbnails...")
    step_start = time.time()
    select_thumbnails(pdf_images_grouped)
    logger.info(f"  -> Thumbnails selected ({time.time() - step_start:.1f}s)")

    # ── Step 3.5: Merge page pairs ──────────────────────────────
    logger.info("")
    logger.info("[3.5/7] Merging page pairs (thumbnail + application pages)...")
    step_start = time.time()
    merge_count, merged_app_pages = merge_page_pairs(pdf_images_grouped)
    logger.info(f"  -> {merge_count} page pairs merged ({time.time() - step_start:.1f}s)")

    # ── Step 4: Extract product names (OCR) ─────────────────────
    logger.info("")
    logger.info("[4/7] Extracting product names (OCR, header-focused)...")
    step_start = time.time()
    product_names = extract_all_names(pdf_path, pdf_names_json,
                                       skip_pages=merged_app_pages)
    logger.info(f"  -> {len(product_names)} product names ({time.time() - step_start:.1f}s)")

    if not product_names:
        logger.error("No product names extracted. Aborting.")
        sys.exit(1)

    # ── Step 5: Create product folders ──────────────────────────
    logger.info("")
    logger.info("[5/7] Creating product folders...")
    step_start = time.time()
    folder_count = create_product_folders(pdf_names_json, pdf_images_grouped, pdf_products)
    logger.info(f"  -> {folder_count} product folders ({time.time() - step_start:.1f}s)")

    # ── Step 6: Generate metadata ───────────────────────────────
    logger.info("")
    logger.info("[6/7] Generating metadata...")
    step_start = time.time()
    meta_count = generate_metadata(pdf_path, pdf_names_json, pdf_products)
    logger.info(f"  -> {meta_count} metadata files ({time.time() - step_start:.1f}s)")

    # ── Summary ─────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  PDF name         : {pdf_name}")
    logger.info(f"  Images extracted : {image_count}")
    logger.info(f"  Page groups      : {len(groups)}")
    logger.info(f"  Page pairs merged: {merge_count}")
    logger.info(f"  Product names    : {len(product_names)}")
    logger.info(f"  Product folders  : {folder_count}")
    logger.info(f"  Metadata files   : {meta_count}")
    logger.info(f"  Total time       : {total_time:.1f}s")
    logger.info(f"  Output           : {pdf_products}")
    logger.info(f"  Log              : {log_file}")


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF

    # If user passes just a filename, look in catalogues/
    if not os.path.isabs(pdf_path) and not os.path.isfile(pdf_path):
        candidate = os.path.join(CATALOGUES_DIR, pdf_path)
        if os.path.isfile(candidate):
            pdf_path = candidate

    run_pipeline(pdf_path)
