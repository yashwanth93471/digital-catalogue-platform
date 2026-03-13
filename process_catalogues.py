"""
Catalogue PDF Automation Pipeline (CLIP Edition)
=================================================
Batch-processes tile / granite catalogue PDFs and extracts per product:

    thumbnail.jpg    -- pure texture slab
    application.jpg  -- room scene showing the texture installed
    metadata.json    -- product name, pages, resolutions, CLIP confidence

Technology stack
----------------
  PyMuPDF          PDF image + text extraction with coordinates
  OpenCV / NumPy   Image preprocessing and blank detection
  Pillow           Format conversion, resizing, OCR enhancement
  CLIP (HF)        Zero-shot classification (texture / room scene / logo ...)
                   + embedding similarity for texture-scene matching
  LayoutParser     (optional) DL document-layout region detection
  Tesseract OCR    Fallback product-name extraction from rendered headers

Usage
-----
  python process_catalogues.py                          # all PDFs in catalogues/
  python process_catalogues.py catalogues/sample.pdf    # single PDF
  python process_catalogues.py catalogues/              # folder of PDFs
"""

import glob
import logging
import os
import sys
import time

# ── project root on sys.path ──────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import CATALOGUES_DIR, PRODUCTS_DIR, TESSERACT_PATH, LOGS_DIR
from pipeline.product_assembler import process_pdf
from pipeline.clip_classifier import CLIPClassifier


# ── logging ───────────────────────────────────────────────────

def _setup_logging() -> str:
    os.makedirs(LOGS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"pipeline_clip_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_file


# ── PDF discovery ─────────────────────────────────────────────

def _find_pdfs(target: str) -> list[str]:
    if os.path.isfile(target) and target.lower().endswith(".pdf"):
        return [target]
    if os.path.isdir(target):
        return sorted(glob.glob(os.path.join(target, "*.pdf")))
    return []


# ── main ──────────────────────────────────────────────────────

def main():
    log_file = _setup_logging()
    logger = logging.getLogger(__name__)

    # Resolve input path
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if not os.path.isabs(target):
            target = os.path.join(PROJECT_ROOT, target)
    else:
        target = CATALOGUES_DIR

    pdfs = _find_pdfs(target)
    if not pdfs:
        logger.error(f"No PDF files found at: {target}")
        sys.exit(1)

    # LayoutParser availability
    try:
        import layoutparser
        lp_status = f"available (v{layoutparser.__version__})"
    except ImportError:
        lp_status = "not installed (using PyMuPDF text-block fallback)"

    logger.info("=" * 60)
    logger.info("CATALOGUE PDF AUTOMATION PIPELINE (CLIP)")
    logger.info("=" * 60)
    logger.info(f"  Input        : {target}")
    logger.info(f"  PDFs found   : {len(pdfs)}")
    logger.info(f"  Output       : {PRODUCTS_DIR}")
    logger.info(f"  Log          : {log_file}")
    logger.info(f"  LayoutParser : {lp_status}")
    logger.info(f"  Tesseract    : {TESSERACT_PATH}")

    # Load CLIP model once for all PDFs
    import transformers
    transformers.logging.set_verbosity_error()
    import logging as _logging
    _logging.getLogger("httpx").setLevel(_logging.WARNING)
    _logging.getLogger("huggingface_hub").setLevel(_logging.WARNING)

    logger.info("Loading CLIP model (one-time)...")
    classifier = CLIPClassifier()
    logger.info("CLIP model ready.")

    total_start = time.time()
    all_results: list[dict] = []

    for pdf_path in pdfs:
        pdf_start = time.time()
        try:
            results = process_pdf(pdf_path, PRODUCTS_DIR, TESSERACT_PATH, classifier)
            all_results.extend(results)
            elapsed = time.time() - pdf_start
            logger.info(
                f"\nCompleted {os.path.basename(pdf_path)}: "
                f"{len(results)} products in {elapsed:.1f}s"
            )
        except Exception:
            logger.exception(f"Failed to process {pdf_path}")

    # ── summary ───────────────────────────────────────────────
    total_time = time.time() - total_start
    with_thumb = sum(1 for r in all_results if r["has_thumbnail"])
    with_app = sum(1 for r in all_results if r["has_application"])
    complete = sum(
        1 for r in all_results if r["has_thumbnail"] and r["has_application"]
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  PDFs processed     : {len(pdfs)}")
    logger.info(f"  Total products     : {len(all_results)}")
    logger.info(f"  With thumbnail     : {with_thumb}")
    logger.info(f"  With application   : {with_app}")
    logger.info(f"  Complete (both)    : {complete}")
    logger.info(f"  Total time         : {total_time:.1f}s")
    logger.info(f"  Output             : {PRODUCTS_DIR}")


if __name__ == "__main__":
    main()
