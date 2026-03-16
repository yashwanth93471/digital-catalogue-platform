"""
Extract product regions from PDF catalogues using PyMuPDF + CLIP.

Pipeline stage:
catalogues -> images_raw

Outputs per page folder:
  images_raw/<catalogue>/page_XXX/thumbnail.jpg
  images_raw/<catalogue>/page_XXX/application.jpg (optional)
  images_raw/<catalogue>/page_XXX/metadata.json
"""

import io
import json
import os
import sys
import time
from typing import Any

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CATALOGUES_DIR, IMAGES_RAW_DIR
import fitz  # PyMuPDF
from PIL import Image

from pipeline.clip_classifier import CLIPClassifier
from pipeline.composite_segmenter import crop_to_content, segment_composite_image


# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

MIN_WIDTH = 300
MIN_HEIGHT = 300

SQUARE_SIZE = (1000, 1000)
VERTICAL_SIZE = (900, 1200)

# Treat near-square textures as square to keep listing layout stable.
SQUARE_AR_MIN = 0.85
SQUARE_AR_MAX = 1.25

MIN_PRODUCT_SCORE = 0.20
MIN_PAGE_PRODUCT_SCORE = 0.30
MIN_FULLPAGE_TEXTURE_SCORE = 0.45
LARGE_IMAGE_COVERAGE = 0.55


# ---------------------------------------------------
# Utility helpers
# ---------------------------------------------------


def _product_score(scores: dict[str, float]) -> float:
    return (
        scores.get("texture", 0.0)
        + scores.get("room_scene", 0.0)
        + scores.get("preview_tile", 0.0)
    )


def _infer_tile_structure(width: int, height: int) -> str:
    aspect_ratio = width / float(height)
    if SQUARE_AR_MIN <= aspect_ratio <= SQUARE_AR_MAX:
        return "square"
    if aspect_ratio < SQUARE_AR_MIN:
        return "vertical"
    return "horizontal"


def _normalize_thumbnail(image: Image.Image) -> tuple[Image.Image, tuple[int, int], str]:
    image = crop_to_content(image).convert("RGB")
    width, height = image.size
    tile_structure = _infer_tile_structure(width, height)

    # Horizontal slabs are normalized to square cards for consistent grids.
    target_size = VERTICAL_SIZE if tile_structure == "vertical" else SQUARE_SIZE
    target_w, target_h = target_size

    scale = max(target_w / float(width), target_h / float(height))
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))

    resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

    left = max((resized_w - target_w) // 2, 0)
    top = max((resized_h - target_h) // 2, 0)
    right = left + target_w
    bottom = top + target_h

    normalized = resized.crop((left, top, right, bottom))
    return normalized, target_size, tile_structure


def _render_page_image(page: fitz.Page, dpi: int = 180) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _extract_embedded_candidates(page: fitz.Page, doc: fitz.Document) -> list[dict[str, Any]]:
    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    candidates: list[dict[str, Any]] = []
    seen_xrefs: set[int] = set()

    for img_index, img in enumerate(page.get_images(full=True), start=1):
        xref = img[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            base_image = doc.extract_image(xref)
            pil_image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
        except Exception:
            continue

        width, height = pil_image.size
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            continue

        try:
            rects = page.get_image_rects(xref)
            if rects:
                r = rects[0]
                coverage = float((r.width * r.height) / page_area)
            else:
                coverage = 0.0
        except Exception:
            coverage = 0.0

        candidates.append(
            {
                "xref": xref,
                "index": img_index,
                "image": pil_image,
                "width": width,
                "height": height,
                "area": width * height,
                "coverage": coverage,
            }
        )

    return candidates


def _pick_from_embedded(candidates: list[dict[str, Any]], classifier: CLIPClassifier):
    scored: list[dict[str, Any]] = []
    for c in candidates:
        scores = classifier.classify(c["image"])
        c2 = dict(c)
        c2["scores"] = scores
        c2["product_score"] = _product_score(scores)
        c2["texture_score"] = scores.get("texture", 0.0) + 0.5 * scores.get("preview_tile", 0.0)
        c2["scene_score"] = scores.get("room_scene", 0.0)
        scored.append(c2)

    filtered = [c for c in scored if c["product_score"] >= MIN_PRODUCT_SCORE]
    if not filtered:
        return None, None, "embedded_none"

    textures = sorted(
        filtered,
        key=lambda c: (c["texture_score"], c["area"]),
        reverse=True,
    )
    scenes = sorted(
        filtered,
        key=lambda c: (c["scene_score"], c["area"]),
        reverse=True,
    )

    thumbnail = textures[0] if textures else None
    application = None
    for c in scenes:
        if thumbnail is None or c["xref"] != thumbnail["xref"]:
            application = c
            break

    if thumbnail is None:
        thumbnail = max(filtered, key=lambda c: (c["product_score"], c["area"]))

    # Avoid false application assignments.
    if application is not None and application["scene_score"] < 0.25:
        application = None

    return thumbnail, application, "embedded"


def _save_jpg(image: Image.Image, path: str) -> None:
    image.convert("RGB").save(path, "JPEG", quality=95)


# ---------------------------------------------------
# Main extraction logic
# ---------------------------------------------------


def extract_images(pdf_path: str, catalogue_name: str, classifier: CLIPClassifier | None = None):
    if classifier is None:
        classifier = CLIPClassifier()

    output_dir = os.path.join(IMAGES_RAW_DIR, catalogue_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] Processing: {os.path.basename(pdf_path)}")

    start = time.time()
    doc = fitz.open(pdf_path)

    extracted_pages = 0
    metadata: list[dict[str, Any]] = []

    for page_idx in tqdm(range(len(doc)), desc="Pages"):
        page_num = page_idx + 1
        page = doc[page_idx]
        page_dir = os.path.join(output_dir, f"page_{page_num:03d}")

        candidates = _extract_embedded_candidates(page, doc)
        large_covering = any(c["coverage"] >= LARGE_IMAGE_COVERAGE for c in candidates)

        thumb_img = None
        app_img = None
        method = None

        # Prefer segmentation for flattened/composite pages.
        if len(candidates) <= 1 or large_covering:
            page_img = _render_page_image(page, dpi=180)
            page_scores = classifier.classify(page_img)

            if _product_score(page_scores) >= MIN_PAGE_PRODUCT_SCORE:
                seg = segment_composite_image(page_img, classifier)
                method = f"segment_{seg['method']}"
                thumb_img = seg.get("thumbnail")
                app_img = seg.get("application")

                # Do not save full-page fallback unless it is confidently texture-like.
                if seg.get("method") == "whole_page" and page_scores.get("texture", 0.0) < MIN_FULLPAGE_TEXTURE_SCORE:
                    thumb_img = None
                    app_img = None
            else:
                method = "skip_low_product_score"

        # Fallback to embedded-image selection.
        if thumb_img is None:
            thumb_c, app_c, emb_method = _pick_from_embedded(candidates, classifier)
            method = emb_method if method is None else f"{method}+{emb_method}"
            if thumb_c is not None:
                thumb_img = thumb_c["image"]
            if app_c is not None:
                app_img = app_c["image"]

        # If no thumbnail, skip page as non-product / undecidable.
        if thumb_img is None:
            continue

        os.makedirs(page_dir, exist_ok=True)

        norm_thumb, target_size, tile_structure = _normalize_thumbnail(thumb_img)
        thumb_path = os.path.join(page_dir, "thumbnail.jpg")
        _save_jpg(norm_thumb, thumb_path)

        has_application = app_img is not None
        app_size = None

        if has_application:
            app_clean = crop_to_content(app_img)
            app_path = os.path.join(page_dir, "application.jpg")
            _save_jpg(app_clean, app_path)
            app_size = f"{app_clean.size[0]}x{app_clean.size[1]}"

        page_meta = {
            "catalogue": catalogue_name,
            "page": page_num,
            "method": method,
            "has_thumbnail": True,
            "has_application": has_application,
            "thumbnail_size": f"{target_size[0]}x{target_size[1]}",
            "tile_structure": tile_structure,
        }
        if app_size:
            page_meta["application_size"] = app_size

        with open(os.path.join(page_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(page_meta, f, indent=2, ensure_ascii=False)

        metadata.append(page_meta)
        extracted_pages += 1

    doc.close()

    summary_path = os.path.join(output_dir, "catalogue_metadata.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"[DONE] Extracted {extracted_pages} page folders in {elapsed:.2f}s to {catalogue_name}/")

    return metadata


# ---------------------------------------------------
# Batch catalogue processing
# ---------------------------------------------------


def process_catalogues(target_pdfs: list[str] | None = None):
    if target_pdfs:
        pdfs = [p for p in target_pdfs if p.lower().endswith(".pdf")]
    else:
        pdfs = sorted([f for f in os.listdir(CATALOGUES_DIR) if f.lower().endswith(".pdf")])

    if not pdfs:
        print("[WARN] No PDFs found in catalogues/")
        return

    print(f"[INFO] Processing {len(pdfs)} PDF(s): {', '.join(pdfs)}\n")

    classifier = CLIPClassifier()
    all_metadata: list[dict[str, Any]] = []

    for pdf in pdfs:
        path = pdf if os.path.isabs(pdf) else os.path.join(CATALOGUES_DIR, pdf)
        if not os.path.exists(path):
            print(f"[WARN] Skipping missing PDF: {pdf}")
            continue

        catalogue_name = os.path.splitext(os.path.basename(pdf))[0]
        metadata = extract_images(path, catalogue_name, classifier=classifier)
        all_metadata.extend(metadata)

    meta_path = os.path.join(IMAGES_RAW_DIR, "images_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Metadata written: {meta_path}")
    print(f"[INFO] Total extracted page folders: {len(all_metadata)}")


# ---------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    process_catalogues(args if args else None)
