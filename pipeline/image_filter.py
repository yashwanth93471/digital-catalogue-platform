"""
Filter out unwanted images before CLIP classification.

Three filtering layers (cheapest first):
  1. **Blank / solid-colour** -- nearly-uniform pixel values (page
     backgrounds, white sheets).
  2. **Repeated across pages** -- perceptual-hash frequency analysis
     detects logos / watermarks that appear on > 40 % of pages.
  3. **Too-small images** -- anything below MIN_AREA is an icon, QR
     code, or decorative element.

All filtering runs *before* CLIP, saving expensive GPU/CPU inference
on images that are clearly not product content.
"""

import numpy as np
from PIL import Image

MIN_AREA = 300 * 300
BG_UNIFORM_RATIO = 0.95          # > 95 % same colour -> background
LOGO_PAGE_RATIO = 0.50           # appears on > 50 % of pages -> logo
DHASH_HAMMING_THRESHOLD = 5      # max hamming distance for "same image"


# ── helpers ────────────────────────────────────────────────────

def _hamming(h1: str, h2: str) -> int:
    return sum(a != b for a, b in zip(h1, h2))


def is_blank_or_solid(pil_image: Image.Image) -> bool:
    """True when the image is a near-uniform colour (page background).

    Conservative thresholds: only catches truly white/black sheets,
    NOT low-variance stone/marble textures.
    """
    arr = np.array(pil_image)
    gray = np.mean(arr, axis=2) if arr.ndim == 3 else arr.astype(float)

    # > 98% white pixels → blank page
    white = float(np.sum(gray > 245)) / gray.size
    if white > 0.98:
        return True

    # > 98% black pixels → blank page
    black = float(np.sum(gray < 10)) / gray.size
    if black > 0.98:
        return True

    # Truly uniform (solid fill) – std < 2 catches printer-solid images
    # but NOT natural stone textures which typically have std > 2.
    if float(np.std(gray)) < 2.0:
        return True

    return False


# ── repeated-image detection ──────────────────────────────────

def detect_repeated_images(pages, page_ratio: float = LOGO_PAGE_RATIO) -> set[str]:
    """Return phash strings that appear on more than *page_ratio* of pages.

    Uses Hamming distance so slight JPEG variations still match.
    """
    # Collect (phash, page_num)
    hash_pages: dict[str, set[int]] = {}
    for page in pages:
        for img in page.images:
            found = False
            for existing_hash in hash_pages:
                if _hamming(img.phash, existing_hash) <= DHASH_HAMMING_THRESHOLD:
                    hash_pages[existing_hash].add(page.page_num)
                    found = True
                    break
            if not found:
                hash_pages[img.phash] = {page.page_num}

    total = len(pages) if pages else 1
    return {h for h, pgs in hash_pages.items() if len(pgs) > total * page_ratio}


def _matches_repeated(phash: str, repeated: set[str]) -> bool:
    for rh in repeated:
        if _hamming(phash, rh) <= DHASH_HAMMING_THRESHOLD:
            return True
    return False


# ── main filter entry point ───────────────────────────────────

def filter_page_images(page, repeated_hashes: set[str]) -> list:
    """Return only the images worth classifying with CLIP.

    Pages with exactly 2 large images are treated as product pages
    (one texture swatch + one room scene).  No filtering is applied
    so that solid-colour textures (pure white, black sparkle, etc.)
    are never mistakenly removed.
    """
    # Area-only candidates (cheapest check)
    area_ok = [img for img in page.images if img.area >= MIN_AREA]

    # 2-image pages: keep both unconditionally (product page layout)
    if len(area_ok) == 2:
        return area_ok

    # For other pages, apply repeated-image and blank filters
    candidates = [img for img in area_ok
                  if not _matches_repeated(img.phash, repeated_hashes)]
    if len(candidates) > 2:
        return [img for img in candidates if not is_blank_or_solid(img.image)]
    return candidates
