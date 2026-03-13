"""
Filter out invalid images from extracted catalogue pages.

Detects and removes:
  1. Unreadable / corrupted images
  2. Very small images (icons, decorative elements)
  3. Pure white or black page-background overlays (>99% uniform
     AND large enough to be a page background, not a product texture)

Solid-color product textures (Pure White, Graphite Black, etc.) are
intentionally KEPT — the classifier handles them downstream.

Usage:
    python scripts/filter_images.py                      # filters images_grouped/
    python scripts/filter_images.py path/to/grouped_dir  # custom folder

Output:
    Removes invalid images in-place and prints a summary.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from config.settings import IMAGES_GROUPED_DIR

# ── Thresholds ──────────────────────────────────────────────────
MIN_AREA = 300 * 300             # minimum pixel area (width * height)
# Page-background detection: only flag as background if BOTH
# the uniformity ratio is extreme AND the image is big enough
# to be a page-size overlay (not a product texture slab).
BG_UNIFORM_RATIO = 0.99          # >99% of pixels match dominant color
BG_MIN_AREA = 1_500_000          # ~1500x1000 — larger than any slab preview

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Detection functions ────────────────────────────────────────

def _white_ratio(gray: np.ndarray) -> float:
    return float(np.sum(gray > 240)) / gray.size

def _black_ratio(gray: np.ndarray) -> float:
    return float(np.sum(gray < 15)) / gray.size


# ── Core filter logic ──────────────────────────────────────────

def is_invalid_image(img_path: str) -> tuple[bool, str, dict]:
    """Check whether an image should be discarded.

    Returns:
        (is_invalid, reason, metrics)
    """
    img = cv2.imread(img_path)
    if img is None:
        return True, "unreadable", {}

    h, w = img.shape[:2]
    area = h * w

    # Check 1: too small (preview swatches, icons, decorative elements)
    if area < MIN_AREA:
        return True, f"too_small ({w}x{h})", {"width": w, "height": h}

    # Check 2: page-background overlay — only if large enough AND
    # nearly 100% uniform.  Small solid-color product textures are kept.
    if area >= BG_MIN_AREA:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        wr = _white_ratio(gray)
        br = _black_ratio(gray)
        if wr > BG_UNIFORM_RATIO:
            return True, f"white page background ({wr:.1%})", {"width": w, "height": h, "white_ratio": round(wr, 3)}
        if br > BG_UNIFORM_RATIO:
            return True, f"black page background ({br:.1%})", {"width": w, "height": h, "black_ratio": round(br, 3)}

    return False, "valid", {"width": w, "height": h}


def filter_folder(folder: str, *, delete: bool = True) -> tuple[list[str], list[str]]:
    """Filter all images in a single folder.

    Returns:
        (valid_paths, removed_paths)
    """
    valid = []
    removed = []

    for fname in sorted(os.listdir(folder)):
        if os.path.splitext(fname)[1].lower() not in VALID_EXTENSIONS:
            continue

        fpath = os.path.join(folder, fname)
        invalid, reason, _ = is_invalid_image(fpath)

        if invalid:
            removed.append((fpath, reason))
            if delete:
                os.remove(fpath)
        else:
            valid.append(fpath)

    return valid, removed


def filter_grouped_images(grouped_dir: str, *, delete: bool = True) -> dict:
    """Filter invalid images from all page subfolders.

    Returns:
        Summary dict with counts per page and totals.
    """
    if not os.path.isdir(grouped_dir):
        print(f"[ERROR] Folder not found: {grouped_dir}")
        return {}

    page_dirs = sorted(
        d for d in os.listdir(grouped_dir)
        if os.path.isdir(os.path.join(grouped_dir, d))
    )

    total_valid = 0
    total_removed = 0
    summary = {}

    print(f"[INFO] Filtering invalid images in {len(page_dirs)} page folders")
    print("-" * 60)

    for page_name in page_dirs:
        page_path = os.path.join(grouped_dir, page_name)
        valid, removed = filter_folder(page_path, delete=delete)

        total_valid += len(valid)
        total_removed += len(removed)

        summary[page_name] = {
            "valid": len(valid),
            "removed": len(removed),
        }

        if removed:
            action = "deleted" if delete else "flagged"
            for fpath, reason in removed:
                print(f"  {page_name}/ {os.path.basename(fpath)} -> {action} ({reason})")

    print("-" * 60)
    print(f"[DONE] {total_valid} valid, {total_removed} removed")

    return {
        "total_valid": total_valid,
        "total_removed": total_removed,
        "pages": summary,
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else IMAGES_GROUPED_DIR
    filter_grouped_images(target, delete=True)
