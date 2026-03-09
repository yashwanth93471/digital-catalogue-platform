"""
Select the slab texture image as thumbnail from grouped images.

For each page folder in images_grouped/:
  - Analyzes images using edge density and spatial variance
  - The most uniform/texture-like image → thumbnail.jpg
  - Remaining images → application1.jpg, application2.jpg, ...

Usage:
    python scripts/select_thumbnail.py
"""

import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.settings import IMAGES_GROUPED_DIR

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def compute_texture_score(img_path: str) -> float:
    """
    Return a texture uniformity score (lower = more uniform/slab-like).

    Combines:
      - Edge density (Canny): room scenes have many edges (furniture, walls)
      - Spatial variance: slab textures are uniform across the image
      - Color std: slab textures have narrow color range
    """
    img = cv2.imread(img_path)
    if img is None:
        return float("inf")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / (h * w)

    # Spatial variance across 4x4 grid
    block_h, block_w = h // 4, w // 4
    block_means = []
    for r in range(4):
        for c in range(4):
            block = gray[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w]
            block_means.append(np.mean(block))
    spatial_var = np.std(block_means)

    # Color channel std
    color_std = np.mean([np.std(img[:, :, c]) for c in range(3)])

    # Combined score — lower means more slab-like
    score = (edge_density * 1000) + (spatial_var * 2) + (color_std * 0.5)
    return score


def select_thumbnail(page_dir: str) -> None:
    """Classify images in a page folder into thumbnail + applications."""
    files = sorted(
        f for f in os.listdir(page_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    )

    if not files:
        return

    page_name = os.path.basename(page_dir)

    if len(files) == 1:
        # Single image → it's the thumbnail
        src = os.path.join(page_dir, files[0])
        dst = os.path.join(page_dir, "thumbnail.jpg")
        if files[0] != "thumbnail.jpg":
            img = cv2.imread(src)
            cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            os.remove(src)
        print(f"  {page_name}/ → thumbnail.jpg (single image)")
        return

    # Multiple images → score each one
    scores = {}
    for f in files:
        path = os.path.join(page_dir, f)
        scores[f] = compute_texture_score(path)

    # Lowest score = most slab-like → thumbnail
    ranked = sorted(scores.items(), key=lambda x: x[1])
    thumbnail_file = ranked[0][0]
    application_files = [f for f, _ in ranked[1:]]

    # Rename thumbnail
    src = os.path.join(page_dir, thumbnail_file)
    dst = os.path.join(page_dir, "thumbnail.jpg")
    img = cv2.imread(src)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    os.remove(src)

    # Rename applications
    for idx, app_file in enumerate(application_files, start=1):
        src = os.path.join(page_dir, app_file)
        dst = os.path.join(page_dir, f"application{idx}.jpg")
        img = cv2.imread(src)
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        os.remove(src)

    app_names = ", ".join(f"application{i}.jpg" for i in range(1, len(application_files) + 1))
    print(f"  {page_name}/ → thumbnail.jpg + {app_names}")
    print(f"           scores: {thumbnail_file}={scores[thumbnail_file]:.1f} (slab) | others: {', '.join(f'{f}={scores[f]:.1f}' for f in application_files)}")


def process_all(grouped_dir: str) -> None:
    """Process all page folders."""
    if not os.path.isdir(grouped_dir):
        print(f"[ERROR] Folder not found: {grouped_dir}")
        return

    page_dirs = sorted(
        d for d in os.listdir(grouped_dir)
        if os.path.isdir(os.path.join(grouped_dir, d))
    )

    print(f"[INFO] Processing {len(page_dirs)} page folders")
    print("-" * 60)

    for page_dir_name in page_dirs:
        page_path = os.path.join(grouped_dir, page_dir_name)
        select_thumbnail(page_path)

    print("-" * 60)
    print("[DONE] Thumbnails selected.")


if __name__ == "__main__":
    process_all(IMAGES_GROUPED_DIR)
