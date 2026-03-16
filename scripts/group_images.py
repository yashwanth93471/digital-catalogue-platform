"""
Group extracted images by page number into subfolders.

Enhanced version with:
• duplicate detection
• metadata collection
• image sorting
• better grouping preparation

Usage:
    python scripts/group_images.py
"""

import sys
import os
import re
import shutil
import hashlib
import json

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import IMAGES_RAW_DIR, IMAGES_GROUPED_DIR


PAGE_PATTERN = re.compile(r"^page_(\d+)_img_\d+\.\w+$", re.IGNORECASE)

METADATA_FILE = "group_metadata.json"


def file_hash(path):
    """Compute hash to detect duplicate images."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def image_area(path):
    """Return image area for sorting."""
    img = cv2.imread(path)
    if img is None:
        return 0
    h, w = img.shape[:2]
    return w * h


def group_images(input_dir, output_dir):

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input folder not found: {input_dir}")
        return {}

    files = sorted(f for f in os.listdir(input_dir) if PAGE_PATTERN.match(f))

    if not files:
        print("[WARN] No matching images found in images_raw/")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    groups = {}
    hashes = set()

    for filename in files:

        page_num = PAGE_PATTERN.match(filename).group(1)

        src_path = os.path.join(input_dir, filename)

        h = file_hash(src_path)

        # skip duplicates
        if h in hashes:
            continue

        hashes.add(h)

        groups.setdefault(page_num, []).append(filename)

    print(f"[INFO] {len(files)} images detected")
    print(f"[INFO] {len(groups)} page groups created")
    print("-" * 60)

    metadata = {}

    for page_num, filenames in sorted(groups.items()):

        page_dir = os.path.join(output_dir, f"page_{page_num}")
        os.makedirs(page_dir, exist_ok=True)

        # sort by image size
        filenames.sort(
            key=lambda f: image_area(os.path.join(input_dir, f)),
            reverse=True
        )

        page_meta = []

        for idx, filename in enumerate(filenames, start=1):

            ext = os.path.splitext(filename)[1]

            src = os.path.join(input_dir, filename)

            dst = os.path.join(page_dir, f"image{idx}{ext}")

            shutil.copy2(src, dst)

            area = image_area(src)

            page_meta.append({
                "file": f"image{idx}{ext}",
                "area": area
            })

        metadata[f"page_{page_num}"] = page_meta

        print(f"page_{page_num}/ -> {len(filenames)} images")

    meta_path = os.path.join(output_dir, METADATA_FILE)

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("-" * 60)
    print(f"[DONE] Grouping complete")
    print(f"[INFO] Metadata saved: {meta_path}")

    return groups


if __name__ == "__main__":
    group_images(IMAGES_RAW_DIR, IMAGES_GROUPED_DIR)
