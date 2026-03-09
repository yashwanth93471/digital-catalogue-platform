"""
Group extracted images by page number into subfolders.

Usage:
    python scripts/group_images.py
"""

import sys
import os
import re
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import IMAGES_RAW_DIR, IMAGES_GROUPED_DIR

# Matches: page_005_img_01.jpeg
PAGE_PATTERN = re.compile(r"^page_(\d+)_img_\d+\.\w+$", re.IGNORECASE)


def group_images(input_dir: str, output_dir: str) -> dict:
    """Group images from input_dir into page subfolders in output_dir."""
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input folder not found: {input_dir}")
        return {}

    files = sorted(f for f in os.listdir(input_dir) if PAGE_PATTERN.match(f))

    if not files:
        print("[WARN] No matching images found in images_raw/")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    groups = {}
    for filename in files:
        page_num = PAGE_PATTERN.match(filename).group(1)
        groups.setdefault(page_num, []).append(filename)

    print(f"[INFO] Found {len(files)} images across {len(groups)} pages")
    print("-" * 50)

    for page_num, filenames in sorted(groups.items()):
        page_dir = os.path.join(output_dir, f"page_{page_num}")
        os.makedirs(page_dir, exist_ok=True)

        for idx, filename in enumerate(filenames, start=1):
            ext = os.path.splitext(filename)[1]
            src = os.path.join(input_dir, filename)
            dst = os.path.join(page_dir, f"image{idx}{ext}")
            shutil.copy2(src, dst)

        print(f"  page_{page_num}/ — {len(filenames)} image(s)")

    print("-" * 50)
    print(f"[DONE] Grouped into: {output_dir}")

    return groups


if __name__ == "__main__":
    group_images(IMAGES_RAW_DIR, IMAGES_GROUPED_DIR)
