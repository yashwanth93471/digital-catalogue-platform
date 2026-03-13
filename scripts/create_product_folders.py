"""
Organize product images into named product folders.

Reads product_names.json and images from images_grouped/,
then creates:
    products/acacia_white/thumbnail.jpg
    products/acacia_white/application1.jpg

Usage:
    python scripts/create_product_folders.py
"""

import sys
import os
import re
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import IMAGES_GROUPED_DIR, PRODUCTS_DIR

NAMES_JSON = os.path.join(PRODUCTS_DIR, "product_names.json")


def sanitize_name(name: str) -> str:
    """Convert product name to lowercase_with_underscores."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def create_product_folders(names_json: str, grouped_dir: str, products_dir: str) -> int:
    """Copy images from grouped page folders into named product folders."""
    if not os.path.isfile(names_json):
        print(f"[ERROR] Product names JSON not found: {names_json}")
        return 0

    with open(names_json, "r", encoding="utf-8") as f:
        product_names = json.load(f)

    print(f"[INFO] Products: {len(product_names)}")
    print(f"[INFO] Source:   {grouped_dir}")
    print(f"[INFO] Output:   {products_dir}")
    print("-" * 55)

    created = 0

    for page_key, product_name in sorted(product_names.items()):
        folder_name = sanitize_name(product_name)
        page_dir = os.path.join(grouped_dir, page_key)

        if not os.path.isdir(page_dir):
            print(f"  {page_key} -> {folder_name}/ -- [SKIP] no images found")
            continue

        product_dir = os.path.join(products_dir, folder_name)
        os.makedirs(product_dir, exist_ok=True)

        files_copied = 0
        for filename in sorted(os.listdir(page_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src = os.path.join(page_dir, filename)
            dst = os.path.join(product_dir, filename)
            shutil.copy2(src, dst)
            files_copied += 1

        created += 1
        print(f"  {page_key} -> {folder_name}/ ({files_copied} file(s))")

    print("-" * 55)
    print(f"[DONE] {created} product folders created in products/")

    return created


if __name__ == "__main__":
    create_product_folders(NAMES_JSON, IMAGES_GROUPED_DIR, PRODUCTS_DIR)
