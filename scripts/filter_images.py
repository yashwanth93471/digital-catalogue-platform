"""
Advanced filtering for catalogue images.

Removes:
1. Corrupted images
2. Very small decorative elements
3. Page background overlays
4. Extremely low-information images

Keeps:
• Solid product textures
• Slab previews
• Application images

Pipeline stage:
images_grouped → cleaned images_grouped
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import IMAGES_GROUPED_DIR


# ---------------------------------------------------
# Thresholds
# ---------------------------------------------------

MIN_AREA = 300 * 300
BG_UNIFORM_RATIO = 0.99
BG_MIN_AREA = 1_500_000

MIN_EDGE_DENSITY = 0.002
MIN_ENTROPY = 2.5

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------
# Image metrics
# ---------------------------------------------------

def image_entropy(gray):
    hist = np.histogram(gray, bins=256)[0]
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def edge_density(gray):
    edges = cv2.Canny(gray, 50, 150)
    return np.sum(edges > 0) / edges.size


def white_ratio(gray):
    return float(np.sum(gray > 240)) / gray.size


def black_ratio(gray):
    return float(np.sum(gray < 15)) / gray.size


# ---------------------------------------------------
# Corruption check
# ---------------------------------------------------

def is_corrupted(path):
    try:
        Image.open(path).verify()
        return False
    except Exception:
        return True


# ---------------------------------------------------
# Core detection
# ---------------------------------------------------

def is_invalid_image(img_path):

    if is_corrupted(img_path):
        return True, "corrupted"

    img = cv2.imread(img_path)

    if img is None:
        return True, "unreadable"

    h, w = img.shape[:2]
    area = h * w

    if area < MIN_AREA:
        return True, f"too_small ({w}x{h})"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    wr = white_ratio(gray)
    br = black_ratio(gray)

    if area >= BG_MIN_AREA and (wr > BG_UNIFORM_RATIO or br > BG_UNIFORM_RATIO):
        return True, "page_background"

    ent = image_entropy(gray)
    ed = edge_density(gray)

    if ent < MIN_ENTROPY and ed < MIN_EDGE_DENSITY:
        return True, "low_information"

    aspect = w / max(h,1)

    if aspect > 5 or aspect < 0.2:
        return True, "banner_or_icon"

    return False, "valid"


# ---------------------------------------------------
# Folder filtering
# ---------------------------------------------------

def filter_folder(folder):

    valid = []
    removed = []

    for fname in sorted(os.listdir(folder)):

        if os.path.splitext(fname)[1].lower() not in VALID_EXTENSIONS:
            continue

        fpath = os.path.join(folder, fname)

        invalid, reason = is_invalid_image(fpath)

        if invalid:
            removed.append((fpath, reason))
            os.remove(fpath)
        else:
            valid.append(fpath)

    return valid, removed


# ---------------------------------------------------
# Main pipeline
# ---------------------------------------------------

def filter_grouped_images(grouped_dir):

    if not os.path.isdir(grouped_dir):
        print(f"[ERROR] Folder not found: {grouped_dir}")
        return

    pages = sorted(
        d for d in os.listdir(grouped_dir)
        if os.path.isdir(os.path.join(grouped_dir, d))
    )

    total_valid = 0
    total_removed = 0

    print(f"[INFO] Filtering {len(pages)} page folders")
    print("-" * 60)

    for page in pages:

        page_path = os.path.join(grouped_dir, page)

        valid, removed = filter_folder(page_path)

        total_valid += len(valid)
        total_removed += len(removed)

        for fpath, reason in removed:
            print(f"{page}/ {os.path.basename(fpath)} -> removed ({reason})")

    print("-" * 60)
    print(f"[DONE] {total_valid} valid images")
    print(f"[DONE] {total_removed} removed images")


if __name__ == "__main__":

    target = sys.argv[1] if len(sys.argv) > 1 else IMAGES_GROUPED_DIR

    filter_grouped_images(target)
