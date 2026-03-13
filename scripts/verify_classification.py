"""
Verification script for select_thumbnail.py classifier.

Runs the classifier on raw extracted images (without renaming/moving)
and prints a detailed report showing how each image would be classified.

Uses ABSOLUTE classification: is_texture based on score >= TEXTURE_THRESHOLD
(not relative "highest score wins").

Usage:
    python scripts/verify_classification.py
    python scripts/verify_classification.py images_raw/sample
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from scripts.select_thumbnail import classify_image, VALID_EXTENSIONS, TEXTURE_THRESHOLD
from config.settings import IMAGES_RAW_DIR


def verify(image_dir: str) -> None:
    """Classify all images and print a verification report."""

    if not os.path.isdir(image_dir):
        print(f"[ERROR] Not found: {image_dir}")
        return

    files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    )

    if not files:
        print(f"[WARN] No images in {image_dir}")
        return

    # Group files by page
    pages: dict[str, list[str]] = {}
    for f in files:
        m = re.match(r"(page_\d+)", f)
        key = m.group(1) if m else "unknown"
        pages.setdefault(key, []).append(f)

    header = (
        f"{'File':<40s} {'Size':>11s} {'EdgeD':>7s} {'EdgeU':>7s} "
        f"{'Entropy':>7s} {'Repeat':>7s} {'SpatV':>7s} {'ColDiv':>7s} "
        f"{'Score':>7s}  Class"
    )

    print("=" * 120)
    print(f"  CLASSIFICATION VERIFICATION - {image_dir}")
    print(f"  Texture threshold: {TEXTURE_THRESHOLD}")
    print("=" * 120)
    print()

    total_texture = 0
    total_scene = 0

    for page_key in sorted(pages):
        page_files = pages[page_key]
        print(f"-- {page_key} ({len(page_files)} image{'s' if len(page_files)>1 else ''}) " + "-" * 80)
        print(header)
        print("-" * 120)

        results = {}
        for f in page_files:
            fp = os.path.join(image_dir, f)
            r = classify_image(fp)
            results[f] = r

        ranked = sorted(results.items(), key=lambda x: x[1].get("texture_score", -999), reverse=True)

        for f, r in ranked:
            if "error" in r:
                print(f"  {f:<40s}  ** {r['error']} **")
                continue

            # Absolute classification
            label = "TEXTURE" if r["is_texture"] else "SCENE"
            if r["is_texture"]:
                total_texture += 1
            else:
                total_scene += 1

            size_str = f"{r['width']}x{r['height']}"
            print(
                f"  {f:<40s} {size_str:>11s} {r['edge_density']:>7.4f} "
                f"{r['edge_uniformity']:>7.4f} {r['entropy']:>7.3f} "
                f"{r['repetition']:>7.4f} {r['spatial_var']:>7.2f} "
                f"{r['color_diversity']:>7.2f} {r['texture_score']:>7.2f}  "
                f"<- {label}"
            )

        print()

    print("=" * 120)
    print(f"  SUMMARY: {total_texture} textures (thumbnail), {total_scene} scenes (application)")
    print("=" * 120)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # Default: first subfolder in images_raw
        subs = [d for d in os.listdir(IMAGES_RAW_DIR)
                if os.path.isdir(os.path.join(IMAGES_RAW_DIR, d))]
        if subs:
            target = os.path.join(IMAGES_RAW_DIR, sorted(subs)[0])
        else:
            target = IMAGES_RAW_DIR

    verify(target)
