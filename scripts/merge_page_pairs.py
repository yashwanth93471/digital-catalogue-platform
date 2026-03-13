"""
Merge page pairs in catalogues where products span two pages.

Many tile/slab catalogues use a two-page layout per product:
  - Page N   (odd):  Full-page texture slab = thumbnail only
  - Page N+1 (even): Room scene (application) + small variant tiles

This script detects such pairs AFTER thumbnail classification and merges
them into a single page folder:
  - Thumbnail stays from the texture page
  - Only the largest room-scene application is kept from the application page
  - Small variant tiles on the application page are discarded
  - The application page folder is removed

Detection is adaptive (works for different catalogue layouts):
  - A page with ONLY a thumbnail and no applications is a "thumbnail-only" page
  - If the next page has applications, they belong to the same product
  - Pages that already have both thumbnail + applications are left alone

Usage:
    python scripts/merge_page_pairs.py
    python scripts/merge_page_pairs.py path/to/grouped_dir
"""

import sys
import os
import shutil
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import IMAGES_GROUPED_DIR

PAGE_NUM_RE = re.compile(r"page_(\d+)")


def _get_page_contents(page_dir: str) -> dict:
    """Analyze contents of a classified page folder.

    Returns dict with:
        has_thumbnail: bool
        has_applications: bool
        application_files: list of filenames
        application_count: int
    """
    files = os.listdir(page_dir) if os.path.isdir(page_dir) else []

    has_thumbnail = "thumbnail.jpg" in files
    app_files = sorted(f for f in files if f.startswith("application") and f.endswith(".jpg"))

    return {
        "has_thumbnail": has_thumbnail,
        "has_applications": len(app_files) > 0,
        "application_files": app_files,
        "application_count": len(app_files),
    }


def _find_largest_application(page_dir: str, app_files: list) -> str | None:
    """Find the largest application image by pixel area."""
    if not app_files:
        return None

    import cv2
    best_file = None
    best_area = 0

    for f in app_files:
        fpath = os.path.join(page_dir, f)
        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w = img.shape[:2]
        area = h * w
        if area > best_area:
            best_area = area
            best_file = f

    return best_file


def merge_page_pairs(grouped_dir: str) -> tuple[int, set]:
    """Detect and merge page pairs where products span two pages.

    Returns:
        (merge_count, merged_app_pages) -- number of merges and set of
        page keys that were merged as application pages (to skip in OCR).
    """
    if not os.path.isdir(grouped_dir):
        print(f"[ERROR] Folder not found: {grouped_dir}")
        return 0, set()

    # Get all page directories sorted by page number
    page_dirs = sorted(
        d for d in os.listdir(grouped_dir)
        if os.path.isdir(os.path.join(grouped_dir, d)) and PAGE_NUM_RE.match(d)
    )

    if len(page_dirs) < 2:
        print("[INFO] Not enough pages for pair detection")
        return 0, set()

    print(f"[INFO] Scanning {len(page_dirs)} page folders for page pairs")
    print("-" * 60)

    merge_count = 0
    merged_app_pages = set()
    skip_next = False

    for i in range(len(page_dirs) - 1):
        if skip_next:
            skip_next = False
            continue

        current_name = page_dirs[i]
        next_name = page_dirs[i + 1]
        current_path = os.path.join(grouped_dir, current_name)
        next_path = os.path.join(grouped_dir, next_name)

        current_info = _get_page_contents(current_path)
        next_info = _get_page_contents(next_path)

        # Detect page pair: current has thumbnail only, next has applications
        if (current_info["has_thumbnail"]
                and not current_info["has_applications"]
                and next_info["has_applications"]):

            # Find the largest application (room scene) from next page
            largest_app = _find_largest_application(next_path, next_info["application_files"])

            if largest_app:
                # Copy only the largest application to the thumbnail page
                src = os.path.join(next_path, largest_app)
                dst = os.path.join(current_path, "application1.jpg")
                shutil.copy2(src, dst)

                # Record the merged application page for skipping in OCR
                merged_app_pages.add(next_name)

                # Remove the application page folder entirely
                shutil.rmtree(next_path)

                merge_count += 1
                skip_next = True

                extra_apps = next_info["application_count"] - 1
                discarded_str = f" [{extra_apps} variant tiles discarded]" if extra_apps > 0 else ""
                print(f"  {current_name} + {next_name} -> merged"
                      f" (thumbnail + application1.jpg){discarded_str}")

                # Also merge thumbnail from next page if it had one
                # (application pages sometimes have their own thumbnail which
                # should be discarded since the real thumbnail is on current page)
            else:
                print(f"  {current_name} + {next_name} -> no large application found, skipping")

    print("-" * 60)
    print(f"[DONE] {merge_count} page pairs merged")

    return merge_count, merged_app_pages


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else IMAGES_GROUPED_DIR
    merge_count, merged = merge_page_pairs(target)
    if merged:
        print(f"Merged app pages to skip in OCR: {sorted(merged)}")
