"""
Test to verify thumbnail resizing vs application original size preservation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import shutil
from scripts.extract_images import extract_images
from scripts.group_images import group_images
from scripts.filter_images import filter_grouped_images
from scripts.select_thumbnail import process_all
from config.settings import CATALOGUES_DIR, IMAGES_RAW_DIR, IMAGES_GROUPED_DIR

# Test with sample.pdf
pdf = os.path.join(CATALOGUES_DIR, "sample.pdf")
raw_dir = os.path.join(IMAGES_RAW_DIR, "test_resize")
grouped_dir = os.path.join(IMAGES_GROUPED_DIR, "test_resize")

# Clean test dirs
for d in [raw_dir, grouped_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)

print("="*80)
print("  RESIZE TEST: Thumbnails vs Applications")
print("="*80)

# Step 1: Extract
print("\n[1/4] Extracting images...")
extract_images(pdf, raw_dir)

# Step 2: Group
print("\n[2/4] Grouping by page...")
group_images(raw_dir, grouped_dir)

# Step 3: Filter blanks
print("\n[3/4] Filtering blank images...")
filter_grouped_images(grouped_dir, delete=True)

# Step 4: Classify and check sizes
print("\n[4/4] Classifying and checking output sizes...")
process_all(grouped_dir)

# Verify sizes
print("\n" + "="*80)
print("  SIZE VERIFICATION")
print("="*80)
print(f"\n{'Page':<12s} {'File':<20s} {'Type':<12s} {'Size (W×H)':<15s} {'Original Size':<15s}")
print("-"*80)

import re
for page_dir in sorted(os.listdir(grouped_dir)):
    page_path = os.path.join(grouped_dir, page_dir)
    if not os.path.isdir(page_path):
        continue
    
    # Get original sizes from raw
    raw_page_files = [f for f in os.listdir(raw_dir) if page_dir in f and f.endswith(('.jpg','.jpeg','.png'))]
    orig_sizes = {}
    for f in raw_page_files:
        img = cv2.imread(os.path.join(raw_dir, f))
        if img is not None:
            orig_sizes[f] = f"{img.shape[1]}×{img.shape[0]}"
    
    # Check output files
    for fname in sorted(os.listdir(page_path)):
        if not fname.endswith('.jpg'):
            continue
        
        fpath = os.path.join(page_path, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        size_str = f"{w}×{h}"
        
        if fname == "thumbnail.jpg":
            ftype = "THUMBNAIL"
            # Find which original file this was
            orig_size = "N/A"
            for orig_f, orig_s in orig_sizes.items():
                if len(orig_sizes) == 1:
                    orig_size = orig_s
                    break
        else:
            ftype = "APPLICATION"
            orig_size = "N/A"
        
        is_resized = "← RESIZED" if size_str == "1528×3150" else "← ORIGINAL"
        print(f"{page_dir:<12s} {fname:<20s} {ftype:<12s} {size_str:<15s} {orig_size:<15s} {is_resized}")

print("\n" + "="*80)
print("✓ Thumbnails (texture) should be 1528×3150")
print("✓ Applications (room) should keep original extracted size")
print("="*80)
