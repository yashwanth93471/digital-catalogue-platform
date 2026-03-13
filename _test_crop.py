"""Quick check if crop_to_content now trims white borders properly."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from PIL import Image
from pipeline.composite_segmenter import crop_to_content

# Test on the Elevation region which has 2-3px white border
img = Image.open('_elev_debug/p2_region0_CONTENT_preview_tile.png')
print(f"Original: {img.size}")
cropped = crop_to_content(img)
print(f"Cropped:  {cropped.size}")
cropped.save("_elev_debug/p2_CROPPED.png")

# Check edge pixels
import numpy as np
arr = np.array(cropped)
print(f"Top-left 3x3 mean: {arr[:3,:3].mean():.0f}")
print(f"Top-right 3x3 mean: {arr[:3,-3:].mean():.0f}")
print(f"Bottom-left 3x3 mean: {arr[-3:,:3].mean():.0f}")
print(f"Bottom-right 3x3 mean: {arr[-3:,-3:].mean():.0f}")

# Also test on other pages
for pn in [4, 6, 8]:
    fn = f'_elev_debug/p{pn}_region0_CONTENT_preview_tile.png'
    if os.path.exists(fn):
        orig = Image.open(fn)
        cr = crop_to_content(orig)
        print(f"P{pn}: {orig.size} -> {cr.size}")
