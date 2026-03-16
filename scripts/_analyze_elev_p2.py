"""Analyze Elevation page 2 layout in detail."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from PIL import Image
from pipeline.composite_segmenter import render_page, analyze_page_layout
from pipeline.clip_classifier import CLIPClassifier

clf = CLIPClassifier()
pdf = "catalogues/18x12 Elevation.pdf"
out = "_elev_page2"
os.makedirs(out, exist_ok=True)

# Analyze page 2 (index 1)
info = analyze_page_layout(pdf, 1, clf)
role = info["role"]
regions = info["regions"]
content = info["content_regions"]
print(f"Page 2: role={role}, {len(regions)} regions, {len(content)} content")
info["page_img"].save(f"{out}/page2_full.png")

for i, r in enumerate(regions):
    bbox = r["bbox"]
    print(f"  Region {i}: bbox={bbox}, cat={r['category']}, area={r['area_pct']:.2%}, scores={r['scores']}")
    r['image'].save(f"{out}/region_{i}_{r['category']}.png")

for i, c in enumerate(content):
    bbox = c["bbox"]
    print(f"  Content {i}: bbox={bbox}, cat={c['category']}, size={c['image'].size}")
    c['image'].save(f"{out}/content_{i}_{c['category']}.png")

# Check what the rendered page looks like
pw, ph = info["page_img"].size
print(f"  Page size: {pw}x{ph}")

# Also check pages 4 and 6
for pn in [3, 5, 7]:
    info2 = analyze_page_layout(pdf, pn, clf)
    print(f"\nPage {pn+1}: role={info2['role']}, {len(info2['content_regions'])} content, {len(info2['regions'])} total regions")
    for i, c in enumerate(info2['content_regions']):
        print(f"  Content {i}: bbox={c['bbox']}, cat={c['category']}, size={c['image'].size}")
