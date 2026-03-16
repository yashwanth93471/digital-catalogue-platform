"""Try to find inner rectangles within Elevation page 2 content region."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import numpy as np
from PIL import Image
from pipeline.composite_segmenter import render_page, analyze_page_layout
from pipeline.clip_classifier import CLIPClassifier

clf = CLIPClassifier()
pdf = "catalogues/18x12 Elevation.pdf"
out = "_elev_page2"

info = analyze_page_layout(pdf, 1, clf)
region = info["content_regions"][0]["image"]
rw, rh = region.size

# Try finding ALL rectangles (no nested removal, much smaller min area)
img_np = np.array(region)
gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 30, 100)

# Dilate to connect edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=2)

contrs, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
total = rw * rh

print(f"Total contours: {len(contrs)}")
print(f"Region: {rw}x{rh}")

# Find all contours > 1% of area with reasonable aspect ratio
rects_found = []
for c in contrs:
    area = cv2.contourArea(c)
    if area < total * 0.01:
        continue
    x, y, w, h = cv2.boundingRect(c)
    if w < 50 or h < 50:
        continue
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    rect_fill = area / (w * h) if w * h > 0 else 0
    rects_found.append((x, y, w, h, len(approx), rect_fill, area / total))

rects_found.sort(key=lambda r: r[6], reverse=True)
print(f"\nContours with area > 1%:")
for i, (x, y, w, h, npts, fill, pct) in enumerate(rects_found[:20]):
    print(f"  [{i}] x={x}, y={y}, w={w}, h={h}, vertices={npts}, fill={fill:.2f}, area={pct:.1%}")
    crop = region.crop((x, y, x+w, y+h))
    crop.save(f"{out}/contour_{i}_{x}_{y}_{w}x{h}.png")

# Also try edge-density based approach: find the tile bounding box
print(f"\n--- Edge density approach ---")
edges_raw = cv2.Canny(gray, 50, 150)

# Column edge density
col_edge = np.mean(edges_raw > 0, axis=0)
row_edge = np.mean(edges_raw > 0, axis=1)

# Threshold: tile area has >3% edge density consistently
thresh = 0.03

# Find left bound of tile (consecutive columns with high edge density)
window = 30
left_tile = rw
for x in range(rw - window):
    if np.mean(col_edge[x:x+window]) > thresh:
        left_tile = x
        break

# Find right bound
right_tile = 0
for x in range(rw - 1, window, -1):
    if np.mean(col_edge[x-window:x]) > thresh:
        right_tile = x
        break

# Find top/bottom in the tile columns region only
tile_rows = edges_raw[:, left_tile:right_tile]
row_edge_tile = np.mean(tile_rows > 0, axis=1)

top_tile = 0
for y in range(rh):
    if row_edge_tile[y] > thresh:
        top_tile = y
        break

bot_tile = rh
for y in range(rh - 1, -1, -1):
    if row_edge_tile[y] > thresh:
        bot_tile = y
        break

print(f"  Tile bounds: left={left_tile}, right={right_tile}, top={top_tile}, bottom={bot_tile}")
print(f"  Tile size: {right_tile-left_tile}x{bot_tile-top_tile}")

tile_crop = region.crop((left_tile, top_tile, right_tile, bot_tile))
tile_crop.save(f"{out}/tile_edgebased.png")
scores = clf.classify(tile_crop)
print(f"  CLIP scores: {scores}")

# Also do this for pages 4, 6, 8
for pn in [3, 5, 7]:
    info2 = analyze_page_layout(pdf, pn, clf)
    if not info2["content_regions"]:
        continue
    reg2 = info2["content_regions"][0]["image"]
    rw2, rh2 = reg2.size
    gray2 = cv2.cvtColor(np.array(reg2), cv2.COLOR_RGB2GRAY)
    edges2 = cv2.Canny(gray2, 50, 150)
    ce2 = np.mean(edges2 > 0, axis=0)
    re2 = np.mean(edges2 > 0, axis=1)
    
    l2 = rw2
    for x in range(rw2 - window):
        if np.mean(ce2[x:x+window]) > thresh:
            l2 = x
            break
    r2 = 0
    for x in range(rw2 - 1, window, -1):
        if np.mean(ce2[x-window:x]) > thresh:
            r2 = x
            break
    
    tile2 = edges2[:, l2:r2]
    re2t = np.mean(tile2 > 0, axis=1) if r2 > l2 else re2
    t2 = 0
    for y in range(rh2):
        if re2t[y] > thresh:
            t2 = y
            break
    b2 = rh2
    for y in range(rh2 - 1, -1, -1):
        if re2t[y] > thresh:
            b2 = y
            break
    
    print(f"\nPage {pn+1}: bounds l={l2}, r={r2}, t={t2}, b={b2}, size={r2-l2}x{b2-t2}")
    if r2 > l2 and b2 > t2:
        tc2 = reg2.crop((l2, t2, r2, b2))
        tc2.save(f"{out}/tile_p{pn+1}.png")
        s2 = clf.classify(tc2)
        print(f"  CLIP: {s2}")
