"""Test: crop detected regions from flat pages, classify with CLIP."""
import fitz, io, cv2, numpy as np
from PIL import Image
from pipeline.clip_classifier import CLIPClassifier

clf = CLIPClassifier()

def detect_and_classify(page, pg_num, name):
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = h * w * 0.05
    regions = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        if rw * rh < min_area:
            continue
        # Skip regions that are >90% of page (that's the whole page)
        if rw * rh > h * w * 0.90:
            continue
        pct = rw * rh / (h * w) * 100
        regions.append((x, y, rw, rh, pct))
    regions.sort(key=lambda r: r[4], reverse=True)
    
    print(f"\n{name} Page {pg_num}: {len(regions)} regions found")
    for i, (x, y, rw, rh, pct) in enumerate(regions[:6]):
        crop = img.crop((x, y, x+rw, y+rh))
        cat, conf = clf.get_category(crop)
        pos = "TOP" if (y + rh/2) < h/2 else "BOT"
        aspect = max(rw, rh) / max(min(rw, rh), 1)
        print(f"  R{i}: ({x},{y}) {rw}x{rh} {pct:.1f}% {pos} asp={aspect:.1f} -> {cat} ({conf:.3f})")

# Also test: what if we just split the page into quadrants or halves?
def split_classify(page, pg_num, name):
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    w, h = img.size
    
    # Top half / bottom half
    top = img.crop((0, 0, w, h//2))
    bot = img.crop((0, h//2, w, h))
    # Left half / right half
    left = img.crop((0, 0, w//2, h))
    right = img.crop((w//2, 0, w, h))
    
    print(f"\n  Split test ({name} P{pg_num}):")
    for label, crop in [("TOP-HALF", top), ("BOT-HALF", bot), ("LEFT-HALF", left), ("RIGHT-HALF", right)]:
        cat, conf = clf.get_category(crop)
        print(f"    {label}: {cat} ({conf:.3f})")

for pdf_path, pages in [
    ("catalogues/16x16 PARKING.pdf", [2, 5, 10]),
    ("catalogues/18x12 Elevation.pdf", [2, 5, 10]),
    ("catalogues/4X2 Hight Glosy.pdf", [1, 4]),
]:
    doc = fitz.open(pdf_path)
    name = pdf_path.split("/")[-1]
    for pg in pages:
        if pg <= len(doc):
            detect_and_classify(doc[pg-1], pg, name)
            split_classify(doc[pg-1], pg, name)
    doc.close()

print("\nDone.")
