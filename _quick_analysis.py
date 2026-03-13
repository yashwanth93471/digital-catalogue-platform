"""Quick region analysis - 1 page per PDF."""
import fitz, io, cv2, numpy as np
from PIL import Image

def analyze(page, pg_num, name):
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # Background
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    bg_val = int(np.argmax(hist))
    
    # Edge detection + contours
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
        pct = rw * rh / (h * w) * 100
        regions.append((x, y, rw, rh, pct))
    regions.sort(key=lambda r: r[4], reverse=True)
    
    print(f"\n{name} Page {pg_num}: rendered {w}x{h}, bg={bg_val}")
    print(f"  Large regions: {len(regions)}")
    for i, (x, y, rw, rh, pct) in enumerate(regions[:8]):
        pos = "TOP" if (y + rh/2) < h/2 else "BOT"
        print(f"    R{i}: ({x},{y}) {rw}x{rh} = {pct:.1f}% {pos}")

for pdf_path, pages in [
    ("catalogues/16x16 PARKING.pdf", [2, 10]),
    ("catalogues/18x12 Elevation.pdf", [2, 10]),
    ("catalogues/4X2 Hight Glosy.pdf", [1, 2]),
]:
    doc = fitz.open(pdf_path)
    name = pdf_path.split("/")[-1]
    for pg in pages:
        if pg <= len(doc):
            analyze(doc[pg-1], pg, name)
    doc.close()

print("\nDone.")
