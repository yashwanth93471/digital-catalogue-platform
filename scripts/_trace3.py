"""Debug: trace Type B for Elevation and Glosy."""
import warnings; warnings.filterwarnings("ignore")
from pipeline.clip_classifier import CLIPClassifier
from pipeline.composite_segmenter import render_page, segment_composite_image
from pipeline.name_extractor import extract_name_ocr

clf = CLIPClassifier()

for pdf, pgs in [
    ("catalogues/18x12 Elevation.pdf", [1, 3, 5, 10, 15, 20, 25]),
    ("catalogues/4X2 Hight Glosy.pdf", [1, 2, 3, 4, 5, 6]),
]:
    print(f"\n{'='*50}")
    print(f"{pdf}")
    print(f"{'='*50}")
    for pg in pgs:
        page_img = render_page(pdf, pg, dpi=200)
        seg = segment_composite_image(page_img, clf)
        cat, conf = clf.get_category(page_img)
        name = extract_name_ocr(pdf, pg, r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        thumb = "Y" if seg.get("thumbnail") else "N"
        app = "Y" if seg.get("application") else "N"
        print(f"  P{pg:2d}: CLIP={cat}({conf:.2f}) seg={seg['method']} T={thumb} A={app} name={name}")
