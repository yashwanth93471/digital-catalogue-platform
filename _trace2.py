"""Debug: trace Type B processing for PARKING pages."""
import warnings; warnings.filterwarnings("ignore")
from pipeline.clip_classifier import CLIPClassifier
from pipeline.composite_segmenter import render_page, segment_composite_image
from pipeline.name_extractor import extract_name_ocr

clf = CLIPClassifier()

pdf = "catalogues/16x16 PARKING.pdf"
for pg in [1, 2, 5, 10, 15, 20, 25, 30]:
    print(f"\nPage {pg}:")
    page_img = render_page(pdf, pg, dpi=200)
    print(f"  Rendered: {page_img.size}")
    
    seg = segment_composite_image(page_img, clf)
    thumb = seg.get("thumbnail")
    app = seg.get("application")
    method = seg["method"]
    print(f"  Segmentation: method={method}")
    print(f"    thumb: {thumb.size if thumb else None}")
    print(f"    app:   {app.size if app else None}")
    
    # CLIP on whole page
    cat, conf = clf.get_category(page_img)
    print(f"  CLIP whole page: {cat} ({conf:.3f})")
    
    # OCR
    name = extract_name_ocr(pdf, pg, r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    print(f"  OCR name: {name}")
