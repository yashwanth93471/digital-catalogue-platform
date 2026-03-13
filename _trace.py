"""Debug: trace exact processing of 16x16 PARKING page by page."""
import warnings; warnings.filterwarnings("ignore")
from pipeline.pdf_type_detector import detect_pdf_type, detect_page_types, PDFType
from pipeline.pdf_parser import parse_pdf
from pipeline.image_filter import detect_repeated_images, filter_page_images

# 1. What type does it detect?
for pdf in ["catalogues/16x16 PARKING.pdf", "catalogues/18x12 Elevation.pdf", "catalogues/4X2 Hight Glosy.pdf"]:
    ptype = detect_pdf_type(pdf)
    ptypes = detect_page_types(pdf)
    flat = sum(1 for t in ptypes if t == PDFType.FLAT_PAGE)
    multi = sum(1 for t in ptypes if t == PDFType.MULTI_IMAGE)
    print(f"{pdf}: type={ptype}, flat={flat}, multi={multi}, total={len(ptypes)}")

# 2. For PARKING specifically, trace filtering
print("\n--- 16x16 PARKING filter trace ---")
pages = parse_pdf("catalogues/16x16 PARKING.pdf")
repeated = detect_repeated_images(pages)
print(f"Repeated hashes: {len(repeated)}")
for pg in pages:
    filtered = filter_page_images(pg, repeated)
    n_raw = len(pg.images)
    n_big = len([i for i in pg.images if i.area >= 90000])
    print(f"  P{pg.page_num:2d}: {n_raw} raw, {n_big} big, {len(filtered)} filtered")
