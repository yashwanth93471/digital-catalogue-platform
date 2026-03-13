"""Quick test: verify PDF type detection on all catalogues."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.pdf_type_detector import detect_pdf_type, detect_page_types, PDFType

pdfs = [
    "sample.pdf",
    "DREAM TILE WORLD.pdf",
    "16x16 PARKING.pdf",
    "18x12 Elevation.pdf",
    "4X2 Hight Glosy.pdf",
]

for name in pdfs:
    path = os.path.join("catalogues", name)
    if not os.path.isfile(path):
        print(f"{name:30s} -> FILE NOT FOUND")
        continue
    pdf_type = detect_pdf_type(path)
    page_types = detect_page_types(path)
    flat = sum(1 for t in page_types if t == PDFType.FLAT_PAGE)
    multi = sum(1 for t in page_types if t == PDFType.MULTI_IMAGE)
    print(f"{name:30s} -> {pdf_type:15s}  ({flat} flat, {multi} multi, {len(page_types)} total)")
