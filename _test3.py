"""Quick test: run all 3 new PDFs and report results."""
import time, warnings; warnings.filterwarnings("ignore")
from pipeline.product_assembler import process_pdf
from pipeline.clip_classifier import CLIPClassifier

clf = CLIPClassifier()

for pdf in ["catalogues/16x16 PARKING.pdf", "catalogues/18x12 Elevation.pdf", "catalogues/4X2 Hight Glosy.pdf"]:
    t0 = time.time()
    results = process_pdf(pdf, "products", r"C:\Program Files\Tesseract-OCR\tesseract.exe", clf)
    elapsed = time.time() - t0
    complete = sum(1 for r in results if r["has_thumbnail"] and r["has_application"])
    thumb_only = sum(1 for r in results if r["has_thumbnail"] and not r["has_application"])
    app_only = sum(1 for r in results if not r["has_thumbnail"] and r["has_application"])
    print(f"\n>>> {pdf}: {elapsed:.0f}s, {len(results)} products, {complete} complete, {thumb_only} thumb-only")
    for r in results:
        t = "T" if r["has_thumbnail"] else "-"
        a = "A" if r["has_application"] else "-"
        print(f"    [{t}{a}] {r['name']}")
