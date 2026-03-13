"""Test ALL 5 PDFs (Type A + Type B) for full regression."""
import sys, os, time, shutil
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.product_assembler import process_pdf
from pipeline.clip_classifier import CLIPClassifier
from config.settings import TESSERACT_PATH

OUTPUT = os.path.join(os.path.dirname(__file__), "products_test")
if os.path.exists(OUTPUT):
    shutil.rmtree(OUTPUT)
os.makedirs(OUTPUT)

clf = CLIPClassifier()

pdfs = [
    "catalogues/sample.pdf",
    "catalogues/DREAM TILE WORLD.pdf",
    "catalogues/4X2 Hight Glosy.pdf",
    "catalogues/18x12 Elevation.pdf",
    "catalogues/16x16 PARKING.pdf",
    "catalogues/Carving Series.pdf",
]

all_results = {}

for pdf in pdfs:
    name = os.path.splitext(os.path.basename(pdf))[0]
    t0 = time.time()
    results = process_pdf(pdf, OUTPUT, TESSERACT_PATH, clf)
    elapsed = time.time() - t0

    complete = sum(1 for r in results if r["has_thumbnail"] and r["has_application"])
    thumb_only = sum(1 for r in results if r["has_thumbnail"] and not r["has_application"])

    all_results[name] = {
        "total": len(results),
        "complete": complete,
        "thumb_only": thumb_only,
        "time": elapsed,
        "items": results,
    }

    print(f"\n>>> {pdf}: {elapsed:.0f}s, {len(results)} products, "
          f"{complete} complete, {thumb_only} thumb-only")
    for r in results:
        tag = "[TA]" if r["has_thumbnail"] and r["has_application"] else "[T-]"
        print(f"    {tag} {r['name']}")

# Summary
print(f"\n\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
for name, info in all_results.items():
    pct = info["complete"] / info["total"] * 100 if info["total"] else 0
    print(f"  {name:30s}  {info['total']:>3d} products  "
          f"{info['complete']:>3d} complete ({pct:5.1f}%)  "
          f"{info['thumb_only']:>3d} thumb-only  {info['time']:.0f}s")
