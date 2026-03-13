"""Test Carving Series PDF extraction."""
import sys, os, time, shutil
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.product_assembler import process_pdf
from pipeline.clip_classifier import CLIPClassifier
from config.settings import TESSERACT_PATH

OUTPUT = os.path.join(os.path.dirname(__file__), "products_test_carving")
if os.path.exists(OUTPUT):
    shutil.rmtree(OUTPUT)
os.makedirs(OUTPUT)

clf = CLIPClassifier()
pdf = "catalogues/Carving Series.pdf"

t0 = time.time()
results = process_pdf(pdf, OUTPUT, TESSERACT_PATH, clf)
elapsed = time.time() - t0

complete = sum(1 for r in results if r["has_thumbnail"] and r["has_application"])
thumb_only = sum(1 for r in results if r["has_thumbnail"] and not r["has_application"])

print(f"\n{'='*60}")
print(f"  Carving Series: {elapsed:.0f}s")
print(f"  Products: {len(results)}")
print(f"  Complete (thumb+app): {complete}")
print(f"  Thumb-only: {thumb_only}")
print(f"{'='*60}")
for r in results:
    tag = "[TA]" if r["has_thumbnail"] and r["has_application"] else "[T-]"
    print(f"  {tag} {r['name']}")
