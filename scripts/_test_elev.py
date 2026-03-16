import sys, os, shutil
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pipeline.product_assembler import process_pdf
from pipeline.clip_classifier import CLIPClassifier
from config.settings import TESSERACT_PATH

OUTPUT = "products_test_elev"
if os.path.exists(OUTPUT):
    shutil.rmtree(OUTPUT)
os.makedirs(OUTPUT)

clf = CLIPClassifier()
results = process_pdf("catalogues/18x12 Elevation.pdf", OUTPUT, TESSERACT_PATH, clf)

complete = sum(1 for r in results if r["has_thumbnail"] and r["has_application"])
thumb_only = sum(1 for r in results if r["has_thumbnail"] and not r["has_application"])
print(f"\n{len(results)} products, {complete} complete, {thumb_only} thumb-only")
for r in results:
    tag = "[TA]" if r["has_thumbnail"] and r["has_application"] else "[T-]"
    print(f"  {tag} {r['name']}")
