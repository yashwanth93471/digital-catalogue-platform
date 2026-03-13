"""Analyze the Carving Series PDF structure."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.composite_segmenter import analyze_page_layout, render_page
from pipeline.clip_classifier import CLIPClassifier
from pipeline.pdf_type_detector import detect_pdf_type, detect_page_types
from config.settings import TESSERACT_PATH
import fitz

clf = CLIPClassifier()
pdf = "catalogues/Carving Series.pdf"

# Basic info
doc = fitz.open(pdf)
print(f"Pages: {len(doc)}")
for i, page in enumerate(doc):
    imgs = page.get_images(full=True)
    texts = page.get_text("blocks")
    text_preview = " ".join(t[4][:60].replace("\n"," ") for t in texts[:3]) if texts else "(no text)"
    print(f"  P{i+1}: {len(imgs)} images, {len(texts)} text blocks | {text_preview}")
doc.close()

# PDF type detection
page_types = detect_page_types(pdf)
pdf_type = detect_pdf_type(pdf)
flat_count = sum(1 for pt in page_types if pt == "flat_page")
multi_count = sum(1 for pt in page_types if pt == "multi_image")
print(f"\nPDF type: {pdf_type}")
print(f"Flat pages: {flat_count}, Multi-image pages: {multi_count}")

# Analyze each page layout
print(f"\n{'='*60}")
print("Page-by-page layout analysis:")
print(f"{'='*60}")
n_pages = len(fitz.open(pdf))
for pn in range(1, n_pages + 1):
    info = analyze_page_layout(pdf, pn, clf, dpi=150)
    page_img = info["page_img"]
    w, h = page_img.size
    print(f"\nPage {pn}: role={info['role']}, {len(info['regions'])} regions, "
          f"{info['n_content_rects']} content, {len(info['logo_regions'])} logos, "
          f"img={w}x{h}")
    print(f"  Full-page CLIP: { {k: round(v,3) for k,v in info['full_scores'].items()} }")
    for i, r in enumerate(info["content_regions"]):
        bx, by, bw, bh = r["bbox"]
        sc = ", ".join(f"{k}:{v:.3f}" for k,v in r['scores'].items())
        print(f"  Content[{i}]: {bw}x{bh} at ({bx},{by}), "
              f"area={r['area_pct']:.1f}%, cat={r['category']}, "
              f"edge_d={r.get('edge_density',0):.3f}, scores=[{sc}]")
    for i, r in enumerate(info["logo_regions"]):
        bx, by, bw, bh = r["bbox"]
        print(f"  Logo[{i}]: {bw}x{bh} at ({bx},{by}), area={r['area_pct']:.1f}%")
    
    # Save rendered pages for visual inspection
    page_img.save(f"_carving_p{pn}.png")

print(f"\nSaved page renders as _carving_p*.png for visual inspection")
