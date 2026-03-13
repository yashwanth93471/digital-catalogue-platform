"""Debug Elevation thumbnail extraction - check what regions are detected and what gets saved."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.composite_segmenter import analyze_page_layout, render_page
from pipeline.clip_classifier import CLIPClassifier
from pipeline.product_assembler import _extract_main_thumbnail, _is_l_shaped_layout
from config.settings import TESSERACT_PATH
from PIL import Image

clf = CLIPClassifier()
pdf = "catalogues/18x12 Elevation.pdf"

os.makedirs("_elev_debug", exist_ok=True)

# Check a few product pages (even pages = info pages in paired layout)
for pn in [2, 4, 6, 8]:
    info = analyze_page_layout(pdf, pn, clf, dpi=150)
    page_img = info["page_img"]
    w, h = page_img.size
    content = info["content_regions"]
    all_regions = info["regions"]
    
    print(f"\n=== Page {pn}: role={info['role']}, img={w}x{h} ===")
    print(f"  Total regions: {len(all_regions)}, content: {len(content)}, logos: {len(info['logo_regions'])}")
    print(f"  L-shaped: {_is_l_shaped_layout(content)}")
    
    for i, r in enumerate(all_regions):
        bx, by, bw, bh = r["bbox"]
        is_content = not r.get("is_logo", False)
        tag = "CONTENT" if is_content else "LOGO"
        print(f"  [{tag}] Region[{i}]: {bw}x{bh} at ({bx},{by}), "
              f"area={r['area_pct']:.1f}%, cat={r['category']}")
        # Save each region
        r["image"].save(f"_elev_debug/p{pn}_region{i}_{tag}_{r['category']}.png")
    
    # Save the extracted thumbnail
    thumb = _extract_main_thumbnail(info)
    if thumb:
        thumb.save(f"_elev_debug/p{pn}_THUMBNAIL.png")
        print(f"  -> Thumbnail saved: {thumb.size}")
    
    # Save full page for reference
    page_img.save(f"_elev_debug/p{pn}_full.png")

    # Also render at higher DPI for comparison
    hi_info = analyze_page_layout(pdf, pn, clf, dpi=200)
    hi_content = hi_info["content_regions"]
    print(f"  [200dpi] content: {len(hi_content)} regions")
    for i, r in enumerate(hi_content):
        bx, by, bw, bh = r["bbox"]
        print(f"    Content[{i}]: {bw}x{bh} at ({bx},{by}), area={r['area_pct']:.1f}%, cat={r['category']}")

print("\nDebug images saved to _elev_debug/")
