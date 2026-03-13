import sys, os
sys.path.insert(0, '.')
from pipeline.composite_segmenter import analyze_page_layout
from pipeline.clip_classifier import CLIPClassifier
clf = CLIPClassifier()
pdf = 'catalogues/Carving Series.pdf'
for pn in [15, 16]:
    info = analyze_page_layout(pdf, pn, clf, dpi=150)
    w, h = info['page_img'].size
    role = info['role']
    nc = info['n_content_rects']
    nr = len(info['regions'])
    print(f"Page {pn}: role={role}, {nr} regions, {nc} content, img={w}x{h}")
    for i, r in enumerate(info['content_regions']):
        bx, by, bw, bh = r['bbox']
        print(f"  C[{i}]: {bw}x{bh} at ({bx},{by}), area={r['area_pct']:.1f}%, cat={r['category']}")
