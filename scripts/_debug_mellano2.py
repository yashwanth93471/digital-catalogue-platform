import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline'))
from composite_segmenter import render_page, _find_large_rects, _remove_nested_rects
import numpy as np

pdf = 'catalogues/MILANO CERAMICA 2.pdf'

for pn in range(40):
    img = render_page(pdf, pn, dpi=100)
    w, h = img.size
    arr = np.array(img)
    rects = _find_large_rects(arr, min_area_ratio=0.02)
    rects = _remove_nested_rects(rects)
    parts = []
    for rx, ry, rw, rh in rects:
        area = (rw * rh) / (w * h)
        asp = rw / rh if rh else 0
        parts.append(f"  ({rx},{ry}) {rw}x{rh} area={area*100:.0f}% asp={asp:.2f}")
    rect_str = " | ".join(parts) if parts else "(none)"
    print(f"P{pn+1:2d} ({w}x{h}): {len(rects)} rects {rect_str}")
