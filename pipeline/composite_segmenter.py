"""
Segment a composite full-page catalogue image into regions.

For Type B (flat-page) PDFs, every page is a single designed image
containing the texture swatch, room scene, product name, and
decorations all merged together.

Segmentation strategies (tried in order):

  1. **Contour analysis** — find large rectangular regions via edge
     detection; classify with CLIP or edge-density heuristic.
  2. **Layout split + CLIP** — horizontal/vertical cuts scored by CLIP
     category probabilities (texture, room_scene, preview_tile).
  3. **Heuristic split** — split at ~40 % from top and assign roles
     using edge-density (lower density = texture swatch).
  4. **Whole-page fallback** — save everything as thumbnail.
"""

import io

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


# ── Render helpers ─────────────────────────────────────────────

def render_page(pdf_path: str, page_num: int, dpi: int = 200) -> Image.Image:
    """Render a PDF page to a PIL RGB image at *dpi* resolution.

    ``page_num`` is 1-indexed.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return img


def render_page_region(pdf_path: str, page_num: int,
                       top_frac: float = 0.0, bottom_frac: float = 1.0,
                       dpi: int = 300) -> Image.Image:
    """Render a vertical slice of a PDF page (for targeted OCR)."""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    r = page.rect
    clip = fitz.Rect(r.x0, r.y0 + r.height * top_frac,
                     r.x1, r.y0 + r.height * bottom_frac)
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return img


# ── Edge-density helper ───────────────────────────────────────

def _edge_density(pil_img: Image.Image) -> float:
    """Fraction of pixels that are Canny edges.

    Texture swatches are uniform (low density ~0.01–0.05).
    Room scenes / product photos have more detail (higher density).
    """
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.sum(edges > 0)) / edges.size


def _product_score(scores: dict) -> float:
    """Sum of categories that represent actual product content."""
    return (scores.get("texture", 0)
            + scores.get("room_scene", 0)
            + scores.get("preview_tile", 0))


def _assign_roles(img_a: Image.Image, img_b: Image.Image,
                  scores_a: dict | None, scores_b: dict | None):
    """Decide which image is thumbnail (texture) and which is application.

    Uses CLIP scores when available; falls back to edge-density.
    Returns ``(thumbnail, application)``.
    """
    # If CLIP clearly distinguishes texture vs room_scene, use that.
    if scores_a and scores_b:
        a_tex = scores_a.get("texture", 0) + scores_a.get("preview_tile", 0) * 0.5
        b_tex = scores_b.get("texture", 0) + scores_b.get("preview_tile", 0) * 0.5
        a_scene = scores_a.get("room_scene", 0)
        b_scene = scores_b.get("room_scene", 0)

        # Clear texture vs scene distinction
        if a_tex > b_tex + 0.15 and b_scene > a_scene:
            return img_a, img_b
        if b_tex > a_tex + 0.15 and a_scene > b_scene:
            return img_b, img_a

    # Fall back to edge density: lower density = texture swatch
    da, db = _edge_density(img_a), _edge_density(img_b)
    if da <= db:
        return img_a, img_b   # a is simpler → texture
    return img_b, img_a


# ── Contour-based region detection ────────────────────────────

def _find_large_rects(img_np: np.ndarray, min_area_ratio: float = 0.05):
    """Find large rectangular regions in the image via contour detection.

    Returns a list of (x, y, w, h) bounding boxes, sorted by area
    descending.  Only keeps rectangles whose area is ≥ min_area_ratio
    of the total image area.
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    total_area = h * w

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area >= total_area * min_area_ratio:
            aspect = max(cw, ch) / (min(cw, ch) + 1)
            if aspect < 10:
                rects.append((x, y, cw, ch, area))

    rects.sort(key=lambda r: r[4], reverse=True)
    return [(x, y, cw, ch) for x, y, cw, ch, _ in rects]


# ── Layout-based segmentation ─────────────────────────────────

def _segment_by_layout(img: Image.Image):
    """Generate candidate horizontal and vertical splits."""
    w, h = img.size
    regions = []

    # Horizontal splits
    for frac in [0.35, 0.40, 0.45, 0.50]:
        sy = int(h * frac)
        top = img.crop((0, 0, w, sy))
        bot = img.crop((0, sy, w, h))
        regions.append(("h", top, bot, frac))

    # Vertical splits (some catalogues use left/right layout)
    for frac in [0.45, 0.50, 0.55]:
        sx = int(w * frac)
        left = img.crop((0, 0, sx, h))
        right = img.crop((sx, 0, w, h))
        regions.append(("v", left, right, frac))

    return regions


def _pick_best_split(regions: list, classifier) -> tuple:
    """Score each candidate split using CLIP.

    Accepts three kinds of good splits:
      A) Classic: one half texture, the other room_scene.
      B) Product-content: both halves are product content (texture +
         preview_tile + room_scene); roles assigned by edge density.
      C) Texture-dominant: one half clearly texture, the other has
         preview_tile — treat preview_tile half as application.

    Returns ``(thumbnail_img, application_img)`` or ``(None, None)``.
    """
    best_score = -1.0
    best_thumb = None
    best_app = None

    for orient, img_a, img_b, frac in regions:
        sa = classifier.classify(img_a)
        sb = classifier.classify(img_b)

        # ── Approach A: classic texture + room_scene ──
        score_a = sa.get("texture", 0) + sb.get("room_scene", 0)
        score_b = sb.get("texture", 0) + sa.get("room_scene", 0)

        # ── Approach B: preview_tile counts as product content ──
        # Treat preview_tile as a partial substitute for texture.
        score_c = (sa.get("texture", 0) + sa.get("preview_tile", 0)
                   + sb.get("room_scene", 0))
        score_d = (sb.get("texture", 0) + sb.get("preview_tile", 0)
                   + sa.get("room_scene", 0))

        # ── Approach C: both halves are product, neither is logo/diagram ──
        prod_a = _product_score(sa)
        prod_b = _product_score(sb)
        score_prod = min(prod_a, prod_b) * 0.85  # scale slightly

        combined = max(score_a, score_b, score_c, score_d, score_prod)

        if combined > best_score:
            best_score = combined
            # Assign roles
            tex_scene_best = max(score_a, score_b, score_c, score_d)
            if tex_scene_best > score_prod:
                # CLIP-based role: pick the arrangement with highest score
                if max(score_a, score_c) >= max(score_b, score_d):
                    best_thumb, best_app = img_a, img_b
                else:
                    best_thumb, best_app = img_b, img_a
            else:
                # Edge-density-based role (both are "product content")
                best_thumb, best_app = _assign_roles(img_a, img_b, sa, sb)

    if best_score >= 0.30:
        return best_thumb, best_app
    return None, None


# ── Contour-based segmentation (primary) ──────────────────────

def _segment_by_contours(img: Image.Image, classifier):
    """Try to find distinct rectangular regions via contours,
    then classify each region.

    Returns ``(thumbnail_img, application_img)`` or ``(None, None)``.
    """
    img_np = np.array(img)
    rects = _find_large_rects(img_np, min_area_ratio=0.08)

    if len(rects) < 2:
        return None, None

    # Remove highly overlapping rects (IoU > 0.5)
    filtered = [rects[0]]
    for x, y, w, h in rects[1:6]:
        overlap = False
        for fx, fy, fw, fh in filtered:
            ix0, iy0 = max(x, fx), max(y, fy)
            ix1, iy1 = min(x + w, fx + fw), min(y + h, fy + fh)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            union = w * h + fw * fh - inter
            if union > 0 and inter / union > 0.5:
                overlap = True
                break
        if not overlap:
            filtered.append((x, y, w, h))

    if len(filtered) < 2:
        return None, None

    # Crop and classify each region; skip logo/diagram
    candidates = []
    for x, y, w, h in filtered[:6]:
        cropped = img.crop((x, y, x + w, y + h))
        scores = classifier.classify(cropped)
        product = _product_score(scores)
        if product < 0.20:          # mostly logo/diagram → skip
            continue
        candidates.append({
            "image": cropped,
            "scores": scores,
            "area": w * h,
        })

    if len(candidates) < 2:
        return None, None

    # Classic: one texture, one room_scene
    best_tex = max(candidates, key=lambda c: c["scores"].get("texture", 0))
    best_scene = max(candidates, key=lambda c: c["scores"].get("room_scene", 0))
    if (best_tex["scores"].get("texture", 0) > 0.3
            and best_scene["scores"].get("room_scene", 0) > 0.3
            and best_tex is not best_scene):
        return best_tex["image"], best_scene["image"]

    # Fallback: both are product content — use edge density for roles
    c1, c2 = candidates[0], candidates[1]
    prod1 = _product_score(c1["scores"])
    prod2 = _product_score(c2["scores"])
    if prod1 > 0.4 and prod2 > 0.4:
        thumb, app = _assign_roles(c1["image"], c2["image"],
                                   c1["scores"], c2["scores"])
        return thumb, app

    return None, None


# ── Heuristic split (no CLIP needed) ──────────────────────────

def _heuristic_split(page_image: Image.Image):
    """Split at 40 % from top and assign roles by edge density.

    Works for the vast majority of catalogue layouts where the
    texture swatch occupies the upper portion and the product
    display / room scene fills the lower portion.
    """
    w, h = page_image.size
    split_y = int(h * 0.40)
    top = page_image.crop((0, 0, w, split_y))
    bot = page_image.crop((0, split_y, w, h))

    d_top = _edge_density(top)
    d_bot = _edge_density(bot)

    # Only accept if there's a meaningful difference in complexity
    diff = abs(d_top - d_bot)
    if diff < 0.005:
        return None, None

    if d_top <= d_bot:
        return top, bot    # top simpler = texture
    return bot, top


# ── Main public API ───────────────────────────────────────────

def segment_composite_image(
    page_image: Image.Image,
    classifier,
) -> dict:
    """Segment a composite catalogue page into thumbnail + application.

    Parameters
    ----------
    page_image : PIL Image
        The full-page composite image.
    classifier : CLIPClassifier
        Loaded CLIP model for classification.

    Returns
    -------
    dict with keys:
        ``thumbnail``    : PIL Image or None
        ``application``  : PIL Image or None
        ``method``       : str describing which strategy succeeded
    """
    # Strategy 1: Contour-based region detection
    tex, scene = _segment_by_contours(page_image, classifier)
    if tex is not None and scene is not None:
        return {"thumbnail": tex, "application": scene, "method": "contour"}

    # Strategy 2: Layout splits scored by CLIP
    regions = _segment_by_layout(page_image)
    tex, scene = _pick_best_split(regions, classifier)
    if tex is not None and scene is not None:
        return {"thumbnail": tex, "application": scene, "method": "layout_split"}

    # Strategy 3: Heuristic split using edge density (no CLIP)
    tex, scene = _heuristic_split(page_image)
    if tex is not None and scene is not None:
        return {"thumbnail": tex, "application": scene, "method": "heuristic_split"}

    # Strategy 4: Whole-page fallback — always save as thumbnail
    return {"thumbnail": page_image, "application": None, "method": "whole_page"}


# ── Nested-rect removal ───────────────────────────────────────

def _remove_nested_rects(rects, containment=0.6):
    """Remove rects that are mostly contained within a larger rect.

    Keeps the *outer* rect and drops the inner one when > containment
    fraction of the inner rect's area overlaps a bigger rect.
    """
    if len(rects) <= 1:
        return rects

    keep = []
    for i, (x, y, w, h) in enumerate(rects):
        inside_larger = False
        for j, (ox, oy, ow, oh) in enumerate(rects):
            if i == j:
                continue
            # intersection
            ix0, iy0 = max(x, ox), max(y, oy)
            ix1, iy1 = min(x + w, ox + ow), min(y + h, oy + oh)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            area_i = w * h
            if area_i > 0 and inter / area_i > containment and ow * oh > area_i:
                inside_larger = True
                break
        if not inside_larger:
            keep.append((x, y, w, h))
    return keep


# ── Page-layout analysis ──────────────────────────────────────

def analyze_page_layout(pdf_path: str, page_num: int, classifier,
                        dpi: int = 150) -> dict:
    """Analyse a single flat-page and return structured layout info.

    Returns a dict with keys:
        page_num, page_img, full_scores,
        regions (list of dicts), content_regions, logo_regions,
        role ('product_info' / 'application' / 'cover' / 'mixed'),
        n_content_rects, single_large (bool).
    """
    page_img = render_page(pdf_path, page_num, dpi=dpi)
    w, h = page_img.size
    total_area = w * h

    full_scores = classifier.classify(page_img)

    # Contour regions
    img_np = np.array(page_img)
    raw_rects = _find_large_rects(img_np, min_area_ratio=0.04)
    rects = _remove_nested_rects(raw_rects)

    regions = []
    for (rx, ry, rw, rh) in rects:
        area_pct = (rw * rh) / total_area
        crop = page_img.crop((rx, ry, rx + rw, ry + rh))
        scores = classifier.classify(crop)
        cat = max(scores, key=scores.get)
        regions.append({
            "bbox": (rx, ry, rw, rh),
            "image": crop,
            "scores": scores,
            "category": cat,
            "area_pct": area_pct,
            "is_logo": scores.get("logo", 0) > 0.5,
            "edge_density": _edge_density(crop),
        })

    content_regions = [r for r in regions if not r["is_logo"]]
    logo_regions = [r for r in regions if r["is_logo"]]

    scene_score = full_scores.get("room_scene", 0)
    logo_score = full_scores.get("logo", 0)
    has_scene_region = any(
        r["scores"].get("room_scene", 0) > 0.3 for r in content_regions
    )
    single_large = (
        len(content_regions) <= 1
        and (not content_regions or content_regions[0]["area_pct"] > 0.5)
    )

    # Determine role
    if logo_score > 0.5 and not content_regions:
        role = "cover"
    elif scene_score > 0.35 and single_large:
        role = "application"
    elif has_scene_region and len(content_regions) >= 2:
        role = "mixed"
    elif len(content_regions) == 0:
        role = "cover" if logo_score > 0.3 else "blank"
    else:
        role = "product_info"

    return {
        "page_num": page_num,
        "page_img": page_img,
        "full_scores": full_scores,
        "regions": regions,
        "content_regions": content_regions,
        "logo_regions": logo_regions,
        "role": role,
        "n_content_rects": len(content_regions),
        "single_large": single_large,
    }


# ── Content-area cropping ─────────────────────────────────────

def crop_to_content(pil_img: Image.Image) -> Image.Image:
    """Crop solid-colour margins (headers, footers, borders).

    Scans from each edge inward, removing rows / columns with very
    low standard deviation (solid background).
    """
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    h, w = gray.shape

    row_std = np.std(gray, axis=1)
    col_std = np.std(gray, axis=0)
    threshold = 15.0

    # Smooth row_std with a simple moving-average
    k = max(h // 30, 3)
    pad = np.pad(row_std, k // 2, mode="edge")
    smoothed_row = np.convolve(pad, np.ones(k) / k, mode="valid")[:h]

    # Find vertical content bounds (small fixed padding to avoid cutting content)
    pad_px = 2
    top = 0
    for i in range(h):
        if smoothed_row[i] > threshold:
            top = max(0, i - pad_px)
            break

    bottom = h
    for i in range(h - 1, -1, -1):
        if smoothed_row[i] > threshold:
            bottom = min(h, i + pad_px + 1)
            break

    # Find horizontal content bounds
    kw = max(w // 30, 3)
    pad_c = np.pad(col_std, kw // 2, mode="edge")
    smoothed_col = np.convolve(pad_c, np.ones(kw) / kw, mode="valid")[:w]

    left = 0
    for j in range(w):
        if smoothed_col[j] > threshold:
            left = max(0, j - pad_px)
            break

    right = w
    for j in range(w - 1, -1, -1):
        if smoothed_col[j] > threshold:
            right = min(w, j + pad_px + 1)
            break

    # Only crop if we keep at least 40 % in each dimension
    new_h, new_w = bottom - top, right - left
    if new_h < h * 0.40 or new_w < w * 0.40:
        return pil_img

    if top == 0 and bottom == h and left == 0 and right == w:
        return pil_img

    return pil_img.crop((left, top, right, bottom))
