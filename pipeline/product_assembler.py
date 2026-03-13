"""
Assemble final products from parsed + classified pages.

Responsibilities
────────────────
  1. **Detect PDF type** — multi-image (Type A) or flat-page (Type B).
  2. **Type A**: group pages, pick thumbnail/application from separate
     embedded images (existing logic, improved).
  3. **Type B**: treat each page as one product, segment the composite
     image into texture + room-scene regions, OCR the product name.
  4. Write ``thumbnail.jpg``, ``application.jpg``, ``metadata.json``
     into ``products/<pdf_name>/<product_name>/``.
"""

import json
import os
import re
import time

from PIL import Image

from .pdf_parser import parse_pdf, ParsedPage, PageImage
from .image_filter import detect_repeated_images, filter_page_images
from .clip_classifier import CLIPClassifier
from .name_extractor import (
    extract_product_name, is_non_product_page_text,
    extract_name_ocr_from_image,
    _ocr_with_variants, _best_ocr_line,
)
from .pdf_type_detector import detect_pdf_type, detect_page_types, PDFType
from .composite_segmenter import (
    render_page, segment_composite_image,
    analyze_page_layout, crop_to_content, _edge_density,
)

# Keep original aspect ratio; cap the long edge at this value.
MAX_THUMBNAIL_DIM = 3150


# ── helpers ────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return s.strip("_")


def _resize_keep_aspect(img: Image.Image, max_dim: int = MAX_THUMBNAIL_DIM) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    if h >= w:
        new_h = max_dim
        new_w = int(w * max_dim / h)
    else:
        new_w = max_dim
        new_h = int(h * max_dim / w)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _save_jpg(img: Image.Image, path: str, quality: int = 95) -> None:
    img.convert("RGB").save(path, "JPEG", quality=quality)


# ── product grouping (Type A — multi-image pages) ────────────

def _group_products(pages: list[ParsedPage], classifier: CLIPClassifier):
    """Group pages into product dicts with thumbnail + application.

    Two layout patterns are handled automatically:
      A) Single page with both texture(s) and room scene(s).
      B) Paired pages: page N has texture only, page N+1 has room scene(s)
         (and possibly small variant texture swatches that are discarded).
    """
    products = []
    i = 0

    while i < len(pages):
        page = pages[i]
        textures = list(page.textures)
        scenes = list(page.room_scenes)

        # Skip pages with no product images
        if not textures and not scenes:
            i += 1
            continue

        # Skip index / catalogue overview pages (many small images)
        if len(page.filtered_images) > 6:
            i += 1
            continue

        product_pages = [page]
        all_textures = list(textures)
        all_scenes = list(scenes)

        # Pattern B: texture-only page paired with next page's scenes
        if textures and not scenes and (i + 1) < len(pages):
            nxt = pages[i + 1]
            if nxt.room_scenes:
                product_pages.append(nxt)
                all_textures.extend(nxt.textures)
                all_scenes.extend(nxt.room_scenes)
                i += 1  # consumed next page

        # ---- pick best thumbnail (highest resolution texture) ----
        thumbnail = None
        if all_textures:
            thumbnail = max(all_textures, key=lambda t: t.area)

        # ---- pick best application (most similar scene to texture) ----
        application = None
        if all_scenes:
            if len(all_scenes) == 1:
                application = all_scenes[0]
            elif thumbnail is not None:
                cands = [(idx, s.image) for idx, s in enumerate(all_scenes)]
                best_idx, _ = classifier.find_best_match(thumbnail.image, cands)
                if best_idx is not None:
                    application = all_scenes[best_idx]
                else:
                    application = max(all_scenes, key=lambda s: s.area)
            else:
                application = max(all_scenes, key=lambda s: s.area)

        products.append({
            "pages": product_pages,
            "thumbnail": thumbnail,
            "application": application,
            "name": None,
        })

        i += 1

    return products


# ── Type A processing (original pipeline, improved) ──────────

def _process_type_a(pages, pdf_path, output_dir, tesseract_path, classifier):
    """Process a multi-image PDF (Type A) using the original pipeline."""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output = os.path.join(output_dir, pdf_name)

    # ── 2. Filter images ─────────────────────────────────────
    print("\n[2/6] Filtering images (blank / logos / small)...")
    repeated = detect_repeated_images(pages)
    print(f"  -> {len(repeated)} repeated patterns detected (logos / watermarks)")

    for page in pages:
        page.filtered_images = filter_page_images(page, repeated)

    total_imgs = sum(len(p.images) for p in pages)
    kept = sum(len(p.filtered_images) for p in pages)
    removed = total_imgs - kept
    print(f"  -> {kept} kept, {removed} removed")

    # ── 3. CLIP classification ────────────────────────────────
    print("\n[3/6] Classifying images with CLIP...")
    for page in pages:
        for img in page.filtered_images:
            cat, conf = classifier.get_category(img.image)
            img.clip_category = cat
            img.clip_confidence = conf

            if cat == "texture":
                page.textures.append(img)
            elif cat == "room_scene":
                page.room_scenes.append(img)
            # logo / preview_tile / diagram -> discarded

        if page.textures or page.room_scenes:
            print(f"  page {page.page_num:>3d}: "
                  f"{len(page.textures)} texture(s), "
                  f"{len(page.room_scenes)} scene(s)")

    # ── 3b. Post-correction: size-based room-scene recovery ───
    corrections = 0
    for page in pages:
        if page.room_scenes or len(page.textures) < 2:
            continue
        by_area = sorted(page.textures, key=lambda t: t.area, reverse=True)
        largest = by_area[0]
        second = by_area[1]
        if largest.area > second.area * 2:
            page.textures.remove(largest)
            page.room_scenes.append(largest)
            largest.clip_category = "room_scene"
            corrections += 1
    if corrections:
        print(f"  -> {corrections} room scene(s) recovered via size heuristic")

    # ── 3c. Post-correction: position-based split for 2-image pages ──
    pos_corrections = 0
    for page in pages:
        if len(page.filtered_images) != 2:
            continue
        if len(page.textures) == 1 and len(page.room_scenes) == 1:
            continue
        imgs = sorted(page.filtered_images, key=lambda im: im.rect.y0)
        top_img, bot_img = imgs[0], imgs[1]
        page.textures.clear()
        page.room_scenes.clear()
        top_img.clip_category = "texture"
        bot_img.clip_category = "room_scene"
        page.textures.append(top_img)
        page.room_scenes.append(bot_img)
        pos_corrections += 1
    if pos_corrections:
        print(f"  -> {pos_corrections} page(s) corrected via position heuristic")

    # ── 4. Group into products ────────────────────────────────
    print("\n[4/6] Grouping pages into products...")
    products = _group_products(pages, classifier)
    print(f"  -> {len(products)} products detected")

    # ── 5. Extract product names ──────────────────────────────
    print("\n[5/6] Extracting product names...")
    seen_names: dict[str, int] = {}

    for prod in products:
        name = None
        for pg in prod["pages"]:
            if is_non_product_page_text(pg):
                continue
            name = extract_product_name(pg, pdf_path, tesseract_path)
            if name:
                break

        if not name:
            name = f"PRODUCT_PAGE_{prod['pages'][0].page_num}"

        base = name
        if base in seen_names:
            name = f"{base} P{prod['pages'][0].page_num}"
        seen_names[base] = seen_names.get(base, 0) + 1

        prod["name"] = name
        pg_nums = "+".join(str(p.page_num) for p in prod["pages"])
        print(f"  page {pg_nums}: {name}")

    # ── 6. Write output ───────────────────────────────────────
    print("\n[6/6] Creating product folders...")
    results = []

    for prod in products:
        folder = _sanitize(prod["name"])
        product_dir = os.path.join(pdf_output, folder)
        os.makedirs(product_dir, exist_ok=True)

        if prod["thumbnail"]:
            thumb = _resize_keep_aspect(prod["thumbnail"].image)
            _save_jpg(thumb, os.path.join(product_dir, "thumbnail.jpg"))

        if prod["application"]:
            _save_jpg(
                prod["application"].image,
                os.path.join(product_dir, "application.jpg"),
            )

        meta = {
            "name": prod["name"],
            "pages": [p.page_num for p in prod["pages"]],
            "has_thumbnail": prod["thumbnail"] is not None,
            "has_application": prod["application"] is not None,
            "pdf_type": "multi_image",
        }
        if prod["thumbnail"]:
            t = prod["thumbnail"]
            meta["thumbnail_size"] = f"{t.width}x{t.height}"
            meta["thumbnail_clip_confidence"] = round(t.clip_confidence, 3)
        if prod["application"]:
            a = prod["application"]
            meta["application_size"] = f"{a.width}x{a.height}"
            meta["application_clip_confidence"] = round(a.clip_confidence, 3)

        with open(os.path.join(product_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        parts = []
        if prod["thumbnail"]:
            parts.append("thumbnail.jpg")
        if prod["application"]:
            parts.append("application.jpg")
        parts.append("metadata.json")
        print(f"  {folder}/ -> {', '.join(parts)}")

        results.append({
            "name": prod["name"],
            "folder": product_dir,
            "has_thumbnail": prod["thumbnail"] is not None,
            "has_application": prod["application"] is not None,
        })

    return results


# ── Type B processing (flat-page composite images) ───────────

def _detect_flat_pattern(analyses: list[dict]) -> str:
    """Detect if flat-page PDF uses paired or individual pages.

    Uses **total region count** (content + logo rects) to distinguish
    structured product-info pages from simple full-page images.

    Returns ``'paired'`` if structured pages alternate with simple
    pages, or ``'individual'`` otherwise.
    """
    content = [a for a in analyses if a["role"] not in ("cover", "blank")]
    if len(content) < 3:
        return "individual"

    # Use total region count (logos count as structure markers)
    # Structured pages have 2+ rects (content + logos), simple have ≤1
    def _page_complexity(a):
        return len(a["regions"])  # all rects including logos

    comp = [_page_complexity(a) for a in content]
    types = ["M" if c >= 2 else "S" for c in comp]

    # Uniform (all same) → individual
    if len(set(types)) == 1:
        return "individual"

    # Count alternating transitions
    alt = sum(1 for i in range(1, len(types))
              if types[i] != types[i - 1])
    alt_ratio = alt / max(len(types) - 1, 1)

    # Among simple pages, how many look like application?
    simple = [a for a in content if _page_complexity(a) <= 1]
    scene_yes = sum(1 for a in simple
                    if a["full_scores"].get("room_scene", 0) > 0.12)
    scene_ratio = scene_yes / max(len(simple), 1)

    if alt_ratio > 0.60 and scene_ratio > 0.12:
        return "paired"
    return "individual"


def _is_l_shaped_layout(content: list[dict]) -> bool:
    """Detect 3-region L-shaped layout (e.g. Carving Series).

    Pattern: a wide bottom region spanning full width,
    + an upper-left medium region (thumbnail),
    + an upper-right small region (variant).
    The wide region is at least 1.5× wider than the medium one,
    and sits clearly below both upper regions.
    """
    if len(content) != 3:
        return False
    by_area = sorted(content, key=lambda r: r["area_pct"], reverse=True)
    widest = by_area[0]
    second = by_area[1]
    smallest = by_area[2]
    wx, wy, ww, wh = widest["bbox"]
    sx, sy, sw, sh = second["bbox"]
    # Wide region must be wider and positioned below the other two
    if ww < sw * 1.3:
        return False
    if wy < sy + sh * 0.3:  # wide region not clearly below
        return False
    return True


def _is_composite_tile_page(content: list[dict]) -> bool:
    """Detect a composite product-info page with text on the left
    and a tile preview on the right (e.g. Elevation PDF).

    Identified by: single content region covering > 80 %% of the page
    with a wide aspect ratio (> 1.35).
    """
    if len(content) != 1:
        return False
    r = content[0]
    if r["area_pct"] < 0.80:
        return False
    rw, rh = r["image"].size
    return (rw / rh) > 1.35


def _find_tile_in_composite(region_img: Image.Image,
                            classifier) -> Image.Image:
    """Isolate the tile image from a composite text+tile region.

    Uses CLIP to test several vertical split points (30 %-50 %)
    and picks the right-hand crop that scores highest on
    texture + preview_tile.
    """
    rw, rh = region_img.size
    best_score = 0.0
    best_crop = region_img
    for pct in (0.30, 0.35, 0.40, 0.45, 0.50):
        x = int(rw * pct)
        right = region_img.crop((x, 0, rw, rh))
        scores = classifier.classify(right)
        tile_score = scores.get("texture", 0) + scores.get("preview_tile", 0)
        if tile_score > best_score:
            best_score = tile_score
            best_crop = right
    return crop_to_content(best_crop)


def _extract_main_thumbnail(page_info: dict,
                            classifier=None) -> Image.Image:
    """Extract the main product-tile image from a product-info page.

    Handles:
    - Composite text+tile page → CLIP-guided tile isolation.
    - 3-region L-shape → upper-left medium rect (wide bottom = app, skip).
    - 2 similar-sized rects → left rect (main) vs right (variant).
    - 1 dominant + smaller rects → dominant one.
    - Single region → use it.
    - Full-page fallback → crop solid margins.
    """
    content = page_info["content_regions"]

    if not content:
        return crop_to_content(page_info["page_img"])

    if len(content) == 1:
        # Composite text+tile page (e.g. Elevation)
        if classifier and _is_composite_tile_page(content):
            return _find_tile_in_composite(content[0]["image"], classifier)
        # Single region — crop it to remove borders / margins
        return crop_to_content(content[0]["image"])

    # 3-region L-shaped layout (Carving Series pattern)
    if _is_l_shaped_layout(content):
        by_area = sorted(content, key=lambda r: r["area_pct"], reverse=True)
        # Exclude the widest (bottom application) region;
        # among the remaining two, pick the larger one (= thumbnail)
        upper = sorted(by_area[1:], key=lambda r: r["area_pct"], reverse=True)
        return crop_to_content(upper[0]["image"])

    # Multiple content regions
    by_area = sorted(content, key=lambda r: r["area_pct"], reverse=True)
    a1, a2 = by_area[0]["area_pct"], by_area[1]["area_pct"]

    if a2 > a1 * 0.65:
        # Similar-sized regions (PARKING pattern) → take the LEFT one
        by_x = sorted(by_area[:2], key=lambda r: r["bbox"][0])
        return crop_to_content(by_x[0]["image"])

    # Significantly different sizes → take the largest
    return crop_to_content(by_area[0]["image"])


def _extract_flat_application(app_info: dict) -> Image.Image:
    """Extract application image from an application / mixed page."""
    content = app_info["content_regions"]

    # Prefer a region explicitly classified as room_scene
    scene_regions = [
        r for r in content
        if r["scores"].get("room_scene", 0) > 0.2
    ]
    if scene_regions:
        best = max(scene_regions, key=lambda r: r["area_pct"])
        return crop_to_content(best["image"])

    # Otherwise use the largest content region
    if content:
        largest = max(content, key=lambda r: r["area_pct"])
        return crop_to_content(largest["image"])

    return crop_to_content(app_info["page_img"])


def _ocr_near_regions(page_info: dict, tesseract_path: str | None) -> str | None:
    """OCR specifically around the main content regions of a page.

    Scans below, above, and beside the main thumbnail region where
    product names typically appear, rather than the full page header
    which often contains repeated brand banners.

    Uses direct OCR (no sub-splitting) for small regions.
    """
    import pytesseract as _pyt
    if tesseract_path:
        _pyt.pytesseract.tesseract_cmd = tesseract_path

    content = page_info["content_regions"]
    page_img = page_info["page_img"]
    w, h = page_img.size

    if not content:
        return None

    targets: list[Image.Image] = []

    # Composite text+tile page: OCR left text portion, prefer largest font
    if _is_composite_tile_page(content):
        region = content[0]["image"]
        rw_c, rh_c = region.size
        # Text area is roughly the left 40 %
        text_crop = region.crop((0, 0, int(rw_c * 0.40), rh_c))
        tw, th = text_crop.size
        gray = text_crop.convert("L")
        if th < 400:
            scale = max(2, 400 // max(th, 1))
            gray = gray.resize((tw * scale, th * scale), Image.LANCZOS)
        # Use image_to_data to get word heights → prefer largest text
        try:
            data = _pyt.image_to_data(gray, output_type=_pyt.Output.DICT)
            line_groups: dict[tuple, dict] = {}
            for i in range(len(data["text"])):
                word = data["text"][i].strip()
                if not word or int(data["conf"][i]) < 30:
                    continue
                key = (data["block_num"][i], data["par_num"][i],
                       data["line_num"][i])
                if key not in line_groups:
                    line_groups[key] = {"words": [], "heights": []}
                line_groups[key]["words"].append(word)
                line_groups[key]["heights"].append(data["height"][i])
            # Sort lines by average height (largest text first)
            import numpy as _np
            candidates = []
            for info in line_groups.values():
                text = " ".join(info["words"])
                avg_h = float(_np.mean(info["heights"]))
                candidates.append((text, avg_h))
            candidates.sort(key=lambda c: c[1], reverse=True)
            for text, _ in candidates:
                name = _best_ocr_line([text])
                if name:
                    return name
        except Exception:
            pass
        # Fallback: standard OCR on text crop
        lines = _ocr_with_variants(gray)
        name = _best_ocr_line(lines)
        if name:
            return name
        return None

    # L-shaped layout: name is to the right of thumb / below variant
    if _is_l_shaped_layout(content):
        by_area = sorted(content, key=lambda r: r["area_pct"], reverse=True)
        smallest = by_area[2]  # variant (upper-right)
        sx, sy, sw, sh = smallest["bbox"]
        below_top = sy + sh
        below_bot = min(h // 2, below_top + int(h * 0.15))
        if below_bot - below_top > 10:
            targets.append(page_img.crop((sx, below_top, w, below_bot)))

    # Special case: 2 similar-sized regions (e.g., PARKING layout)
    # Product name is directly below the LEFT tile
    elif len(content) >= 2:
        by_area = sorted(content, key=lambda r: r["area_pct"], reverse=True)
        a1, a2 = by_area[0]["area_pct"], by_area[1]["area_pct"]
        if a2 > a1 * 0.65:
            by_x = sorted(by_area[:2], key=lambda r: r["bbox"][0])
            lx, ly, lw, lh = by_x[0]["bbox"]  # left tile
            below_top = ly + lh
            below_bot = min(h, below_top + int(h * 0.12))
            if below_bot - below_top > 10:
                targets.append(page_img.crop((lx, below_top,
                                              lx + lw, below_bot)))

    # General targets
    all_rects = [r["bbox"] for r in content]
    min_x = min(rx for rx, ry, rw, rh in all_rects)
    max_x = max(rx + rw for rx, ry, rw, rh in all_rects)
    min_y = min(ry for rx, ry, rw, rh in all_rects)
    max_y = max(ry + rh for rx, ry, rw, rh in all_rects)

    # Below the content
    below_top = max_y
    below_bot = min(h, max_y + int(h * 0.15))
    if below_bot - below_top > 20:
        targets.append(page_img.crop((0, below_top, w, below_bot)))

    # Above the content
    above_top = max(0, min_y - int(h * 0.12))
    above_bot = min_y
    if above_bot - above_top > 20:
        targets.append(page_img.crop((0, above_top, w, above_bot)))

    # Left side of content (for layouts with text on left, e.g. Elevation)
    if min_x > w * 0.15:
        targets.append(page_img.crop((0, 0, min_x, h)))

    # Direct OCR each target (no sub-splitting for small regions)
    for region in targets:
        # Upscale small crops for better OCR
        rw, rh = region.size
        if rh < 200:
            scale = max(2, 200 // max(rh, 1))
            region = region.resize((rw * scale, rh * scale), Image.LANCZOS)
        gray = region.convert("L")
        lines = _ocr_with_variants(gray)
        name = _best_ocr_line(lines)
        if name:
            return name

    return None


def _group_flat_products(analyses: list[dict], pattern: str) -> list[dict]:
    """Group analysed pages into product dicts.

    Each product has ``info`` (the product-info page) and optionally
    ``app`` (the application page analysis dict).
    """
    content = [a for a in analyses if a["role"] not in ("cover", "blank")]
    products: list[dict] = []

    if pattern == "paired":
        i = 0
        while i < len(content):
            page = content[i]
            if page["role"] == "application":
                # Orphan application page (no preceding info) → skip
                i += 1
                continue

            product = {"info": page, "app": None}

            # Pair with next page if it's a simpler page (fewer rects)
            if i + 1 < len(content):
                nxt = content[i + 1]
                my_rects = len(page["regions"])
                nxt_rects = len(nxt["regions"])
                is_simpler = nxt_rects < my_rects or nxt_rects <= 1
                if is_simpler and nxt["role"] != "cover":
                    product["app"] = nxt
                    i += 2
                    products.append(product)
                    continue

            products.append(product)
            i += 1
    else:
        # Individual: each page is its own product
        for page in content:
            product = {"info": page, "app": None}

            # "mixed" pages contain both texture and scene
            if page["role"] == "mixed":
                product["app"] = page  # extract scene from same page

            # L-shaped layout: thumbnail + app on same page
            elif _is_l_shaped_layout(page["content_regions"]):
                product["app"] = page

            products.append(product)

    return products


def _process_type_b(pages, pdf_path, output_dir, tesseract_path, classifier):
    """Process a flat-page PDF (Type B) with universal layout analysis.

    Phases:
      1. Analyse every page (contour regions, CLIP, role).
      2. Detect pairing pattern (individual vs. paired pages).
      3. Group pages into products.
      4. Extract main thumbnail + application + product name.
      5. Save output.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output = os.path.join(output_dir, pdf_name)

    # ── Phase 1: analyse every page ──────────────────────────
    print("\n[FLAT-PAGE] Analysing page layouts...")
    analyses: list[dict] = []
    for page in pages:
        if not page.images:
            continue
        info = analyze_page_layout(pdf_path, page.page_num, classifier)
        info["parsed_page"] = page
        analyses.append(info)
        print(f"  page {page.page_num:>3d}: {info['role']:<14s} "
              f"({info['n_content_rects']} content regions)")

    if not analyses:
        return []

    # ── Phase 2: detect pairing pattern ──────────────────────
    pattern = _detect_flat_pattern(analyses)
    print(f"\n  -> Page pattern: {pattern}")

    # ── Phase 3: group into products ─────────────────────────
    products = _group_flat_products(analyses, pattern)
    print(f"  -> {len(products)} products detected")

    # ── Phase 4–5: extract content and save ──────────────────
    results = []
    seen_names: dict[str, int] = {}

    for prod in products:
        info_page = prod["info"]
        app_page = prod.get("app")
        pn = info_page["page_num"]

        # ── Thumbnail ────────────────────────────────────────
        thumb_img = _extract_main_thumbnail(info_page, classifier)

        # ── Application ──────────────────────────────────────
        app_img = None
        if app_page and app_page is not info_page:
            # Separate application page
            app_img = _extract_flat_application(app_page)
        elif info_page["role"] == "mixed":
            # Application region within the same page
            for r in info_page["content_regions"]:
                if r["scores"].get("room_scene", 0) > 0.2:
                    app_img = crop_to_content(r["image"])
                    break

        # L-shaped layout: the widest bottom region is the application
        if not app_img and _is_l_shaped_layout(info_page["content_regions"]):
            by_area = sorted(info_page["content_regions"],
                             key=lambda r: r["area_pct"], reverse=True)
            app_img = crop_to_content(by_area[0]["image"])

        # ── Product name ─────────────────────────────────────
        # For flat pages: try targeted region-aware OCR first (near tiles),
        # then standard text blocks, then full-page OCR as fallback.
        parsed = info_page["parsed_page"]
        name = _ocr_near_regions(info_page, tesseract_path)
        if not name:
            name = extract_product_name(parsed, pdf_path, tesseract_path)
        if not name:
            name = extract_name_ocr_from_image(
                info_page["page_img"], tesseract_path
            )
        if not name:
            name = f"PRODUCT_PAGE_{pn}"

        base = name
        if base in seen_names:
            name = f"{base} P{pn}"
        seen_names[base] = seen_names.get(base, 0) + 1

        # ── Print status ─────────────────────────────────────
        parts = []
        if thumb_img:
            parts.append("thumb")
        if app_img:
            parts.append("app")
        pg_nums = str(pn)
        if app_page and app_page is not info_page:
            pg_nums += f"+{app_page['page_num']}"
        status = ", ".join(parts) if parts else "none"
        print(f"\n  page {pg_nums:>5s}: {name} ({status})")

        # ── Save output ──────────────────────────────────────
        folder = _sanitize(name)
        product_dir = os.path.join(pdf_output, folder)
        os.makedirs(product_dir, exist_ok=True)

        has_thumb = thumb_img is not None
        has_app = app_img is not None

        if has_thumb:
            thumb = _resize_keep_aspect(thumb_img)
            _save_jpg(thumb, os.path.join(product_dir, "thumbnail.jpg"))

        if has_app:
            _save_jpg(app_img, os.path.join(product_dir, "application.jpg"))

        meta = {
            "name": name,
            "pages": [pn] + (
                [app_page["page_num"]]
                if app_page and app_page is not info_page
                else []
            ),
            "has_thumbnail": has_thumb,
            "has_application": has_app,
            "pdf_type": "flat_page",
            "pattern": pattern,
        }
        if has_thumb:
            meta["thumbnail_size"] = f"{thumb_img.size[0]}x{thumb_img.size[1]}"
        if has_app:
            meta["application_size"] = f"{app_img.size[0]}x{app_img.size[1]}"

        with open(os.path.join(product_dir, "metadata.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        results.append({
            "name": name,
            "folder": product_dir,
            "has_thumbnail": has_thumb,
            "has_application": has_app,
        })

    return results


# ── main public entry point ───────────────────────────────────

def process_pdf(pdf_path: str, output_dir: str,
                tesseract_path: str | None = None,
                classifier: CLIPClassifier | None = None) -> list[dict]:
    """Full adaptive pipeline for a single PDF.

    Automatically detects the PDF structure type and routes to
    the appropriate processing path.

    Returns a list of product-result dicts (one per product found).
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    print(f"\n{'=' * 60}")
    print(f"Processing: {pdf_name}")
    print(f"{'=' * 60}")

    # ── 0. Detect PDF type ────────────────────────────────────
    print("\n[0/6] Detecting PDF structure type...")
    pdf_type = detect_pdf_type(pdf_path)
    page_types = detect_page_types(pdf_path)
    flat_count = sum(1 for t in page_types if t == PDFType.FLAT_PAGE)
    multi_count = sum(1 for t in page_types if t == PDFType.MULTI_IMAGE)
    print(f"  -> PDF type: {pdf_type}")
    print(f"  -> Pages: {flat_count} flat-page, {multi_count} multi-image")

    # ── 1. Parse PDF ──────────────────────────────────────────
    print("\n[1/6] Parsing PDF (PyMuPDF)...")
    pages = parse_pdf(pdf_path)
    total_imgs = sum(len(p.images) for p in pages)
    total_text = sum(len(p.text_blocks) for p in pages)
    print(f"  -> {len(pages)} pages, {total_imgs} images, {total_text} text blocks")

    if not pages:
        print("[WARN] No pages found.")
        return []

    # Load CLIP classifier
    if classifier is None:
        classifier = CLIPClassifier()

    # ── Route by type ─────────────────────────────────────────
    start = time.time()

    if pdf_type == PDFType.FLAT_PAGE:
        results = _process_type_b(pages, pdf_path, output_dir, tesseract_path, classifier)
    elif pdf_type == PDFType.MIXED:
        # Process each page according to its type
        flat_pages = [p for p, t in zip(pages, page_types) if t == PDFType.FLAT_PAGE]
        multi_pages = [p for p, t in zip(pages, page_types) if t == PDFType.MULTI_IMAGE]

        results = []
        if multi_pages:
            print(f"\n--- Processing {len(multi_pages)} multi-image pages ---")
            results.extend(
                _process_type_a(multi_pages, pdf_path, output_dir, tesseract_path, classifier)
            )
        if flat_pages:
            print(f"\n--- Processing {len(flat_pages)} flat-page pages ---")
            results.extend(
                _process_type_b(flat_pages, pdf_path, output_dir, tesseract_path, classifier)
            )
    else:
        results = _process_type_a(pages, pdf_path, output_dir, tesseract_path, classifier)

    elapsed = time.time() - start

    # ── Summary ───────────────────────────────────────────────
    with_thumb = sum(1 for r in results if r["has_thumbnail"])
    with_app = sum(1 for r in results if r["has_application"])
    complete = sum(1 for r in results if r["has_thumbnail"] and r["has_application"])

    print(f"\n{'-' * 40}")
    print(f"  PDF type       : {pdf_type}")
    print(f"  Products found : {len(results)}")
    print(f"  With thumbnail : {with_thumb}")
    print(f"  With application: {with_app}")
    print(f"  Complete (both): {complete}")
    print(f"  Time           : {elapsed:.1f}s")

    return results
