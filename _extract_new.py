"""Extract products from 4 new PDFs with targeted logic.

PDF 1: DREAM HYDERABAD - Every page: big app top + small thumb bottom-left + name below
PDF 2: MOROCON GVT - P1 intro, odd=2 thumbs+name, even=application  
PDF 3: MILLANO CERAMICA - P1-4 intro, P42-43 ignore, triplets: sample→thumbs→app
PDF 4: PARK COLLECTION - Extract only application images + product names
"""
import sys, os, re, json, shutil
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import fitz
import numpy as np
import pytesseract as pyt
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pipeline.composite_segmenter import (
    render_page, _find_large_rects, _remove_nested_rects, crop_to_content,
)
from pipeline.clip_classifier import CLIPClassifier
from pipeline.name_extractor import _ocr_with_variants, _best_ocr_line

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pyt.pytesseract.tesseract_cmd = TESSERACT

THUMB_ASPECT = 3 / 4  # width / height (portrait) from reference

clf = CLIPClassifier()


def _sanitize(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return s.strip("_")


def _save_jpg(img, path, quality=95):
    img.convert("RGB").save(path, "JPEG", quality=quality)


def _resize_thumb(img, max_dim=3150):
    """Resize keeping aspect, cap long edge."""
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


def _force_portrait(img):
    """Ensure image is portrait (h > w). If landscape, crop center to portrait."""
    w, h = img.size
    if h >= w:
        return img  # already portrait
    # Crop to portrait from center
    target_w = int(h * THUMB_ASPECT)
    if target_w > w:
        target_w = w
    left = (w - target_w) // 2
    return img.crop((left, 0, left + target_w, h))


def _ocr_region_text(img, psm=6):
    """OCR a region and return raw text."""
    gray = img.convert("L")
    w, h = gray.size
    if h < 200:
        scale = max(2, 200 // max(h, 1))
        gray = gray.resize((w * scale, h * scale), Image.LANCZOS)
    enhanced = ImageEnhance.Contrast(gray).enhance(2.5)
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    try:
        text = pyt.image_to_string(enhanced, config=f"--psm {psm}")
        return text.strip()
    except Exception:
        return ""


def _ocr_product_name(img):
    """OCR a small region for product name, return cleaned name or None."""
    lines = _ocr_with_variants(img.convert("L"))
    name = _best_ocr_line(lines)
    return name


def _save_product(output_dir, pdf_name, name, thumb_img=None, app_img=None, pages=None):
    """Save product to disk."""
    pdf_dir = os.path.join(output_dir, _sanitize(pdf_name))
    os.makedirs(pdf_dir, exist_ok=True)
    
    folder = _sanitize(name) if name else "unknown"
    product_dir = os.path.join(pdf_dir, folder)
    os.makedirs(product_dir, exist_ok=True)
    
    has_thumb = thumb_img is not None
    has_app = app_img is not None
    
    if has_thumb:
        thumb = _resize_thumb(thumb_img)
        _save_jpg(thumb, os.path.join(product_dir, "thumbnail.jpg"))
    
    if has_app:
        _save_jpg(app_img, os.path.join(product_dir, "application.jpg"))
    
    meta = {
        "name": name or "unknown",
        "pages": pages or [],
        "has_thumbnail": has_thumb,
        "has_application": has_app,
    }
    with open(os.path.join(product_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    return meta


# ═══════════════════════════════════════════════════════════════
# PDF 1: DREAM HYDERABAD
# ═══════════════════════════════════════════════════════════════
def extract_dream_hyderabad(pdf_path, output_dir):
    """Every page: big application top, small thumbnail bottom-left, name below thumb."""
    print(f"\n{'='*60}")
    print(f"  DREAM HYDERABAD")
    print(f"{'='*60}")
    
    doc = fitz.open(pdf_path)
    n = len(doc)
    results = []
    
    for pn in range(n):
        img = render_page(pdf_path, pn, dpi=200)
        w, h = img.size
        img_np = np.array(img)
        
        rects = _find_large_rects(img_np, min_area_ratio=0.03)
        rects = _remove_nested_rects(rects)
        
        if len(rects) < 2:
            print(f"  P{pn+1}: skip (only {len(rects)} rects)")
            continue
        
        # Classify rects
        rect_data = []
        for rx, ry, rw, rh in rects:
            crop = img.crop((rx, ry, rx+rw, ry+rh))
            scores = clf.classify(crop)
            area = (rw * rh) / (w * h)
            rect_data.append({
                "bbox": (rx, ry, rw, rh),
                "image": crop,
                "scores": scores,
                "area": area,
            })
        
        # Sort by area: largest = application, smallest in bottom half = thumbnail
        by_area = sorted(rect_data, key=lambda r: r["area"], reverse=True)
        
        app_rect = by_area[0]  # largest = application (top)
        
        # Find thumbnail: small rect in bottom portion
        thumb_rect = None
        for r in by_area[1:]:
            rx, ry, rw, rh = r["bbox"]
            if ry > h * 0.5:  # bottom half
                thumb_rect = r
                break
        
        if not thumb_rect:
            thumb_rect = by_area[1]  # fallback: second largest
        
        # Extract
        app_img = crop_to_content(app_rect["image"])
        thumb_img = crop_to_content(thumb_rect["image"])
        thumb_img = _force_portrait(thumb_img)
        
        # OCR name: below thumbnail
        tx, ty, tw, th_r = thumb_rect["bbox"]
        name_top = ty + th_r
        name_bot = min(h, name_top + int(h * 0.10))
        name_region = img.crop((tx, name_top, tx + tw + 100, name_bot))
        name = _ocr_product_name(name_region)
        
        if not name:
            # Try wider region
            name_region2 = img.crop((0, name_top, w // 2, name_bot))
            name = _ocr_product_name(name_region2)
        
        if not name:
            name = f"PRODUCT_P{pn+1}"
        
        tag = "[TA]" if app_img and thumb_img else "[T-]"
        print(f"  {tag} P{pn+1}: {name}")
        
        meta = _save_product(output_dir, "DREAM HYDERABAD", name,
                            thumb_img, app_img, [pn+1])
        meta["name"] = name
        results.append(meta)
    
    doc.close()
    complete = sum(1 for r in results if r["has_thumbnail"] and r["has_application"])
    print(f"\n  -> {len(results)} products, {complete} complete")
    return results


# ═══════════════════════════════════════════════════════════════
# PDF 2: MOROCON GVT
# ═══════════════════════════════════════════════════════════════
def extract_morocon_gvt(pdf_path, output_dir):
    """Skip P1, classify at 200dpi: THUMB=2+ sq rects, APP=rect ≥40%. Group THUMBs→APP."""
    print(f"\n{'='*60}")
    print(f"  MOROCON GVT")
    print(f"{'='*60}")
    
    doc = fitz.open(pdf_path)
    n = len(doc)
    results = []
    
    # Step 1: Classify all pages at 200dpi (skip P1 = index 0)
    page_types = []  # 'thumb', 'app', or 'skip'
    for pn in range(n):
        if pn == 0:
            page_types.append('skip')
            print(f"  classify P{pn+1}: SKIP (intro)")
            continue
        
        img = render_page(pdf_path, pn, dpi=200)
        w, h = img.size
        img_np = np.array(img)
        rects = _find_large_rects(img_np, min_area_ratio=0.01)
        rects = _remove_nested_rects(rects)
        
        sq_count = 0
        has_large = False
        for rx, ry, rw, rh in rects:
            area = (rw * rh) / (w * h)
            aspect = rw / rh if rh > 0 else 0
            if 0.08 < area < 0.25 and 0.8 < aspect < 1.3:
                sq_count += 1
            if area >= 0.40:
                has_large = True
        
        if sq_count >= 2:
            page_types.append('thumb')
            print(f"  classify P{pn+1}: THUMB ({sq_count} sq)")
        elif has_large:
            page_types.append('app')
            print(f"  classify P{pn+1}: APP")
        elif sq_count == 1:
            # Single-thumbnail pages (like P7, P14 with 1 big + small previews)
            page_types.append('thumb')
            print(f"  classify P{pn+1}: THUMB (single)")
        else:
            page_types.append('skip')
            print(f"  classify P{pn+1}: SKIP")
    
    # Step 2: Group consecutive thumb pages → next app page
    i = 1  # start after P1
    while i < n:
        if page_types[i] != 'thumb':
            i += 1
            continue
        
        # Collect consecutive thumb pages
        thumb_pages = []
        while i < n and page_types[i] == 'thumb':
            thumb_pages.append(i)
            i += 1
        
        # Next app page
        app_pn = None
        if i < n and page_types[i] == 'app':
            app_pn = i
            i += 1
        
        # Render application image
        app_img = None
        if app_pn is not None:
            app_full = render_page(pdf_path, app_pn, dpi=200)
            app_img = crop_to_content(app_full)
        
        # Extract thumbnails from each thumb page
        for tpn in thumb_pages:
            img = render_page(pdf_path, tpn, dpi=200)
            tw, th = img.size
            img_np = np.array(img)
            rects = _find_large_rects(img_np, min_area_ratio=0.01)
            rects = _remove_nested_rects(rects)
            
            thumb_rects = []
            for rx, ry, rw, rh in rects:
                area = (rw * rh) / (tw * th)
                aspect = rw / rh if rh > 0 else 0
                if 0.08 < area < 0.25 and 0.8 < aspect < 1.3:
                    thumb_rects.append((rx, ry, rw, rh))
            
            thumb_rects.sort(key=lambda r: r[0])  # left to right
            
            if not thumb_rects:
                continue
            
            # OCR product name: above thumbnails
            name = None
            min_y = min(ry for rx, ry, rw, rh in thumb_rects)
            name_top = max(0, min_y - int(th * 0.15))
            name_bot = min_y
            name_region = img.crop((0, name_top, tw, name_bot))
            name = _ocr_product_name(name_region)
            if not name:
                name_region2 = img.crop((0, 0, tw, min_y))
                name = _ocr_product_name(name_region2)
            
            if not name:
                name = f"PRODUCT_P{tpn+1}"
            
            for ti, (rx, ry, rw, rh) in enumerate(thumb_rects):
                crop = img.crop((rx, ry, rx+rw, ry+rh))
                thumb_cropped = crop_to_content(crop)
                thumb_cropped = _force_portrait(thumb_cropped)
                
                suffix = f" {ti+1}" if len(thumb_rects) > 1 else ""
                prod_name = f"{name}{suffix}"
                
                pages = [tpn+1]
                if app_pn is not None:
                    pages.append(app_pn+1)
                
                tag = "[TA]" if app_img else "[T-]"
                print(f"  {tag} P{tpn+1}: {prod_name}")
                
                meta = _save_product(output_dir, "MOROCON GVT", prod_name,
                                    thumb_cropped, app_img, pages)
                meta["name"] = prod_name
                results.append(meta)
    
    doc.close()
    complete = sum(1 for r in results if r["has_thumbnail"] and r["has_application"])
    print(f"\n  -> {len(results)} products, {complete} complete")
    return results


# ═══════════════════════════════════════════════════════════════
# PDF 3: MILLANO CERAMICA
# ═══════════════════════════════════════════════════════════════
def extract_millano(pdf_path, output_dir):
    """Skip P1-6, P42-43. Strict triplets from P7: name/sample → thumbnails → application."""
    print(f"\n{'='*60}")
    print(f"  MILLANO CERAMICA")
    print(f"{'='*60}")
    
    doc = fitz.open(pdf_path)
    n = len(doc)
    results = []
    
    # Strict triplets from index 6 (P7) to index 40 (P41)
    start = 6   # P7
    end = 41    # P41 inclusive
    
    pn = start
    while pn + 1 <= end:
        name_pn = pn       # name/sample page (has product name text + texture)
        thumb_pn = pn + 1  # thumbnail page (2 tile rects)
        app_pn = pn + 2 if pn + 2 <= end else None  # application (full page)
        
        # Extract product name from name page text
        name = None
        page = doc[name_pn]
        text = page.get_text().strip()
        if text:
            # Name is before dimension like "600x1200MM"
            # eg: "DOSIMO WOOD BROWN 600x1200MM 2'x4' | 24\"x48\" GLOSSY..."
            # Also handle: "J-DELIGHT ONYX SKY HL J-DELIGHT ONYX SKY 600x1200MM..."
            m = re.match(r'^(.+?)\s+\d+\s*[xX×]\s*\d+', text.replace('\n', ' '))
            if m:
                raw = m.group(1).strip()
                # If " HL " appears, take before it
                if ' HL ' in raw:
                    raw = raw.split(' HL ')[0].strip()
                name = raw.upper()
        
        if not name:
            name = f"P{name_pn+1}"
        
        print(f"  triplet P{name_pn+1}-P{thumb_pn+1}" + 
              (f"-P{app_pn+1}" if app_pn else "") + f": {name}")
        
        # Extract thumbnails from thumb page
        if thumb_pn < n:
            img = render_page(pdf_path, thumb_pn, dpi=200)
            w, h = img.size
            img_np = np.array(img)
            rects = _find_large_rects(img_np, min_area_ratio=0.05)
            rects = _remove_nested_rects(rects)
            
            tile_rects = []
            for rx, ry, rw, rh in rects:
                area = (rw * rh) / (w * h)
                if area > 0.05:
                    tile_rects.append((rx, ry, rw, rh, area))
            
            # Sort by area descending, take top 2
            tile_rects.sort(key=lambda r: r[4], reverse=True)
            tile_rects = tile_rects[:2]
            # Sort left to right for consistent naming
            tile_rects.sort(key=lambda r: r[0])
            
            # Extract application image
            app_img = None
            if app_pn and app_pn < n:
                app_full = render_page(pdf_path, app_pn, dpi=200)
                app_img = crop_to_content(app_full)
            
            for ti, (rx, ry, rw, rh, _a) in enumerate(tile_rects):
                crop = img.crop((rx, ry, rx+rw, ry+rh))
                thumb = crop_to_content(crop)
                thumb = _force_portrait(thumb)
                
                suffix = f" {ti+1}" if len(tile_rects) > 1 else ""
                prod_name = f"{name}{suffix}"
                
                pages = [thumb_pn+1]
                if app_pn:
                    pages.append(app_pn+1)
                
                tag = "[TA]" if app_img else "[T-]"
                print(f"  {tag} P{thumb_pn+1}: {prod_name}")
                
                meta = _save_product(output_dir, "MILLANO CERAMICA", prod_name,
                                    thumb, app_img, pages)
                meta["name"] = prod_name
                results.append(meta)
        
        pn += 3  # next triplet
    
    doc.close()
    complete = sum(1 for r in results if r["has_thumbnail"] and r["has_application"])
    print(f"\n  -> {len(results)} products, {complete} complete")
    return results


# ═══════════════════════════════════════════════════════════════
# PDF 4: PARK COLLECTION
# ═══════════════════════════════════════════════════════════════
def extract_park_collection(pdf_path, output_dir):
    """Extract only application images + product names (COTTA X from top OCR)."""
    print(f"\n{'='*60}")
    print(f"  PARK COLLECTION")  
    print(f"{'='*60}")
    
    doc = fitz.open(pdf_path)
    n = len(doc)
    results = []
    
    for pn in range(n):
        img = render_page(pdf_path, pn, dpi=200)
        w, h = img.size
        
        # OCR top 20% to find product name
        top_area = img.crop((0, 0, w, int(h * 0.20)))
        top_text = pyt.image_to_string(top_area).strip()
        
        # Look for COTTA/GOTTA pattern (GOTTA is OCR misread of COTTA)
        name = None
        m = re.search(r'(?:COTTA|GOTTA|CQTTA)\s+([A-Z]+)', top_text, re.IGNORECASE)
        if m:
            name = f"COTTA {m.group(1).upper()}"
        
        if not name:
            # Also try bottom 20%
            bot_area = img.crop((0, int(h * 0.80), w, h))
            bot_text = pyt.image_to_string(bot_area).strip()
            m = re.search(r'(?:COTTA|GOTTA|CQTTA)\s+([A-Z]+)', bot_text, re.IGNORECASE)
            if m:
                name = f"COTTA {m.group(1).upper()}"
        
        if not name:
            # Retry at 150dpi (some pages read better at lower DPI)
            img_lo = render_page(pdf_path, pn, dpi=150)
            wl, hl = img_lo.size
            top_lo = img_lo.crop((0, 0, wl, int(hl * 0.20)))
            top_text_lo = pyt.image_to_string(top_lo).strip()
            m = re.search(r'(?:COTTA|GOTTA|CQTTA)\s+([A-Z]+)', top_text_lo, re.IGNORECASE)
            if m:
                name = f"COTTA {m.group(1).upper()}"
        
        if not name:
            print(f"  P{pn+1}: skip (no COTTA name found)")
            continue
        
        # This page is an application image with a COTTA product name
        app_img = crop_to_content(img)
        
        tag = "[A ]"
        print(f"  {tag} P{pn+1}: {name}")
        
        meta = _save_product(output_dir, "PARK COLLECTION", name,
                            None, app_img, [pn+1])
        meta["name"] = name
        results.append(meta)
    
    doc.close()
    print(f"\n  -> {len(results)} products (app only)")
    return results


# ═══════════════════════════════════════════════════════════════
# PDF 5: 800x800 GLOSSY CATALOGUE
# ═══════════════════════════════════════════════════════════════
def extract_glossy_catalogue(pdf_path, output_dir):
    """Skip P1-7 intro. Triplets from P8: presentation → spec(tile thumbnail) → floor-view.
    Page A (pres): room scene in large rect, but it often shows the PREVIOUS product's tile.
    Page B (spec): square tile close-up on right side (~32% area) → thumbnail.
    Page C (floor): "FLOOR / PRODUCT_NAME" page showing the product's tile → fallback application.

    Strategy: For each product, pick the best-matching application image by comparing
    the thumbnail texture against candidate room scenes using histogram correlation.
    Candidates: NEXT product's Page A room scene (most common) and own Page C.
    """
    print(f"\n{'='*60}")
    print(f"  800x800 GLOSSY CATALOGUE")
    print(f"{'='*60}")

    doc = fitz.open(pdf_path)
    n = len(doc)

    # ── helpers ──────────────────────────────────────────────
    def _hist_corr(i1, i2):
        """HSV histogram correlation between two PIL images."""
        a1 = cv2.cvtColor(np.array(i1.resize((256, 256))), cv2.COLOR_RGB2HSV)
        a2 = cv2.cvtColor(np.array(i2.resize((256, 256))), cv2.COLOR_RGB2HSV)
        h1 = cv2.calcHist([a1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        h2 = cv2.calcHist([a2], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(h1, h1); cv2.normalize(h2, h2)
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

    def _floor_patch(img):
        w, h = img.size
        return img.crop((w // 6, h // 2, w * 5 // 6, h))

    def _extract_room_rect(page_idx):
        """Extract the biggest rect (>40% area) from a presentation page."""
        for dd in [150, 100]:
            lo = render_page(pdf_path, page_idx, dpi=dd)
            pw, ph = lo.size
            arr = np.array(lo)
            rects = _find_large_rects(arr, min_area_ratio=0.05)
            rects = _remove_nested_rects(rects)
            best = None
            for rx, ry, rw, rh in rects:
                a = (rw * rh) / (pw * ph)
                if a > 0.40 and (best is None or a > best[4]):
                    best = (rx, ry, rw, rh, a)
            if best:
                rx, ry, rw, rh, _ = best
                hi = render_page(pdf_path, page_idx, dpi=200)
                s = hi.size[0] / pw
                crop = hi.crop((int(rx*s), int(ry*s), int((rx+rw)*s), int((ry+rh)*s)))
                return crop_to_content(crop)
        return None

    # ── first pass: collect all triplet data ─────────────────
    triplets = []          # (pres_pn, spec_pn, var_pn, name)
    pn = 7
    while pn + 2 < n:
        pres_pn, spec_pn, var_pn = pn, pn + 1, pn + 2
        page = doc[pres_pn]
        text = page.get_text().strip()
        name = None
        if text:
            m = re.search(r'800\s*(?:X\s*800\s*)?MM\s*\n?\s*(.+)', text, re.IGNORECASE)
            if m:
                raw = m.group(1).strip().split('\n')[0].strip()
                if raw and len(raw) > 1:
                    name = raw.upper()
        if not name:
            name = f"P{pres_pn+1}"
        triplets.append((pres_pn, spec_pn, var_pn, name))
        pn += 3
    doc.close()

    # ── extract thumbnails, room scenes, page-C images ───────
    thumbs = []        # thumbnail PIL images
    rooms_A = []       # room scene from each product's Page A
    pages_C = []       # full-page image from Page C (floor view)

    for pres_pn, spec_pn, var_pn, name in triplets:
        # Thumbnail from spec page
        lo = render_page(pdf_path, spec_pn, dpi=100)
        wl, hl = lo.size
        arr = np.array(lo)
        rects = _find_large_rects(arr, min_area_ratio=0.01)
        rects = _remove_nested_rects(rects)
        hi = render_page(pdf_path, spec_pn, dpi=200)
        sc = hi.size[0] / wl
        thumb = None
        for rx, ry, rw, rh in rects:
            area = (rw * rh) / (wl * hl)
            asp = rw / rh if rh > 0 else 0
            if 0.8 < asp < 1.25 and area > 0.15:
                crop = hi.crop((int(rx*sc), int(ry*sc), int((rx+rw)*sc), int((ry+rh)*sc)))
                thumb = crop_to_content(crop)
                thumb = _force_portrait(thumb)
                break
        thumbs.append(thumb)

        # Room scene from Page A
        rooms_A.append(_extract_room_rect(pres_pn))

        # Page C full image
        pc = render_page(pdf_path, var_pn, dpi=200)
        pages_C.append(crop_to_content(pc))

    # ── second pass: pick best application for each product ──
    results = []
    for i, (pres_pn, spec_pn, var_pn, name) in enumerate(triplets):
        thumb_img = thumbs[i]

        # Score candidates: NEXT Page A room, own Page C
        next_room = rooms_A[i + 1] if i + 1 < len(triplets) else None
        own_C = pages_C[i]

        app_img = None
        app_src = "?"

        if thumb_img is not None:
            c_next = _hist_corr(thumb_img, _floor_patch(next_room)) if next_room else -1
            c_ownC = _hist_corr(thumb_img, _floor_patch(own_C)) if own_C else -1

            if c_next > c_ownC:
                app_img = next_room
                app_src = f"nextA P{triplets[i+1][0]+1}"
            else:
                app_img = own_C
                app_src = f"ownC P{var_pn+1}"
        else:
            # No thumbnail to compare — fall back to own Page C
            app_img = own_C
            app_src = f"ownC P{var_pn+1}"

        tag = "[TA]" if thumb_img and app_img else "[T-]" if thumb_img else "[A ]"
        print(f"  {name:24s} {tag}  app={app_src}")

        pages = [spec_pn + 1, var_pn + 1]
        meta = _save_product(output_dir, "800X800 GLOSSY CATALOGUE", name,
                             thumb_img, app_img, pages)
        meta["name"] = name
        results.append(meta)

    complete = sum(1 for r in results if r["has_thumbnail"] and r["has_application"])
    print(f"\n  -> {len(results)} products, {complete} complete")
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    OUTPUT = "products_new_test"
    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)
    os.makedirs(OUTPUT)
    
    r5 = extract_glossy_catalogue(
        "catalogues/800x800 Glossy Catalogue 2022 - Copy.pdf", OUTPUT)
    
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for label, res in [("GLOSSY", r5)]:
        total = len(res)
        complete = sum(1 for r in res if r.get("has_thumbnail") and r.get("has_application"))
        app_only = sum(1 for r in res if not r.get("has_thumbnail") and r.get("has_application"))
        print(f"  {label:20s}: {total:>3d} products, {complete:>3d} complete, {app_only:>3d} app-only")
