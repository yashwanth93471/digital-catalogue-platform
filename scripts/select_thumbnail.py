"""
Classify tile catalogue images into TEXTURE (slab) vs APPLICATION (room scene)
using strict, data-calibrated computer-vision rules.

Classification approach
───────────────────────
  1. Compute 6 features per image (edge density, edge uniformity,
     entropy, pattern repetition, spatial variance, color diversity).
  2. Assign a composite ``texture_score`` (0–100, higher = texture).
  3. Apply ABSOLUTE thresholds for hard classification:
       texture_score >= 55  ->  TEXTURE   (thumbnail)
       texture_score <  55  ->  APPLICATION (room scene)
  4. Single-image pages use absolute classification (not auto-thumbnail).
  5. Multi-image pages: best texture = thumbnail; rest = applications.
     If NO image qualifies as texture, the page gets NO thumbnail.

Calibration data (from real catalogue analysis)
───────────────────────────────────────────────
  Textures:     ed<0.025  ent<5.0  sv<2.0  cs<10   var<65
  Scenes:       ed>0.027  ent>5.9  sv>17   cs>45   var>2100
  Gap zone:     5.0 < ent < 5.9  — rare but handled by composite score

Output
──────
  thumbnail.jpg           — standardised to 1528x3150 (portrait slab)
  application1.jpg, …     — kept at original size
  Pages with only scenes  — all become application images, no thumbnail

Usage:
    python scripts/select_thumbnail.py
    python scripts/select_thumbnail.py path/to/grouped_dir
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.settings import IMAGES_GROUPED_DIR

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Standard thumbnail output size (width x height) — portrait slab format.
THUMBNAIL_WIDTH = 1528
THUMBNAIL_HEIGHT = 3150

# Absolute threshold: images scoring below this are APPLICATION, not TEXTURE.
TEXTURE_THRESHOLD = 55.0


# ── Feature extraction ─────────────────────────────────────────

def _edge_features(gray: np.ndarray) -> tuple[float, float]:
    """Return (edge_density, edge_uniformity_std) from Canny."""
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    density = np.count_nonzero(edges) / edges.size

    bh, bw = h // 4, w // 4
    block_densities = []
    for r in range(4):
        for c in range(4):
            blk = edges[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            block_densities.append(np.count_nonzero(blk) / blk.size)
    uniformity_std = float(np.std(block_densities))

    return density, uniformity_std


def _entropy(gray: np.ndarray) -> float:
    """Shannon entropy of the grayscale histogram (bits)."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    nonzero = hist[hist > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _pattern_repetition(gray: np.ndarray) -> float:
    """Correlation between top-half and bottom-half.

    Tile textures have high correlation; room scenes vary top-to-bottom.
    """
    h = gray.shape[0]
    mid = h // 2
    top = gray[:mid, :].astype(np.float64).flatten()
    bot = gray[mid:mid * 2, :].astype(np.float64).flatten()
    min_len = min(len(top), len(bot))
    top, bot = top[:min_len], bot[:min_len]

    if np.std(top) < 1e-6 or np.std(bot) < 1e-6:
        return 1.0

    return float(np.abs(np.corrcoef(top, bot)[0, 1]))


def _spatial_variance(gray: np.ndarray) -> float:
    """Std of mean intensities across a 4x4 grid.

    Low = uniform surface (texture); high = varied scene (application).
    """
    h, w = gray.shape
    bh, bw = h // 4, w // 4
    means = []
    for r in range(4):
        for c in range(4):
            blk = gray[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            means.append(float(np.mean(blk)))
    return float(np.std(means))


def _color_diversity(img_bgr: np.ndarray) -> float:
    """Mean per-channel std deviation. Low = narrow palette (texture)."""
    return float(np.mean([np.std(img_bgr[:, :, c].astype(np.float64))
                          for c in range(3)]))


# ── Combined classifier ───────────────────────────────────────

def classify_image(img_path: str) -> dict:
    """Compute all features and a composite texture score.

    Returns dict with metrics and ``texture_score`` (0-100).
    Higher = more likely a tile texture / slab.
    ``is_texture`` is True when the score meets the absolute threshold.
    """
    img = cv2.imread(img_path)
    if img is None:
        return {"texture_score": -999.0, "is_texture": False, "error": "unreadable"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    edge_density, edge_uniformity = _edge_features(gray)
    entropy = _entropy(gray)
    repetition = _pattern_repetition(gray)
    spatial_var = _spatial_variance(gray)
    color_div = _color_diversity(img)

    # ── Sub-scores: each rewards texture-like values ───────────
    # Calibrated from real catalogue data:
    #   Textures:  ed<0.025  eu<0.005  ent<5.0  sv<2.0  cs<10
    #   Scenes:    ed>0.027  eu>0.015  ent>5.9  sv>17   cs>45

    # Edge density: lower -> more texture-like (weight 25)
    s_edge = max(0.0, 1.0 - edge_density / 0.08) * 25

    # Edge uniformity: lower std -> more texture-like (weight 15)
    s_edge_uni = max(0.0, 1.0 - edge_uniformity / 0.07) * 15

    # Entropy: lower -> more texture-like (weight 25)
    s_entropy = max(0.0, 1.0 - entropy / 8.0) * 25

    # Pattern repetition: higher corr -> more texture-like (weight 10)
    s_repeat = repetition * 10

    # Spatial variance: lower -> more texture-like (weight 15)
    s_spatial = max(0.0, 1.0 - spatial_var / 60.0) * 15

    # Color diversity: lower -> more texture-like (weight 10)
    s_color = max(0.0, 1.0 - color_div / 85.0) * 10

    texture_score = s_edge + s_edge_uni + s_entropy + s_repeat + s_spatial + s_color
    is_texture = texture_score >= TEXTURE_THRESHOLD

    # Hard override: if spatial variance AND color diversity are both
    # very low, the image is definitely a uniform texture surface —
    # even if edge density is high (e.g. sparkle/grain patterns).
    # Room scenes always have spatial_var > 15 and color_div > 40.
    if spatial_var < 5.0 and color_div < 20.0:
        is_texture = True

    return {
        "width": w,
        "height": h,
        "edge_density": round(edge_density, 5),
        "edge_uniformity": round(edge_uniformity, 5),
        "entropy": round(entropy, 3),
        "repetition": round(repetition, 4),
        "spatial_var": round(spatial_var, 2),
        "color_diversity": round(color_div, 2),
        "texture_score": round(texture_score, 2),
        "is_texture": is_texture,
    }


# ── Thumbnail standardisation ─────────────────────────────────

def _standardise_thumbnail(img: np.ndarray,
                           target_w: int = THUMBNAIL_WIDTH,
                           target_h: int = THUMBNAIL_HEIGHT) -> np.ndarray:
    """Resize a texture image to the standard slab dimensions."""
    if target_w is None or target_h is None:
        return img

    h, w = img.shape[:2]
    if w == target_w and h == target_h:
        return img

    interp = cv2.INTER_AREA if (w * h > target_w * target_h) else cv2.INTER_LANCZOS4
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


# ── Per-page classification ───────────────────────────────────

def _detect_variant_tiles(textures: list, applications: list) -> tuple:
    """Identify small variant/pattern tiles that should be discarded.

    On catalogue pages with a room scene + multiple small texture samples
    (e.g., 3 pattern variant tiles at the bottom), the small textures are
    just display variants, not the main product thumbnail.

    Returns:
        (best_texture, keep_applications, discarded_count)
    """
    if len(textures) <= 1 or not applications:
        return textures, applications, 0

    # Find the largest texture by pixel area
    for f, r in textures:
        r["_area"] = r["width"] * r["height"]

    textures_sorted = sorted(textures, key=lambda x: x[1]["_area"], reverse=True)
    largest_area = textures_sorted[0][1]["_area"]

    # Check if there are multiple small textures of similar size
    # (variant tiles are typically all the same dimensions)
    small_textures = [(f, r) for f, r in textures_sorted[1:]
                      if r["_area"] < largest_area * 0.5]

    if len(small_textures) >= 2:
        # Multiple small textures alongside a large one + room scenes
        # -> the small textures are variant tiles, discard them
        sizes = [r["_area"] for _, r in small_textures]
        max_size = max(sizes)
        min_size = min(sizes)
        # Variant tiles are typically same size (within 20% of each other)
        if min_size > max_size * 0.7:
            keep_textures = [textures_sorted[0]]
            return keep_textures, applications, len(small_textures)

    return textures, applications, 0


def select_thumbnail(page_dir: str) -> None:
    """Classify images in a page folder into thumbnail + applications.

    Rules:
      - Each image gets an absolute is_texture label (score >= 55).
      - If multiple textures exist, the highest-scoring one is the thumbnail.
      - Small variant tiles (multiple similar-sized textures on a page with
        room scenes) are detected and discarded.
      - All non-texture images become application1.jpg, application2.jpg, ...
      - If NO image is a texture, the page has no thumbnail (all are applications).
      - Single-image pages still use absolute classification.
    """
    files = sorted(
        f for f in os.listdir(page_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    )

    if not files:
        return

    page_name = os.path.basename(page_dir)

    # Classify every image
    results = {}
    for f in files:
        path = os.path.join(page_dir, f)
        results[f] = classify_image(path)

    # Separate textures from applications using absolute threshold
    textures = [(f, r) for f, r in results.items() if r["is_texture"]]
    applications = [(f, r) for f, r in results.items() if not r["is_texture"]]

    # Detect and discard small variant tiles
    textures, applications, discarded = _detect_variant_tiles(textures, applications)

    # Sort textures by score descending -- best one becomes thumbnail
    textures.sort(key=lambda x: x[1]["texture_score"], reverse=True)

    thumbnail_file = None
    if textures:
        thumbnail_file = textures[0][0]
        # Any extra textures beyond the first also become applications
        for f, r in textures[1:]:
            applications.append((f, r))

    # Sort applications by filename for deterministic ordering
    applications.sort(key=lambda x: x[0])

    # ── Write thumbnail ────────────────────────────────────────
    if thumbnail_file is not None:
        src = os.path.join(page_dir, thumbnail_file)
        dst = os.path.join(page_dir, "thumbnail.jpg")
        img = cv2.imread(src)
        img = _standardise_thumbnail(img)
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        os.remove(src)

        score = results[thumbnail_file]["texture_score"]
        extra = f" [{discarded} variant tiles removed]" if discarded else ""
        print(f"  {page_name}/ -> thumbnail.jpg (score={score:.1f}){extra}", end="")
    else:
        print(f"  {page_name}/ -> NO thumbnail (all images are scenes)", end="")

    # ── Write application images (original size) ───────────────
    if applications:
        for idx, (app_file, _) in enumerate(applications, start=1):
            src = os.path.join(page_dir, app_file)
            dst = os.path.join(page_dir, f"application{idx}.jpg")
            img = cv2.imread(src)
            cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            os.remove(src)
        app_names = ", ".join(f"application{i}.jpg" for i in range(1, len(applications) + 1))
        print(f" + {app_names}")
    else:
        print()

    # ── Clean up discarded variant tile source files ───────────
    if discarded:
        for f in os.listdir(page_dir):
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS:
                if f != "thumbnail.jpg" and not f.startswith("application"):
                    os.remove(os.path.join(page_dir, f))


# ── Batch processing ──────────────────────────────────────────

def process_all(grouped_dir: str) -> None:
    """Process all page folders under grouped_dir."""
    if not os.path.isdir(grouped_dir):
        print(f"[ERROR] Folder not found: {grouped_dir}")
        return

    page_dirs = sorted(
        d for d in os.listdir(grouped_dir)
        if os.path.isdir(os.path.join(grouped_dir, d))
    )

    print(f"[INFO] Classifying images in {len(page_dirs)} page folders")
    print(f"[INFO] Thumbnail standard size: {THUMBNAIL_WIDTH}x{THUMBNAIL_HEIGHT}")
    print(f"[INFO] Texture threshold: {TEXTURE_THRESHOLD}")
    print("-" * 60)

    for page_dir_name in page_dirs:
        page_path = os.path.join(grouped_dir, page_dir_name)
        select_thumbnail(page_path)

    print("-" * 60)
    print("[DONE] Classification complete.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else IMAGES_GROUPED_DIR
    process_all(target)
