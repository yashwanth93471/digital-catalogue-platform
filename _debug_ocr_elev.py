"""Debug OCR of composite text area on Elevation page 2."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pytesseract as pyt
from PIL import Image, ImageEnhance, ImageFilter
from pipeline.composite_segmenter import analyze_page_layout
from pipeline.clip_classifier import CLIPClassifier

pyt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
clf = CLIPClassifier()
pdf = "catalogues/18x12 Elevation.pdf"
out = "_elev_page2"

info = analyze_page_layout(pdf, 1, clf)
region = info["content_regions"][0]["image"]
rw, rh = region.size
print(f"Region: {rw}x{rh}")

# Text area: left 40%
text_crop = region.crop((0, 0, int(rw * 0.40), rh))
tw, th = text_crop.size
print(f"Text crop: {tw}x{th}")
text_crop.save(f"{out}/text_area.png")

# Upscale
gray = text_crop.convert("L")
scale = max(2, 400 // max(th, 1))
gray_up = gray.resize((tw * scale, th * scale), Image.LANCZOS)

# image_to_data
data = pyt.image_to_data(gray_up, output_type=pyt.Output.DICT)
line_groups = {}
for i in range(len(data["text"])):
    word = data["text"][i].strip()
    if not word or int(data["conf"][i]) < 30:
        continue
    key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
    if key not in line_groups:
        line_groups[key] = {"words": [], "heights": [], "confs": []}
    line_groups[key]["words"].append(word)
    line_groups[key]["heights"].append(data["height"][i])
    line_groups[key]["confs"].append(int(data["conf"][i]))

print(f"\nFound {len(line_groups)} text lines:")
candidates = []
for key, info_d in line_groups.items():
    text = " ".join(info_d["words"])
    avg_h = float(np.mean(info_d["heights"]))
    avg_conf = float(np.mean(info_d["confs"]))
    candidates.append((text, avg_h, avg_conf))
    print(f"  [{key}] h={avg_h:.0f}, conf={avg_conf:.0f}: '{text}'")

candidates.sort(key=lambda c: c[1], reverse=True)
print(f"\nSorted by height (largest first):")
for text, h, conf in candidates:
    print(f"  h={h:.0f}: '{text}'")

# Also try standard OCR
print("\nStandard OCR lines:")
v1 = ImageEnhance.Contrast(gray_up).enhance(2.0)
v1 = v1.filter(ImageFilter.SHARPEN)
text_out = pyt.image_to_string(v1)
for line in text_out.split("\n"):
    if line.strip():
        print(f"  '{line.strip()}'")
