"""Try different OCR preprocessing on the text area of Elevation page 2."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import numpy as np
import pytesseract as pyt
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pipeline.composite_segmenter import render_page, analyze_page_layout
from pipeline.clip_classifier import CLIPClassifier

pyt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
clf = CLIPClassifier()
pdf = "catalogues/18x12 Elevation.pdf"
out = "_elev_page2"

# Render at HIGHER DPI for better OCR
from pipeline.composite_segmenter import render_page as rp
page_img_300 = rp(pdf, 1, dpi=300)
pw, ph = page_img_300.size
print(f"Page 2 at 300dpi: {pw}x{ph}")

# The text area is roughly left 35%
text_area = page_img_300.crop((0, 0, int(pw * 0.35), ph))
tw, th = text_area.size
print(f"Text area: {tw}x{th}")
text_area.save(f"{out}/text_300dpi.png")

# Try multiple preprocessing approaches
gray = np.array(text_area.convert("L"))

# 1. High contrast
v1 = ImageEnhance.Contrast(text_area.convert("L")).enhance(3.0)
v1 = v1.filter(ImageFilter.SHARPEN)
text1 = pyt.image_to_string(v1)
print(f"\n1. High contrast:")
for l in text1.split("\n"):
    if l.strip():
        print(f"  '{l.strip()}'")

# 2. Otsu binarization
_, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
text2 = pyt.image_to_string(Image.fromarray(binarized))
print(f"\n2. Otsu binary:")
for l in text2.split("\n"):
    if l.strip():
        print(f"  '{l.strip()}'")

# 3. Adaptive thresholding
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 51, 10)
text3 = pyt.image_to_string(Image.fromarray(adaptive))
print(f"\n3. Adaptive threshold:")
for l in text3.split("\n"):
    if l.strip():
        print(f"  '{l.strip()}'")

# 4. Invert
inverted = ImageOps.invert(text_area.convert("L"))
v4 = ImageEnhance.Contrast(inverted).enhance(3.0)
text4 = pyt.image_to_string(v4)
print(f"\n4. Inverted + high contrast:")
for l in text4.split("\n"):
    if l.strip():
        print(f"  '{l.strip()}'")

# 5. Focus on just the middle vertical area where name appears
# Product name appears roughly in the center-left, between 20-60% of height
name_area = text_area.crop((0, int(th*0.20), tw, int(th*0.65)))
name_area.save(f"{out}/name_area_300dpi.png")
nw, nh = name_area.size
# Upscale
scale = max(1, 600 // max(nh, 1))
name_big = name_area.resize((nw * scale, nh * scale), Image.LANCZOS)
gray_name = np.array(name_big.convert("L"))

# Contrast enhance
v5 = ImageEnhance.Contrast(name_big.convert("L")).enhance(3.0)
v5 = v5.filter(ImageFilter.SHARPEN)
text5 = pyt.image_to_string(v5)
print(f"\n5. Name area (20-65% h), contrast+sharpen:")
for l in text5.split("\n"):
    if l.strip():
        print(f"  '{l.strip()}'")

# Otsu on name area
_, bname = cv2.threshold(gray_name, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
text6 = pyt.image_to_string(Image.fromarray(bname))
print(f"\n6. Name area, Otsu:")
for l in text6.split("\n"):
    if l.strip():
        print(f"  '{l.strip()}'")

# 7. Try PSM 6 (block of text) and PSM 7 (single line) on name area
text7 = pyt.image_to_string(v5, config="--psm 6")
print(f"\n7. Name area, PSM 6:")
for l in text7.split("\n"):
    if l.strip():
        print(f"  '{l.strip()}'")

# 8. Try a very focused crop: just the big text area
# From the image, the product name "LOTUS CHOCO" is roughly at y=30-55%, x=5-80% of text area  
focus = text_area.crop((int(tw*0.05), int(th*0.30), int(tw*0.85), int(th*0.55)))
focus.save(f"{out}/focus_name.png")
fw, fh = focus.size
focus_big = focus.resize((fw*3, fh*3), Image.LANCZOS)
gray_focus = np.array(focus_big.convert("L"))
_, bfocus = cv2.threshold(gray_focus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
text8 = pyt.image_to_string(Image.fromarray(bfocus), config="--psm 6")
print(f"\n8. Focus on name text, Otsu + PSM 6:")
for l in text8.split("\n"):
    if l.strip():
        print(f"  '{l.strip()}'")

# 9. Try with specific threshold (the text might be dark brown ~120-150 on cream ~220-240)
mask = (gray < 180).astype(np.uint8) * 255
text9 = pyt.image_to_string(Image.fromarray(mask))
print(f"\n9. Threshold < 180:")
for l in text9.split("\n"):
    if l.strip():
        print(f"  '{l.strip()}'")
