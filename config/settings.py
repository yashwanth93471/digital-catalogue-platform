"""
Central configuration for the Catalogue PDF Automation System.
All paths and settings are defined here.
"""

import os

# ─── Project Root ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Folder Paths ──────────────────────────────────────────────
CATALOGUES_DIR = os.path.join(PROJECT_ROOT, "catalogues")
IMAGES_RAW_DIR = os.path.join(PROJECT_ROOT, "images_raw")
IMAGES_GROUPED_DIR = os.path.join(PROJECT_ROOT, "images_grouped")
PRODUCTS_DIR = os.path.join(PROJECT_ROOT, "products")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ─── External Tool Paths ───────────────────────────────────────
POPPLER_PATH = r"C:\poppler\poppler-24.08.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ─── Image Settings ────────────────────────────────────────────
IMAGE_DPI = 300
IMAGE_FORMAT = "png"
