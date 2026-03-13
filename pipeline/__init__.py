"""
CLIP-powered Catalogue Image Extraction Pipeline
=================================================

Libraries & their roles
-----------------------
  PyMuPDF (fitz)       - Parse PDF: extract embedded images with page
                         coordinates, extract text blocks with font sizes
                         and positions for product name detection.

  OpenCV (cv2)         - Image preprocessing: grayscale conversion, Canny
                         edge detection, contour analysis for composite
                         image segmentation, color-space transforms,
                         thresholding for OCR fallback enhancement.

  NumPy                - Array operations for fast pixel-level analysis
                         (blank detection, color statistics).

  Pillow (PIL)         - Image format conversion, resizing with Lanczos
                         resampling, contrast/sharpness enhancement for OCR.

  transformers (CLIP)  - Zero-shot image classification via
                         openai/clip-vit-base-patch32: classifies every
                         extracted image as texture-slab, room-scene,
                         preview-tile, logo, or diagram WITHOUT any
                         catalogue-specific training data.
                         Also provides image embeddings for cosine-similarity
                         matching between texture thumbnails and room-scene
                         application images.

  LayoutParser         - (Optional) Deep-learning document layout analysis.
                         When available, detects title / figure / text regions
                         on rendered pages to improve product-name extraction
                         accuracy. Falls back to PyMuPDF text-block positions
                         and font sizes when not installed.

Pipeline Modules
----------------
  pdf_type_detector    - Detect PDF structure: multi-image (Type A) or
                         flat-page composite (Type B) per-page detection.

  pdf_parser           - Extract embedded images and text blocks using
                         PyMuPDF.

  image_filter         - Remove blank/background images + detect repeated
                         logos/watermarks via perceptual hashing.

  clip_classifier      - Zero-shot CLIP classification and embedding
                         similarity matching.

  composite_segmenter  - Segment composite full-page catalogue images
                         into texture + room-scene regions (contour
                         detection + layout heuristics + CLIP verification).

  name_extractor       - Multi-strategy product name extraction (text blocks,
                         LayoutParser, multi-region OCR).

  product_assembler    - Adaptive assembler: routes to Type A or Type B
                         processing pipeline based on PDF structure.
"""
