# Digital Catalogue Platform

Automated system for extracting product images from PDF catalogues and organizing them into product folders.

## Folder Structure

```
├── catalogues/       → Input PDF catalogues
├── images_raw/       → Extracted images (straight from PDFs)
├── images_grouped/   → Images grouped by product/page
├── products/         → Final organized product folders
├── scripts/          → Python automation scripts
├── logs/             → Processing logs and reports
├── config/           → Configuration files
```

## Folder Descriptions

| Folder | Purpose |
|---|---|
| `catalogues/` | Drop your source PDF catalogue files here. The scripts will read from this folder. |
| `images_raw/` | Raw images extracted directly from the PDFs, before any sorting or processing. |
| `images_grouped/` | Intermediate stage — images grouped by product, page, or category after analysis. |
| `products/` | Final output — each subfolder represents one product, containing its cleaned images. |
| `scripts/` | All Python automation scripts (extraction, grouping, renaming, etc.). |
| `logs/` | Timestamped log files tracking each processing run, errors, and statistics. |
| `config/` | Configuration files (e.g., extraction settings, naming rules, folder mappings). |
