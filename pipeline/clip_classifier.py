"""
CLIP-based zero-shot image classification and similarity matching.

Uses ``openai/clip-vit-base-patch32`` from HuggingFace transformers.

How it works
────────────
  1. Five natural-language category descriptions are defined.
  2. For each image, CLIP computes the cosine similarity between the
     image embedding and every text description.
  3. Softmax over those similarities gives a probability distribution.
  4. The category with the highest probability is the predicted label.

For texture-to-application matching the raw image embeddings are
compared directly (cosine similarity) to find the room scene whose
tile pattern is most visually similar to a given texture slab.
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

MODEL_NAME = "openai/clip-vit-base-patch32"

# Carefully crafted descriptions -- CLIP matches images to these prompts.
CATEGORY_LABELS = {
    "texture": (
        "a full frame closeup photograph of a natural stone marble "
        "granite ceramic tile material surface slab showing texture "
        "pattern with veining grain or repeating detail filling the "
        "entire image with no furniture or room context"
    ),
    "room_scene": (
        "a wide-angle interior room scene photograph of a modern "
        "kitchen bathroom living room bedroom hallway showing tile "
        "flooring or wall tiles installed in a real furnished space "
        "with furniture lighting and natural perspective"
    ),
    "preview_tile": (
        "a row of small square tile color swatch preview samples "
        "arranged together showing different colour or pattern "
        "variations of the same tile"
    ),
    "logo": (
        "a small centered corporate brand logo trademark symbol "
        "emblem icon with company name text in a designed typeface "
        "as a graphic identity mark"
    ),
    "diagram": (
        "a technical dimensions measurement specification diagram "
        "drawing blueprint showing tile sizes and layout patterns"
    ),
}


class CLIPClassifier:
    """Zero-shot image classifier powered by OpenAI CLIP."""

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[CLIP] Loading model: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()

        self._label_keys = list(CATEGORY_LABELS.keys())
        self._label_texts = list(CATEGORY_LABELS.values())
        print(f"[CLIP] Ready. Categories: {self._label_keys}")

    # ── classification ─────────────────────────────────────────

    def classify(self, pil_image: Image.Image) -> dict[str, float]:
        """Return ``{category: probability}`` for *pil_image*."""
        inputs = self.processor(
            text=self._label_texts,
            images=pil_image,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            out = self.model(**inputs)
        probs = out.logits_per_image.softmax(dim=1)[0]
        return {k: float(probs[i]) for i, k in enumerate(self._label_keys)}

    def get_category(self, pil_image: Image.Image) -> tuple[str, float]:
        """Return ``(best_category, confidence)``."""
        scores = self.classify(pil_image)
        best = max(scores, key=scores.get)
        return best, scores[best]

    # ── embedding / similarity ─────────────────────────────────

    def get_embedding(self, pil_image: Image.Image) -> torch.Tensor:
        """Return a unit-normalised CLIP image embedding (512-d)."""
        inputs = self.processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        vec = features[0]
        return vec / vec.norm()

    @staticmethod
    def similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
        """Cosine similarity between two normalised embeddings."""
        return float(torch.dot(emb_a, emb_b))

    def find_best_match(
        self,
        reference: Image.Image,
        candidates: list[tuple[int, Image.Image]],
    ) -> tuple[int | None, float]:
        """Return (index, similarity) for the candidate most visually
        similar to *reference*.  Useful for pairing a texture with the
        room scene that contains its pattern.
        """
        if not candidates:
            return None, 0.0
        ref_emb = self.get_embedding(reference)
        best_idx, best_sim = None, -1.0
        for idx, cimg in candidates:
            sim = self.similarity(ref_emb, self.get_embedding(cimg))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        return best_idx, best_sim
