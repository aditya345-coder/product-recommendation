from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class OccasionRecommender:
    """Semantic product recommendation engine using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.products: List[Dict[str, Any]] = []
        self.product_embeddings: Optional[np.ndarray] = None

    def load_model(self) -> None:
        """Load the sentence-transformers model."""
        print(f"Loading model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded successfully.")

    def build_index(self, products: List[Dict[str, Any]]) -> None:
        """Build product embedding index from catalog."""
        self.products = products

        if not products:
            print("Warning: Empty product catalog.")
            self.product_embeddings = np.array([])
            return

        # Extract semantic texts
        texts = [p.get("semantic_text", "") for p in products]

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} products...")
        self.product_embeddings = self.model.encode(texts, show_progress_bar=True)

        # Normalize embeddings for cosine similarity via dot product
        norms = np.linalg.norm(self.product_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        self.product_embeddings = self.product_embeddings / norms

        print(f"Embedding matrix shape: {self.product_embeddings.shape}")

    def recommend(
        self,
        occasion: str,
        top_k: int = 12,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Get top-K product recommendations for an occasion query."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not occasion or not occasion.strip():
            return []

        if self.product_embeddings is None or len(self.products) == 0:
            return []

        # Generate query embedding
        query_embedding = self.model.encode([occasion.strip()])

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_norm = np.where(query_norm == 0, 1, query_norm)
        query_embedding = query_embedding / query_norm

        # Calculate cosine similarity via dot product (embeddings already normalized)
        scores = self.product_embeddings @ query_embedding.T
        scores = scores.flatten()

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if min_score is not None and score < min_score:
                continue

            product = self.products[idx].copy()
            product["score"] = round(score, 4)
            results.append(product)

        return results

    def get_health_info(self) -> Dict[str, Any]:
        """Return health/status information."""
        return {
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
            "catalog_count": len(self.products),
            "embeddings_ready": self.product_embeddings is not None,
        }
