import faiss
import numpy as np
import logging
import json
from app.embedding.model_loader import embedding_model

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class CodeSearchEngine:
    def __init__(self, vector_index_path: str, metadata_path: str):
        try:
            logger.info("🔁 Loading FAISS index and metadata...")
            self.index = faiss.read_index(vector_index_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info("✅ Search engine initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize search engine: {e}")
            raise

    def search(self, query: str, top_k: int = 3):
        logger.info(f"🔍 Searching for: {query}")
        vector = embedding_model.encode(query)

        if vector is None:
            logger.warning("⚠️ Query produced no vector. Aborting search.")
            return []

        vector = np.array([vector], dtype="float32")
        distances, indices = self.index.search(vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(dist)
                results.append(result)

        return results
