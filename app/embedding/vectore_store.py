import faiss
import numpy as np
import logging
import json

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)  # L2 = Euclidean distance
        self.metadata = []
        logger.info(f"📦 Initialized FAISS index with dimension: {dim}")

    def add(self, vector: np.ndarray, meta: dict):
        if vector is None:
            logger.warning("⚠️ Skipped adding null vector.")
            return

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype="float32")

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        self.index.add(vector)
        self.metadata.append(meta)
        logger.debug(f"✅ Added vector with metadata: {meta}")

    def save(self, index_path: str):
        try:
            faiss.write_index(self.index, index_path)
            logger.info(f"💾 Saved FAISS index to: {index_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save index: {e}")

    def save_metadata(self, meta_path: str):
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"📝 Saved metadata to: {meta_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {e}")
