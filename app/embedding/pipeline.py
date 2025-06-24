import os
import logging
from app.embedding.model_loader import embedding_model
from app.embedding.vector_store import VectorStore

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def embed_codebase(codebase_path: str, output_index_path: str):
    logger.info(f"📂 Starting embedding for codebase at: {codebase_path}")
    store = VectorStore(dim=768)  # BGE-Code outputs 768-dim vectors
    file_count = 0

    for root, _, files in os.walk(codebase_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_code = f.read()
                        if not raw_code.strip():
                            logger.warning(f"⚠️ Skipping empty file: {file_path}")
                            continue

                        vector = embedding_model.encode(raw_code)
                        store.add(vector, {"file": file_path})
                        logger.info(f"✅ Embedded: {file_path}")
                        file_count += 1

                except Exception as e:
                    logger.error(f"❌ Failed to process {file_path}: {e}")

    # Save FAISS index and metadata
    store.save(output_index_path)
    store.save_metadata(output_index_path + ".meta.json")
    logger.info(f"📦 Finished embedding {file_count} file(s). Index saved to: {output_index_path}")
