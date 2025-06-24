import os
import zipfile
import logging
from app.embedding.pipeline import embed_codebase
from app.embedding.search import CodeSearchEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
UPLOAD_DIR = "uploads"
VECTOR_PATH = "vector.index"
META_PATH = VECTOR_PATH + ".meta.json"

def extract_zip(zip_path):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_DIR)
        logger.info(f"✅ Extracted ZIP to '{UPLOAD_DIR}'")
    except Exception as e:
        logger.error(f"❌ Failed to extract zip: {e}")
        exit(1)

def run_embedding():
    logger.info("⚙️ Starting embedding pipeline...")
    embed_codebase(UPLOAD_DIR, VECTOR_PATH)

def start_search():
    engine = CodeSearchEngine(VECTOR_PATH, META_PATH)
    while True:
        query = input("\n🔍 Enter search query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        results = engine.search(query, top_k=5)
        if not results:
            print("❌ No results found.")
            continue
        for i, res in enumerate(results, 1):
            print(f"\nResult {i}:")
            print("📄 File:", res["file"])
            print("📏 Score:", res["score"])

def main():
    zip_input = input("📦 Enter path to zipped codebase: ").strip()
    extract_zip(zip_input)
    run_embedding()
    start_search()

if __name__ == "__main__":
    main()
