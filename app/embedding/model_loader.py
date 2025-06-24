import logging
from transformers import AutoTokenizer, AutoModel
import torch

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class CodeEmbeddingModel:
    def __init__(self, model_name="BAAI/bge-code"):
        try:
            logger.info(f"🔁 Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            logger.info("✅ Model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise

    def encode(self, text: str):
        if not text.strip():
            logger.warning("⚠️ Empty input text received. Skipping embedding.")
            return None

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embedding
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            return None

# Global instance for reuse
embedding_model = CodeEmbeddingModel()
