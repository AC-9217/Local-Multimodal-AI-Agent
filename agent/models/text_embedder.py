from sentence_transformers import SentenceTransformer
from agent.config import Config
from agent.utils.logging import setup_logger

logger = setup_logger(__name__)

class TextEmbedder:
    def __init__(self, model_name: str = Config.TEXT_MODEL_NAME, device: str = Config.DEVICE):
        logger.info(f"Loading Text Embedder: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        
    def embed_texts(self, texts: list[str]):
        # Normalize embeddings for cosine similarity
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
