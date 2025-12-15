from pathlib import Path
from PIL import Image
import hashlib
from agent.index.vector_store import VectorStore
from agent.models.image_embedder import ImageEmbedder
from agent.utils.logging import setup_logger
from agent.config import Config

logger = setup_logger(__name__)

class ImageSearchService:
    def __init__(self, image_embedder: ImageEmbedder = None, vector_store: VectorStore = None):
        self.image_embedder = image_embedder or ImageEmbedder()
        self.vector_store = vector_store or VectorStore()

    def index_images(self, image_dir: str = str(Config.IMAGES_DIR)):
        root = Path(image_dir)
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        
        image_paths = []
        for p in root.glob("**/*"):
            if p.suffix.lower() in valid_exts:
                image_paths.append(p)
                
        logger.info(f"Found {len(image_paths)} images.")
        
        batch_size = 32
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images = []
            valid_batch_paths = []
            
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    valid_batch_paths.append(p)
                except Exception as e:
                    logger.error(f"Failed to open image {p}: {e}")
            
            if not images:
                continue
                
            embeddings = self.image_embedder.embed_images(images)
            
            ids = [hashlib.sha256(str(p).encode()).hexdigest() for p in valid_batch_paths]
            metadatas = [{"path": str(p), "filename": p.name} for p in valid_batch_paths]
            
            self.vector_store.add_images(ids, embeddings, metadatas)
            logger.info(f"Indexed batch {i // batch_size + 1}")

    def search_image(self, query: str, top_k: int = 5):
        logger.info(f"Searching images for: {query}")
        text_embedding = self.image_embedder.embed_text([query])[0]
        results = self.vector_store.search_images(text_embedding, n_results=top_k)
        return results
