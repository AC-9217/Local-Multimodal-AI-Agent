import chromadb
from agent.config import Config
from agent.utils.logging import setup_logger

logger = setup_logger(__name__)

class VectorStore:
    def __init__(self, persist_dir: str = str(Config.INDEX_DIR)):
        logger.info(f"Initializing VectorStore at {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        self.papers_files = self.client.get_or_create_collection(name=Config.COLLECTION_PAPERS_FILES)
        self.papers_chunks = self.client.get_or_create_collection(name=Config.COLLECTION_PAPERS_CHUNKS)
        self.images = self.client.get_or_create_collection(name=Config.COLLECTION_IMAGES)

    def add_paper_file(self, doc_id: str, embedding, metadata: dict):
        self.papers_files.upsert(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )

    def add_paper_chunks(self, ids: list[str], embeddings, metadatas: list[dict], documents: list[str]):
        if not ids:
            return
        self.papers_chunks.upsert(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadatas,
            documents=documents
        )

    def add_images(self, ids: list[str], embeddings, metadatas: list[dict]):
        if not ids:
            return
        self.images.upsert(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadatas
        )

    def search_paper_files(self, query_embedding, n_results=5):
        return self.papers_files.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

    def search_paper_chunks(self, query_embedding, n_results=5):
        return self.papers_chunks.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
    def search_images(self, query_embedding, n_results=5):
        return self.images.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
