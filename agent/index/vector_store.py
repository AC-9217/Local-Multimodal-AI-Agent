import chromadb
from agent.config import Config
from agent.utils.logging import setup_logger

logger = setup_logger(__name__)

class VectorStore:
    """
    Wrapper for ChromaDB vector store.
    ChromaDB 向量数据库的封装类。
    """
    def __init__(self, persist_dir: str = str(Config.INDEX_DIR)):
        """
        Initialize the VectorStore.
        初始化向量数据库。

        Args:
            persist_dir (str): Directory to persist data. 数据持久化目录。
        """
        logger.info(f"Initializing VectorStore at {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        self.papers_files = self.client.get_or_create_collection(name=Config.COLLECTION_PAPERS_FILES)
        self.papers_chunks = self.client.get_or_create_collection(name=Config.COLLECTION_PAPERS_CHUNKS)
        self.images = self.client.get_or_create_collection(name=Config.COLLECTION_IMAGES)

    def add_paper_file(self, doc_id: str, embedding, metadata: dict):
        """
        Add a paper file entry.
        添加论文文件条目。
        """
        self.papers_files.upsert(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )

    def add_paper_chunks(self, ids: list[str], embeddings, metadatas: list[dict], documents: list[str]):
        """
        Add paper chunks.
        添加论文片段。
        """
        if not ids:
            return
        self.papers_chunks.upsert(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadatas,
            documents=documents
        )

    def add_images(self, ids: list[str], embeddings, metadatas: list[dict]):
        """
        Add image embeddings.
        添加图像嵌入。
        """
        if not ids:
            return
        self.images.upsert(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadatas
        )

    def search_paper_files(self, query_embedding, n_results=5):
        """
        Search for paper files.
        搜索论文文件。
        """
        return self.papers_files.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

    def search_paper_chunks(self, query_embedding, n_results=5):
        """
        Search for paper chunks.
        搜索论文片段。
        """
        return self.papers_chunks.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
    def search_images(self, query_embedding, n_results=5):
        """
        Search for images.
        搜索图片。
        """
        return self.images.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
