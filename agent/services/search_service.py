from agent.index.vector_store import VectorStore
from agent.models.text_embedder import TextEmbedder
from agent.utils.logging import setup_logger

logger = setup_logger(__name__)

class SearchService:
    def __init__(self, text_embedder: TextEmbedder = None, vector_store: VectorStore = None):
        self.text_embedder = text_embedder or TextEmbedder()
        self.vector_store = vector_store or VectorStore()

    def search_paper(self, query: str, top_k: int = 5, return_snippets: bool = False, return_files: bool = True):
        logger.info(f"Searching for: {query}")
        query_embedding = self.text_embedder.embed_texts([query])[0]
        
        results = {}
        
        if return_files:
            file_results = self.vector_store.search_paper_files(query_embedding, n_results=top_k)
            results["files"] = file_results

        if return_snippets:
            chunk_results = self.vector_store.search_paper_chunks(query_embedding, n_results=top_k)
            results["snippets"] = chunk_results
            
        return results
