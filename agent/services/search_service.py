from agent.index.vector_store import VectorStore
from agent.models.text_embedder import TextEmbedder
from agent.utils.logging import setup_logger

logger = setup_logger(__name__)

class SearchService:
    """
    Service for searching papers.
    文献检索服务类。
    """
    def __init__(self, text_embedder: TextEmbedder = None, vector_store: VectorStore = None):
        """
        Initialize the search service.
        初始化搜索服务。

        Args:
            text_embedder: Text embedding model instance. 文本嵌入模型实例。
            vector_store: Vector store instance. 向量数据库实例。
        """
        self.text_embedder = text_embedder or TextEmbedder()
        self.vector_store = vector_store or VectorStore()

    def search_paper(self, query: str, top_k: int = 5, return_snippets: bool = False, return_files: bool = True):
        """
        Search for papers based on a query.
        根据查询内容搜索论文。

        Args:
            query (str): The search query. 查询字符串。
            top_k (int): Number of top results to return. 返回的前 K 个结果。
            return_snippets (bool): Whether to return chunk snippets. 是否返回文本片段。
            return_files (bool): Whether to return file-level results. 是否返回文件级结果。
        
        Returns:
            dict: Search results containing files and/or snippets. 包含文件和/或片段的搜索结果字典。
        """
        logger.info(f"Searching for: {query}")
        # Generate embedding for the query
        # 为查询生成嵌入向量
        query_embedding = self.text_embedder.embed_texts([query])[0]
        
        results = {}
        
        if return_files:
            # Search in the papers_files collection
            # 在论文文件集合中搜索
            file_results = self.vector_store.search_paper_files(query_embedding, n_results=top_k)
            results["files"] = file_results

        if return_snippets:
            # Search in the papers_chunks collection
            # 在论文片段集合中搜索
            chunk_results = self.vector_store.search_paper_chunks(query_embedding, n_results=top_k)
            results["snippets"] = chunk_results
            
        return results
