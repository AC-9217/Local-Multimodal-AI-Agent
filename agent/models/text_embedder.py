from sentence_transformers import SentenceTransformer
from agent.config import Config
from agent.utils.logging import setup_logger

logger = setup_logger(__name__)

class TextEmbedder:
    """
    Wrapper for text embedding model (SentenceTransformer).
    文本嵌入模型（SentenceTransformer）的封装类。
    """
    def __init__(self, model_name: str = Config.TEXT_MODEL_NAME, device: str = Config.DEVICE):
        """
        Initialize the TextEmbedder.
        初始化文本嵌入器。

        Args:
            model_name (str): Name of the model. 模型名称。
            device (str): Device to run the model on. 运行模型的设备。
        """
        logger.info(f"Loading Text Embedder: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        
    def embed_texts(self, texts: list[str]):
        """
        Embed a list of texts.
        对文本列表进行嵌入。

        Args:
            texts (list[str]): List of texts. 文本列表。

        Returns:
            numpy.ndarray: Array of embeddings. 嵌入向量数组。
        """
        # Normalize embeddings for cosine similarity
        # 归一化嵌入向量以用于余弦相似度计算
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
