import torch
from pathlib import Path
import os

# Set Hugging Face mirror for China access
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class Config:
    # Paths
    # 项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    # 数据存储目录
    DATA_DIR = PROJECT_ROOT / "data"
    # 论文存储目录
    PAPERS_DIR = DATA_DIR / "papers"
    # 图片存储目录
    IMAGES_DIR = DATA_DIR / "images"
    # 索引存储目录
    INDEX_DIR = DATA_DIR / "index"
    # 缓存目录
    CACHE_DIR = DATA_DIR / "cache"

    # Models
    # 文本嵌入模型名称
    # 升级为支持多语言且性能更强的模型 (768维)
    TEXT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # OpenCLIP model config
    # 图像嵌入模型配置 (CLIP)
    # 升级为 Huge 多语言模型 (XLM-Roberta) 以支持中文并提供更高精度 (1024维)
    # 注意：首次运行会自动下载模型，且索引与旧版(768维)不兼容，建议重建索引
    IMAGE_MODEL_NAME = "xlm-roberta-large-ViT-H-14"
    IMAGE_MODEL_PRETRAINED = "frozen_laion5b_s13b_b90k"

    # Device
    # 设备选择: 如果有 CUDA 则使用 CUDA，否则使用 CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Indexing
    # 文本分块大小
    # 减小分块大小以提高检索粒度
    CHUNK_SIZE = 800
    # 文本分块重叠大小
    CHUNK_OVERLAP = 150
    
    # Chroma
    # 向量数据库集合名称
    # 论文文件集合
    COLLECTION_PAPERS_FILES = "papers_files"
    # 论文片段集合
    COLLECTION_PAPERS_CHUNKS = "papers_chunks"
    # 图片集合
    # 更新集合名称以避免与旧维度(768)冲突
    COLLECTION_IMAGES = "images_xlm_h14"

    @classmethod
    def setup(cls):
        """
        Setup necessary directories.
        初始化必要的目录结构。
        """
        cls.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        cls.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        cls.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

Config.setup()
