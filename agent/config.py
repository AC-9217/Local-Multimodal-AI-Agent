import torch
from pathlib import Path
import os

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
    TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    # OpenCLIP model config
    # 图像嵌入模型配置 (CLIP)
    IMAGE_MODEL_NAME = "ViT-B-32"
    IMAGE_MODEL_PRETRAINED = "laion2b_s34b_b79k"

    # Device
    # 设备选择: 如果有 CUDA 则使用 CUDA，否则使用 CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Indexing
    # 文本分块大小
    CHUNK_SIZE = 1000
    # 文本分块重叠大小
    CHUNK_OVERLAP = 100
    
    # Chroma
    # 向量数据库集合名称
    # 论文文件集合
    COLLECTION_PAPERS_FILES = "papers_files"
    # 论文片段集合
    COLLECTION_PAPERS_CHUNKS = "papers_chunks"
    # 图片集合
    COLLECTION_IMAGES = "images"

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
