import torch
from pathlib import Path
import os

class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    PAPERS_DIR = DATA_DIR / "papers"
    IMAGES_DIR = DATA_DIR / "images"
    INDEX_DIR = DATA_DIR / "index"
    CACHE_DIR = DATA_DIR / "cache"

    # Models
    TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    # OpenCLIP model config
    IMAGE_MODEL_NAME = "ViT-B-32"
    IMAGE_MODEL_PRETRAINED = "laion2b_s34b_b79k"

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Indexing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    
    # Chroma
    COLLECTION_PAPERS_FILES = "papers_files"
    COLLECTION_PAPERS_CHUNKS = "papers_chunks"
    COLLECTION_IMAGES = "images"

    @classmethod
    def setup(cls):
        cls.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        cls.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        cls.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

Config.setup()
