import fitz  # PyMuPDF
from agent.config import Config
from agent.utils.logging import setup_logger

logger = setup_logger(__name__)

class PDFParser:
    @staticmethod
    def parse(file_path: str, chunk_size: int = Config.CHUNK_SIZE, overlap: int = Config.CHUNK_OVERLAP):
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {file_path}: {e}")
            return "", []

        all_chunks = []
        full_text = []

        for page_num, page in enumerate(doc):
            text = page.get_text()
            full_text.append(text)
            
            if not text.strip():
                continue

            # Sliding window on this page
            start = 0
            text_len = len(text)
            
            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunk_text = text[start:end]
                
                # Only add substantial chunks
                if len(chunk_text.strip()) > 50:
                    all_chunks.append({
                        "text": chunk_text,
                        "page_id": page_num + 1,
                        "char_start": start,
                        "char_end": end
                    })
                
                if end == text_len:
                    break
                
                start += chunk_size - overlap

        doc.close()
        return "\n".join(full_text), all_chunks
