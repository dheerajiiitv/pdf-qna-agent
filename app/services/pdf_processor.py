from typing import List, Tuple
import pdfplumber
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class PDFProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.chunk_size = 1000
        self.chunk_overlap = 100

    def extract_text(self) -> str:
        logger.info(f"Extracting text from {self.file_path}")
        with pdfplumber.open(self.file_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages])
        return text

    def split_text(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks
