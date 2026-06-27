# src/ingestion/ocr.py
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OCRProcessor:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self._ocr = RapidOCR(
            det_use_cuda=True,
            rec_use_cuda=True,
        )
        logger.info('RapidOCR initialized with CUDA execution providers')

    def process_page(self, image_bytes: bytes) -> str:
        result, elapse = self._ocr(image_bytes)
        if not result:
            return ''
        lines = [item[1] for item in result if item and len(item) > 1]
        return '\n'.join(lines)

    def process_page_from_path(self, image_path: str) -> str:
        with open(image_path, 'rb') as f:
            return self.process_page(f.read())
