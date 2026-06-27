# src/indexing/embedder.py
import numpy as np
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    def __init__(self, model_path: str = None, device: str = None, batch_size: int = None):
        from FlagEmbedding import BGEM3FlagModel
        self._model_path = model_path or settings.embedding_model
        self._device = device or settings.device
        self._batch_size = batch_size or settings.embedding_batch_size
        self._model = BGEM3FlagModel(
            self._model_path,
            use_fp16=True,
            device=self._device,
        )
        logger.info(f'BGE-M3 loaded from {self._model_path} on {self._device}')

    def encode_batch(self, texts: list, batch_size: int = None, max_length: int = 8192) -> np.ndarray:
        bs = batch_size or self._batch_size
        output = self._model.encode(
            texts,
            batch_size=bs,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output['dense_vecs']

    def encode_chunks(self, chunks: list) -> np.ndarray:
        prefixed = [
            chunk['embedding_prefix'] + ' ' + chunk['text']
            for chunk in chunks
        ]
        return self.encode_batch(prefixed)

    def encode_query(self, query: str) -> np.ndarray:
        vecs = self.encode_batch([query])
        return vecs[0]
