# src/indexing/bm25_index.py
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list:
    return text.lower().split()


class BM25Index:
    def __init__(self):
        self._index = None
        self._chunks = []

    def build(self, chunks: list) -> None:
        from rank_bm25 import BM25Okapi
        self._chunks = chunks
        tokenized = [_tokenize(chunk['text']) for chunk in chunks]
        self._index = BM25Okapi(tokenized, k1=settings.bm25_k1, b=settings.bm25_b)
        logger.info(
            f'BM25 index built: {len(chunks)} docs, '
            f'k1={settings.bm25_k1}, b={settings.bm25_b}'
        )

    def search(self, query: str, top_k: int = None) -> list:
        if self._index is None:
            return []
        k = top_k or settings.sparse_top_k
        tokens = _tokenize(query)
        scores = self._index.get_scores(tokens)
        import numpy as np
        ranked_indices = np.argsort(scores)[::-1][:k]
        results = []
        for rank, idx in enumerate(ranked_indices):
            if scores[idx] > 0:
                chunk = dict(self._chunks[idx])
                chunk['bm25_score'] = float(scores[idx])
                results.append(chunk)
        return results
