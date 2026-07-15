# src/retrieval/reranker.py
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    def __init__(self, model_path: str = None):
        from sentence_transformers import CrossEncoder
        path = model_path or settings.reranker_model
        self._model = CrossEncoder(path, max_length=512)
        logger.info(f'BGE-Reranker loaded from {path}')

    def rerank(self, query: str, chunks: list, top_k: int = None) -> list:
        k = top_k or settings.reranker_top_k
        if not chunks:
            return []
        pairs = [(query, chunk['text']) for chunk in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )
        result = []
        for score, chunk in ranked[:k]:
            c = dict(chunk)
            c['reranker_score'] = float(score)
            result.append(c)
        return result
