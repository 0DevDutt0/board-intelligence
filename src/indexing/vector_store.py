# src/indexing/vector_store.py
import uuid
import numpy as np
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_COLLECTION = 'chunks'
_VECTOR_DIM = 1024


class VectorStore:
    def __init__(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self._client = QdrantClient(':memory:')
        self._client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=VectorParams(size=_VECTOR_DIM, distance=Distance.COSINE),
        )
        self._chunks = []
        logger.info('Qdrant in-memory collection created')

    def build(self, chunks: list, embeddings: np.ndarray) -> None:
        from qdrant_client.models import PointStruct
        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=i,
                    vector=vec.tolist(),
                    payload=chunk,
                )
            )
        self._client.upsert(collection_name=_COLLECTION, points=points)
        self._chunks = chunks
        logger.info(f'Qdrant indexed {len(points)} vectors')

    def search(self, query_vector: np.ndarray, top_k: int = None) -> list:
        k = top_k or settings.dense_top_k
        # qdrant-client >= 1.9 uses query_points; fall back to legacy search
        try:
            response = self._client.query_points(
                collection_name=_COLLECTION,
                query=query_vector.tolist(),
                limit=k,
                with_payload=True,
            )
            hits = response.points
        except AttributeError:
            hits = self._client.search(
                collection_name=_COLLECTION,
                query_vector=query_vector.tolist(),
                limit=k,
                with_payload=True,
            )
        chunks = []
        for hit in hits:
            chunk = dict(hit.payload)
            chunk['dense_score'] = hit.score
            chunks.append(chunk)
        return chunks
