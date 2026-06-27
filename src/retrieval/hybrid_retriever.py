# src/retrieval/hybrid_retriever.py
import asyncio
import re
from src.utils.config import settings
from src.utils.logger import get_logger

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(self):
        self._llm_client = None

    def _get_llm_client(self):
        if self._llm_client is None:
            from src.generation.llm_client import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client

    async def _generate_hyde_answer(self, query: str) -> str:
        client = self._get_llm_client()
        hyde_prompt = [
            {
                'role': 'system',
                'content': (
                    'You are a helpful assistant. Write a short, factual passage '
                    'that would directly answer the following question. '
                    'Do not hedge or qualify. Write as if the answer appears in a document.'
                ),
            },
            {'role': 'user', 'content': query},
        ]
        tokens = []
        async for token in client.stream_completion(hyde_prompt, max_tokens=200):
            tokens.append(token)
        raw = ''.join(tokens)
        return _THINK_RE.sub('', raw).strip()

    async def retrieve(
        self,
        query: str,
        embedder,
        bm25,
        vector_store,
        use_hyde: bool = None,
    ) -> tuple:
        do_hyde = use_hyde if use_hyde is not None else settings.use_hyde
        word_count = len(query.split())

        embed_query = query
        if do_hyde and word_count < settings.hyde_min_query_words:
            try:
                hyde_answer = await self._generate_hyde_answer(query)
                if hyde_answer.strip():
                    embed_query = hyde_answer
                    logger.info(f'HyDE expanded query ({word_count} words)')
            except Exception as exc:
                logger.warning(f'HyDE generation failed: {exc}. Using original query.')

        loop = asyncio.get_event_loop()

        query_vec = await loop.run_in_executor(None, embedder.encode_query, embed_query)

        dense_future = loop.run_in_executor(
            None, vector_store.search, query_vec, settings.dense_top_k
        )
        sparse_future = loop.run_in_executor(
            None, bm25.search, query, settings.sparse_top_k
        )

        dense_results, sparse_results = await asyncio.gather(dense_future, sparse_future)
        logger.info(
            f'Retrieved dense={len(dense_results)}, sparse={len(sparse_results)}'
        )
        return dense_results, sparse_results
