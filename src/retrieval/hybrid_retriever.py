# src/retrieval/hybrid_retriever.py
import asyncio
import re
from src.utils.config import settings
from src.utils.logger import get_logger

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

logger = get_logger(__name__)

# Domain-specific HyDE prompt.  The generic "helpful assistant" prompt produced
# hypotheticals whose vocabulary diverged from board governance document style,
# weakening dense retrieval.  This prompt anchors the hypothetical in the
# formal language that actually appears in governance reports.
_HYDE_SYSTEM = (
    'You are reading a corporate board of directors governance report. '
    'Write a short factual passage (3-5 sentences) exactly as it would appear '
    'in such a document to directly answer the question. '
    'Use formal governance vocabulary: committees advise, review, discuss, '
    'monitor, assess; boards approve, oversee, determine, adopt. '
    'Be specific and concise. Write only the passage, no preamble or commentary.'
)


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
            {'role': 'system', 'content': _HYDE_SYSTEM},
            {'role': 'user', 'content': query},
        ]
        tokens = []
        async for token in client.stream_completion(hyde_prompt, max_tokens=150):
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
        # The word-count gate (hyde_min_query_words) is enforced in query.py
        # before this call.  Do not re-apply it here -- the two checks were
        # mutually exclusive and caused HyDE to never fire.
        do_hyde = use_hyde if use_hyde is not None else settings.use_hyde

        embed_query = query
        if do_hyde:
            try:
                hyde_answer = await self._generate_hyde_answer(query)
                if hyde_answer.strip():
                    if settings.hyde_concat_query:
                        # Concatenate original query with the hypothetical so
                        # the embedding stays anchored to the question's keywords
                        # even when the hypothetical drifts.
                        embed_query = f'{query}\n\n{hyde_answer}'
                    else:
                        embed_query = hyde_answer
                    logger.info(
                        f'HyDE expanded query; hypothetical preview: '
                        f'{hyde_answer[:120].replace(chr(10), " ")!r}'
                    )
            except Exception as exc:
                logger.warning(f'HyDE generation failed: {exc}. Using original query.')

        loop = asyncio.get_event_loop()

        query_vec = await loop.run_in_executor(None, embedder.encode_query, embed_query)

        # BM25 always uses the original query -- keyword matching should not
        # be polluted with LLM-generated vocabulary.
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
