# src/api/routes/query.py
import asyncio
import json
import re
import time
from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.utils.config import settings
from src.utils.logger import get_logger
from src.api.session import session_store
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.rrf_fusion import reciprocal_rank_fusion
from src.generation.prompt_builder import build_prompt
from src.generation.citation_injector import validate_citations

router = APIRouter()
logger = get_logger(__name__)

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    return _THINK_RE.sub('', text).strip()

_retriever = HybridRetriever()


class QueryRequest(BaseModel):
    question: str
    use_hyde: bool = True


def _sse_event(event: str, data: dict) -> str:
    return f'event: {event}\ndata: {json.dumps(data)}\n\n'


async def _stream_query(
    question: str,
    use_hyde: bool,
    session_id: str,
    app_state: dict,
):
    t_start = time.perf_counter()
    state = await session_store.get_session(session_id)
    if state is None:
        yield _sse_event('error', {'error': 'Session not found'})
        return

    pipeline = await session_store.get_pipeline(session_id)
    bm25 = pipeline.get('bm25')
    vector_store = pipeline.get('vector_store')
    valid_pages_set = pipeline.get('valid_pages_set', set())
    all_chunks = pipeline.get('chunks', [])

    if bm25 is None or vector_store is None:
        yield _sse_event('error', {'error': 'Pipeline not ready for this session'})
        return

    embedder = app_state['embedder']
    reranker = app_state['reranker']
    llm_client = app_state['llm_client']
    guard = app_state['guard']

    conv_history = state.get('conversation_history', [])

    # For very short / overview queries, skip HyDE -- it produces poor hypotheticals
    # for vague questions and the <think> block polluted the embedding before the fix.
    word_count = len(question.split())
    effective_hyde = use_hyde and word_count >= settings.hyde_min_query_words

    try:
        dense, sparse = await _retriever.retrieve(
            question, embedder, bm25, vector_store, effective_hyde,
            conversation_history=conv_history,
        )
    except Exception as exc:
        logger.error(f'Retrieval failed: {exc}')
        dense, sparse = [], []

    fused = reciprocal_rank_fusion([dense, sparse], k=settings.rrf_k)

    try:
        loop = asyncio.get_event_loop()
        top_chunks = await loop.run_in_executor(
            None, reranker.rerank, question, fused, settings.reranker_top_k
        )
    except Exception as exc:
        logger.error(f'Reranking failed: {exc}')
        top_chunks = fused[:settings.reranker_top_k]

    # Drop chunks whose reranker score is below the minimum threshold.
    # Low-scoring chunks are tangentially related noise that wastes context
    # tokens without improving answer quality.
    if settings.reranker_min_score > 0 and top_chunks:
        filtered = [
            c for c in top_chunks
            if c.get('reranker_score', 1.0) >= settings.reranker_min_score
        ]
        # Always keep at least 3 chunks so short documents still get context.
        if len(filtered) >= 3:
            dropped = len(top_chunks) - len(filtered)
            if dropped:
                logger.info(
                    f'Reranker score filter: dropped {dropped} chunk(s) '
                    f'below {settings.reranker_min_score} threshold'
                )
            top_chunks = filtered

    # Fallback: if retrieval returned nothing, use all indexed chunks so the
    # model always has context (works well for single-page or small documents).
    if not top_chunks and all_chunks:
        logger.warning('Retrieval returned 0 chunks, using all session chunks as fallback')
        top_chunks = all_chunks[:settings.reranker_top_k * 2]

    retrieved_pages = sorted({
        c['page_number'] for c in top_chunks
        if c.get('page_number') is not None
    })
    messages = build_prompt(question, top_chunks, conv_history, valid_pages=retrieved_pages)

    # Estimate input token count (approx 4 chars per token) and cap output tokens
    # so that input + output never exceeds the model's 8192-token context window.
    MODEL_CONTEXT_LIMIT = 8192
    OUTPUT_HEADROOM = 200  # safety buffer for tokenizer rounding
    prompt_chars = sum(
        len(m.get('content', '')) for m in messages if isinstance(m.get('content'), str)
    )
    estimated_input_tokens = prompt_chars // 4
    available_output = MODEL_CONTEXT_LIMIT - estimated_input_tokens - OUTPUT_HEADROOM
    safe_max_tokens = max(256, min(settings.vllm_max_tokens, available_output))
    logger.info(
        f'Context: ~{estimated_input_tokens} input tokens, '
        f'capped output to {safe_max_tokens} tokens'
    )

    full_response = []
    token_count = 0

    try:
        async for token in llm_client.stream_completion(messages, max_tokens=safe_max_tokens):
            full_response.append(token)
            token_count += 1
            yield _sse_event('chunk', {'text': token})
    except Exception as exc:
        logger.error(f'LLM streaming failed: {exc}')
        yield _sse_event('error', {
            'error': f'Generation failed: {exc}',
            'stage': 'generation',
            'recoverable': True,
        })
        return

    response_text = ''.join(full_response)
    answer_text = _strip_think_blocks(response_text)
    validated_text = validate_citations(answer_text, set(retrieved_pages))

    citation_chunks = [
        {
            'page_number': c.get('page_number'),
            'chunk_type': c.get('chunk_type'),
            'section_heading': c.get('section_heading'),
            'text': c.get('text', '')[:300],
            'reranker_score': c.get('reranker_score', 0.0),
        }
        for c in top_chunks
    ]
    yield _sse_event('citations', {'chunks': citation_chunks})

    # Send the validated text so the frontend replaces streamed tokens with
    # the citation-checked version. Any [Page N] not in retrieved evidence
    # is already replaced with [unverifiable citation] at this point.
    yield _sse_event('corrected', {'text': validated_text})

    guard_task = asyncio.create_task(guard.verify(validated_text, top_chunks, question=question))

    t_elapsed = time.perf_counter() - t_start
    yield _sse_event('done', {
        'total_tokens': token_count,
        'latency_ms': int(t_elapsed * 1000),
    })

    try:
        warnings = await asyncio.wait_for(guard_task, timeout=30.0)
        # Always send verification event so the client can show Verified or warnings
        yield _sse_event('verification', {'warnings': warnings or []})
    except asyncio.TimeoutError:
        logger.warning('Hallucination guard timed out')
    except Exception as exc:
        logger.error(f'Hallucination guard error: {exc}')

    await session_store.append_conversation(session_id, question, validated_text)


@router.post('/query')
async def query(
    body: QueryRequest,
    request: Request,
    x_session_id: str = Header(..., alias='X-Session-ID'),
):
    state = await session_store.get_session(x_session_id)
    if state is None:
        raise HTTPException(status_code=404, detail='Session not found or expired')

    app_state = request.app.state.app_state

    async def event_generator():
        async for event in _stream_query(
            body.question, body.use_hyde, x_session_id, app_state
        ):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )
