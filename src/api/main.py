# src/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.utils.config import settings
from src.utils.logger import get_logger
from src.api.session import session_store
from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router
from src.api.routes.health import router as health_router

logger = get_logger(__name__)

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Board Intelligence System starting up')

    logger.info('Loading BGE-M3 embedder')
    from src.indexing.embedder import Embedder
    embedder = Embedder()

    logger.info('Loading BGE-Reranker')
    from src.retrieval.reranker import Reranker
    reranker = Reranker()

    logger.info('Loading LLM client')
    from src.generation.llm_client import LLMClient
    llm_client = LLMClient()

    logger.info('Loading hallucination guard')
    from src.generation.hallucination_guard import HallucinationGuard
    guard = HallucinationGuard()

    logger.info('Initializing document parser and chunker')
    from src.ingestion.parser import DocumentParser
    from src.ingestion.chunker import SemanticChunker
    from src.utils.config import settings as _settings
    parser = DocumentParser(_settings.docling_artifacts)
    chunker = SemanticChunker()

    app_state.update({
        'embedder': embedder,
        'reranker': reranker,
        'llm_client': llm_client,
        'guard': guard,
        'parser': parser,
        'chunker': chunker,
        'models_loaded': True,
    })
    app.state.app_state = app_state

    session_store.start_cleanup()
    logger.info('Startup complete')

    yield

    logger.info('Shutting down Board Intelligence System')
    if session_store._cleanup_task:
        session_store._cleanup_task.cancel()


app = FastAPI(
    title='Board Intelligence System',
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'DELETE'],
    allow_headers=['Content-Type', 'X-Session-ID'],
)

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(health_router)


@app.delete('/session/{session_id}')
async def delete_session(session_id: str):
    deleted = await session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail='Session not found')
    return JSONResponse({'deleted': True})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f'Unhandled exception: {exc}', exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            'error': 'Internal server error',
            'stage': 'unknown',
            'recoverable': False,
        },
    )
