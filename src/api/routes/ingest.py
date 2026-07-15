# src/api/routes/ingest.py
import os
import time
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from src.utils.config import settings
from src.utils.security import hash_pdf_bytes
from src.utils.logger import get_logger
from src.api.session import session_store
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import VectorStore

router = APIRouter()
logger = get_logger(__name__)


@router.post('/ingest')
async def ingest(request: Request, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF files are accepted')

    pdf_bytes = await file.read()
    if len(pdf_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f'File exceeds {settings.max_upload_mb} MB limit',
        )

    doc_hash = hash_pdf_bytes(pdf_bytes)
    t_start = time.perf_counter()

    # Create temp session directory
    import uuid
    session_id_tmp = str(uuid.uuid4())
    session_dir = settings.temp_session_path / session_id_tmp
    session_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = session_dir / 'document.pdf'
    with open(pdf_path, 'wb') as f:
        f.write(pdf_bytes)
    os.chmod(pdf_path, 0o600)

    try:
        app_state = request.app.state.app_state
        embedder = app_state['embedder']
        parser = app_state['parser']
        chunker = app_state['chunker']

        logger.info(f'Parsing {file.filename} ({len(pdf_bytes)} bytes)')
        pages = parser.parse(str(pdf_path))
        page_count = len(pages)

        logger.info(f'Chunking {page_count} pages')
        chunks = chunker.chunk_pages(pages, doc_hash, embedder=embedder)
        chunk_count = len(chunks)

        logger.info(f'Embedding {chunk_count} chunks')
        embeddings = embedder.encode_chunks(chunks)

        logger.info('Building BM25 index')
        bm25 = BM25Index()
        bm25.build(chunks)

        logger.info('Building Qdrant vector store')
        vector_store = VectorStore()
        vector_store.build(chunks, embeddings)

        valid_pages = list({c['page_number'] for c in chunks})
        t_elapsed = time.perf_counter() - t_start

        ingest_stats = {
            'ingest_time_seconds': round(t_elapsed, 2),
            'page_count': page_count,
            'chunk_count': chunk_count,
        }

        session_id = await session_store.create_session({
            'doc_hash': doc_hash,
            'doc_filename': file.filename,
            'page_count': page_count,
            'chunk_count': chunk_count,
            'valid_pages': valid_pages,
            'ingest_stats': ingest_stats,
            'pipeline': {
                'bm25': bm25,
                'vector_store': vector_store,
                'chunks': chunks,
                'valid_pages_set': set(valid_pages),
            },
        })

        # Rename session dir to match session_id
        final_dir = settings.temp_session_path / session_id
        session_dir.rename(final_dir)

        logger.info(
            f'Ingest complete: session={session_id}, pages={page_count}, '
            f'chunks={chunk_count}, time={t_elapsed:.2f}s'
        )

        return JSONResponse({
            'session_id': session_id,
            'page_count': page_count,
            'chunk_count': chunk_count,
            'ingest_time_seconds': round(t_elapsed, 2),
        })

    except Exception as exc:
        import shutil
        shutil.rmtree(session_dir, ignore_errors=True)
        logger.error(f'Ingest failed: {exc}', exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={'error': str(exc), 'stage': 'ingest', 'recoverable': False},
        )
