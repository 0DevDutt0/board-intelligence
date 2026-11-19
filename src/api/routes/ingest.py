# src/api/routes/ingest.py
import time
from urllib.parse import unquote
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from src.utils.config import settings
from src.utils.security import hash_pdf_bytes
from src.utils.logger import get_logger
from src.api.session import session_store
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import VectorStore

router = APIRouter()
logger = get_logger(__name__)


def _upload_too_large() -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=f'File exceeds {settings.max_upload_mb} MB limit',
    )


async def _read_upload(request: Request, file: UploadFile | None) -> tuple[str, bytes]:
    if file is not None:
        name = file.filename or ''
        if not name.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail='Only PDF files are accepted')
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise _upload_too_large()
    elif request.headers.get('content-type', '').split(';', 1)[0].strip().lower() == 'application/pdf':
        name = unquote(request.headers.get('x-filename', 'document.pdf'))
        buffer = bytearray()
        async for chunk in request.stream():
            if len(buffer) + len(chunk) > settings.max_upload_bytes:
                raise _upload_too_large()
            buffer.extend(chunk)
        data = bytes(buffer)
    else:
        raise HTTPException(status_code=415, detail='Upload an application/pdf body or a multipart file')

    if b'%PDF-' not in data[:1024]:
        raise HTTPException(status_code=400, detail='The upload is not a PDF')
    return name.replace('\\', '/').rsplit('/', 1)[-1][:255] or 'document.pdf', data


def _build_pipeline(pdf_bytes: bytes, doc_hash: str, app_state: dict) -> tuple[list, list, dict]:
    import pymupdf

    embedder = app_state['embedder']
    try:
        pages = app_state['parser'].parse(pdf_bytes)
    except (pymupdf.FileDataError, ValueError):
        raise HTTPException(status_code=422, detail='The PDF could not be read')

    chunks = app_state['chunker'].chunk_pages(pages, doc_hash, embedder=embedder)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail='No readable text found. Image-only PDFs require the local OCR artifacts.',
        )

    embeddings = embedder.encode_chunks(chunks)
    bm25 = BM25Index()
    bm25.build(chunks)
    vector_store = VectorStore()
    vector_store.build(chunks, embeddings)
    return pages, chunks, {'bm25': bm25, 'vector_store': vector_store, 'chunks': chunks}


@router.post('/ingest')
async def ingest(request: Request, file: UploadFile | None = File(default=None)):
    doc_filename, pdf_bytes = await _read_upload(request, file)
    doc_hash = hash_pdf_bytes(pdf_bytes)
    t_start = time.perf_counter()

    try:
        pages, chunks, pipeline = await run_in_threadpool(
            _build_pipeline, pdf_bytes, doc_hash, request.app.state.app_state,
        )
    except HTTPException:
        raise
    except Exception:
        logger.error('Document processing failed')
        raise HTTPException(
            status_code=500,
            detail={'error': 'Local document processing failed', 'stage': 'ingest', 'recoverable': False},
        )

    ingest_stats = {
        'ingest_time_seconds': round(time.perf_counter() - t_start, 2),
        'page_count': len(pages),
        'chunk_count': len(chunks),
    }
    session_id = await session_store.create_session({
        **ingest_stats,
        'doc_hash': doc_hash,
        'doc_filename': doc_filename,
        'valid_pages': sorted({chunk['page_number'] for chunk in chunks}),
        'ingest_stats': ingest_stats,
        'pipeline': pipeline,
    })
    logger.info('Ingest completed: %s pages, %s chunks', len(pages), len(chunks))
    return JSONResponse({'session_id': session_id, 'doc_filename': doc_filename, **ingest_stats})
