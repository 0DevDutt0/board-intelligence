import json
import re

import httpx
import numpy as np
import pytest
import pytest_asyncio

from examples.generate_board_packs import build_pdf, load_board_packs
from src.api.main import app
from src.api.session import session_store
from src.generation.citation_injector import ABSTENTION
from src.generation.hallucination_guard import HallucinationGuard
from src.ingestion.chunker import SemanticChunker
from src.ingestion.parser import DocumentParser
from src.retrieval.hybrid_retriever import HybridRetriever


class FixedEmbedder:
    def encode_batch(self, texts, **kwargs):
        return np.tile(self.encode_query(''), (len(texts), 1))

    def encode_query(self, question):
        vector = np.zeros(1024)
        vector[0] = 1
        return vector

    def encode_chunks(self, chunks):
        return self.encode_batch(chunks)


class IdentityReranker:
    def rerank(self, question, chunks, top_k):
        return chunks[:top_k]


class ScriptedLLM:
    def __init__(self):
        self.answer = '- The Board approved phase one of dock electrification. [Page 2]'

    async def stream_completion(self, messages, **kwargs):
        yield self.answer


@pytest_asyncio.fixture
async def api(monkeypatch):
    model = ScriptedLLM()
    guard = object.__new__(HallucinationGuard)
    guard._threshold = 0.7
    guard._nli_score = lambda claim, premise: float(
        'Board approved phase one' in claim and 'Board approved phase one' in premise
    )
    state = {
        'parser': DocumentParser('unused'),
        'chunker': SemanticChunker(),
        'embedder': FixedEmbedder(),
        'reranker': IdentityReranker(),
        'llm_client': model,
        'retriever': HybridRetriever(model),
        'guard': guard,
    }
    monkeypatch.setattr(app.state, 'app_state', state, raising=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
        yield client, model, guard
    await session_store.shutdown()


@pytest.fixture(scope='module')
def board_pdf():
    return build_pdf(load_board_packs()['harbor-works'])


async def ingest(client, board_pdf):
    response = await client.post('/ingest', content=board_pdf, headers={
        'Content-Type': 'application/pdf', 'X-Filename': 'Board%20Minutes.pdf',
    })
    assert response.status_code == 200, response.text
    return response.json()


async def ask(client, session_id):
    response = await client.post('/query', json={
        'question': 'What did the Board approve?', 'use_hyde': False,
    }, headers={'X-Session-ID': session_id})
    assert response.status_code == 200
    return {
        event: json.loads(data)
        for event, data in re.findall(r'^event: (.+)\ndata: (.+)$', response.text, re.MULTILINE)
    }


@pytest.mark.asyncio
async def test_upload_stream_and_end_session(api, board_pdf):
    client, _, _ = api
    data = await ingest(client, board_pdf)
    assert data['page_count'] == 3
    assert data['chunk_count'] >= 3
    assert data['doc_filename'] == 'Board Minutes.pdf'

    session_id = data['session_id']
    events = await ask(client, session_id)
    assert '[Page 2]' in events['corrected']['text']
    assert 2 in {item['page_number'] for item in events['citations']['chunks']}
    assert events['verification']['status'] == 'supported'
    assert 'latency_ms' in events['done']
    assert (await client.delete(f'/session/{session_id}')).json() == {'deleted': True}
    response = await client.post('/query', json={'question': 'Anything?'},
                                 headers={'X-Session-ID': session_id})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_bad_citation_and_abstention_are_distinct(api, board_pdf):
    client, model, _ = api
    session = (await ingest(client, board_pdf))['session_id']
    model.answer = '- The Board approved phase one. [Page 90]'
    events = await ask(client, session)
    assert '[unverifiable citation]' in events['corrected']['text']
    assert events['verification']['status'] == 'needs_review'
    assert events['verification']['warnings'][0]['kind'] == 'invalid_citation'

    model.answer = ABSTENTION
    events = await ask(client, session)
    assert events['verification']['status'] == 'abstained'


@pytest.mark.asyncio
async def test_failed_verification_is_explicit(api, board_pdf, monkeypatch):
    client, _, guard = api
    session = (await ingest(client, board_pdf))['session_id']

    def fail(*args):
        raise RuntimeError('model unavailable')

    monkeypatch.setattr(guard, '_nli_score', fail)
    events = await ask(client, session)
    assert events['verification']['status'] == 'unavailable'
    assert events['corrected']['text']


@pytest.mark.asyncio
async def test_multipart_upload_remains_supported(api, board_pdf):
    client, _, _ = api
    response = await client.post('/ingest', files={'file': ('example.pdf', board_pdf, 'application/pdf')})
    assert response.status_code == 200
    assert response.json()['page_count'] == 3


@pytest.mark.asyncio
async def test_raw_upload_is_bounded(api, monkeypatch):
    from src.utils.config import settings
    client, _, _ = api
    monkeypatch.setattr(settings, 'max_upload_mb', 1)
    response = await client.post('/ingest', content=b'%PDF-' + b'x' * (1024 * 1024),
                                 headers={'Content-Type': 'application/pdf'})
    assert response.status_code == 413
