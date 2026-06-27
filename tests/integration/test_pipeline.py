# tests/integration/test_pipeline.py
import asyncio
import os
import time
import pytest
import httpx
from pathlib import Path

API_BASE = os.environ.get('BIS_API_URL', 'http://127.0.0.1:8000')
TEST_PDF = Path(__file__).parent.parent / 'fixtures' / 'sample_board_doc.pdf'

pytestmark = pytest.mark.skipif(
    not TEST_PDF.exists(),
    reason='sample_board_doc.pdf fixture not found',
)


@pytest.fixture(scope='module')
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope='module')
async def session_id():
    async with httpx.AsyncClient(base_url=API_BASE, timeout=180.0) as client:
        t_start = time.perf_counter()
        with open(TEST_PDF, 'rb') as f:
            resp = await client.post(
                '/ingest',
                files={'file': ('sample_board_doc.pdf', f, 'application/pdf')},
            )
        elapsed = time.perf_counter() - t_start
        assert resp.status_code == 200, f'Ingest failed: {resp.text}'
        data = resp.json()
        assert 'session_id' in data
        assert elapsed < 120, f'Ingest took {elapsed:.1f}s, expected < 120s'
        return data['session_id']


@pytest.mark.asyncio
async def test_ingest_returns_expected_fields(session_id):
    async with httpx.AsyncClient(base_url=API_BASE, timeout=30.0) as client:
        resp = await client.post(
            '/ingest',
            files={'file': ('sample_board_doc.pdf', open(TEST_PDF, 'rb'), 'application/pdf')},
        )
    data = resp.json()
    assert 'session_id' in data
    assert 'page_count' in data
    assert 'chunk_count' in data
    assert 'ingest_time_seconds' in data
    assert data['page_count'] > 0
    assert data['chunk_count'] > 0


@pytest.mark.asyncio
async def test_query_returns_cited_answer(session_id):
    async with httpx.AsyncClient(base_url=API_BASE, timeout=60.0) as client:
        t_start = time.perf_counter()
        events = []
        async with client.stream(
            'POST',
            '/query',
            json={'question': 'What were the main agenda items discussed?', 'use_hyde': False},
            headers={'X-Session-ID': session_id},
        ) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith('data:'):
                    events.append(line[5:].strip())
        elapsed = time.perf_counter() - t_start

    import json
    text_parts = []
    got_citations = False
    got_done = False
    for e in events:
        if not e:
            continue
        try:
            obj = json.loads(e)
        except Exception:
            continue
        if 'text' in obj:
            text_parts.append(obj['text'])
        if 'chunks' in obj:
            got_citations = True
        if 'total_tokens' in obj:
            got_done = True

    full_text = ''.join(text_parts)
    assert len(full_text) > 20, 'Response too short'
    assert got_citations, 'No citations event received'
    assert got_done, 'No done event received'
    assert elapsed < 12, f'Query took {elapsed:.1f}s, expected < 12s'


@pytest.mark.asyncio
async def test_unanswerable_triggers_abstention(session_id):
    async with httpx.AsyncClient(base_url=API_BASE, timeout=60.0) as client:
        events = []
        async with client.stream(
            'POST',
            '/query',
            json={
                'question': 'What is the weather like on Mars today?',
                'use_hyde': False,
            },
            headers={'X-Session-ID': session_id},
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith('data:'):
                    events.append(line[5:].strip())

    import json
    text_parts = []
    for e in events:
        if not e:
            continue
        try:
            obj = json.loads(e)
        except Exception:
            continue
        if 'text' in obj:
            text_parts.append(obj['text'])

    full_text = ''.join(text_parts).lower()
    abstention_phrases = [
        'cannot find',
        'insufficient information',
        'not in the provided document',
        'no information',
    ]
    assert any(p in full_text for p in abstention_phrases), (
        f'Expected abstention but got: {full_text[:200]}'
    )
