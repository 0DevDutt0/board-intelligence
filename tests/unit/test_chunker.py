# tests/unit/test_chunker.py
import pytest
from unittest.mock import patch, MagicMock


def _make_pages(texts_by_page: list) -> list:
    pages = []
    for i, text in enumerate(texts_by_page):
        pages.append({
            'page_number': i + 1,
            'text': text,
            'render_mode': 'native',
            'tables': [],
        })
    return pages


def _chunk_without_model(pages, doc_hash='testhash', **kwargs):
    from src.ingestion.chunker import SemanticChunker
    chunker = SemanticChunker(**kwargs)
    # Patch out the embedding model -- use fixed-window chunking
    with patch.object(chunker, '_detect_topic_shift', return_value=[]):
        return chunker.chunk_pages(pages, doc_hash)


def test_chunk_schema():
    pages = _make_pages(['The board approved a budget of $10 million for the next fiscal year.'])
    chunks = _chunk_without_model(pages)
    assert len(chunks) >= 1
    c = chunks[0]
    assert 'text' in c
    assert 'page_number' in c
    assert 'chunk_index' in c
    assert 'section_heading' in c
    assert 'chunk_type' in c
    assert 'doc_hash' in c
    assert 'token_count' in c
    assert 'embedding_prefix' in c


def test_chunk_page_number_preserved():
    pages = _make_pages([
        'First page content about revenue and budget.',
        'Second page content about headcount and operations.',
    ])
    chunks = _chunk_without_model(pages)
    page_numbers = {c['page_number'] for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers


def test_chunk_doc_hash_propagated():
    pages = _make_pages(['Some board meeting content.'])
    chunks = _chunk_without_model(pages, doc_hash='deadbeef')
    assert all(c['doc_hash'] == 'deadbeef' for c in chunks)


def test_chunk_indices_sequential():
    long_text = ' '.join(['The board reviewed quarterly results.'] * 50)
    pages = _make_pages([long_text])
    chunks = _chunk_without_model(pages)
    indices = [c['chunk_index'] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_table_chunk_type():
    from src.ingestion.chunker import SemanticChunker

    mock_table = MagicMock()
    mock_table.export_to_markdown.return_value = (
        '| Col A | Col B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |'
    )

    pages = [
        {
            'page_number': 2,
            'text': 'Summary of financial tables.',
            'render_mode': 'native',
            'tables': [mock_table],
        }
    ]

    chunker = SemanticChunker()
    with patch.object(chunker, '_detect_topic_shift', return_value=[]):
        chunks = chunker.chunk_pages(pages, 'abc')

    table_chunks = [c for c in chunks if c['chunk_type'] == 'table']
    assert len(table_chunks) >= 1
    assert table_chunks[0]['page_number'] == 2
