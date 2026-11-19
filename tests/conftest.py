# tests/conftest.py
import pytest
import os
from pathlib import Path

os.environ.setdefault('SESSION_SECRET_KEY', 'test-secret-key-for-unit-tests-only')
os.environ.setdefault('DEVICE', 'cpu')
os.environ.setdefault('VLLM_BASE_URL', 'http://localhost:11436')


@pytest.fixture
def sample_pdf_path() -> Path:
    return Path(__file__).parents[1] / 'examples' / 'board-minutes.pdf'


@pytest.fixture
def sample_chunks() -> list:
    return [
        {
            'text': 'The board approved a capital expenditure of $5.2 million for Q3 2026.',
            'page_number': 3,
            'chunk_index': 0,
            'section_heading': 'Capital Expenditure',
            'chunk_type': 'text',
            'doc_hash': 'abc123',
            'token_count': 22,
            'embedding_prefix': '[Page 3, Section: Capital Expenditure]',
        },
        {
            'text': 'Revenue for the fiscal year reached $120 million, up 15% year-over-year.',
            'page_number': 5,
            'chunk_index': 1,
            'section_heading': 'Financial Summary',
            'chunk_type': 'text',
            'doc_hash': 'abc123',
            'token_count': 20,
            'embedding_prefix': '[Page 5, Section: Financial Summary]',
        },
    ]
