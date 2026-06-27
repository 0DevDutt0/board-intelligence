# tests/unit/test_parser.py
import pytest
import io
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_mock_pages(count=5):
    return [
        {
            'page_number': i + 1,
            'text': f'This is page {i + 1} with some content about the board meeting.',
            'render_mode': 'native',
            'tables': [],
        }
        for i in range(count)
    ]


def test_detect_render_mode_native(tmp_path):
    import pymupdf
    pdf_path = tmp_path / 'test.pdf'
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 100), 'This page has lots of native text content for testing purposes.')
    doc.save(str(pdf_path))
    doc.close()

    from src.ingestion.parser import _detect_render_mode
    mode = _detect_render_mode(str(pdf_path))
    assert mode == 'native'


def test_detect_render_mode_scanned(tmp_path):
    import pymupdf
    pdf_path = tmp_path / 'blank.pdf'
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    from src.ingestion.parser import _detect_render_mode
    mode = _detect_render_mode(str(pdf_path))
    assert mode == 'scanned'


def test_parser_falls_back_to_pymupdf(tmp_path):
    import pymupdf
    pdf_path = tmp_path / 'test.pdf'
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 100), 'Board meeting minutes. Revenue $5M.')
    doc.save(str(pdf_path))
    doc.close()

    with patch('src.ingestion.parser._parse_with_docling', side_effect=RuntimeError('docling unavailable')):
        from src.ingestion.parser import DocumentParser
        parser = DocumentParser('/models/docling-models')
        pages = parser.parse(str(pdf_path))
        assert len(pages) >= 1
        assert all('page_number' in p for p in pages)


def test_parser_returns_correct_schema(tmp_path):
    import pymupdf
    pdf_path = tmp_path / 'schema_test.pdf'
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 100), 'The board approved the annual budget of $12 million.')
    doc.save(str(pdf_path))
    doc.close()

    mock_pages = _make_mock_pages(1)
    with patch('src.ingestion.parser._parse_with_docling', return_value=mock_pages):
        from src.ingestion.parser import DocumentParser
        parser = DocumentParser('/models/docling-models')
        pages = parser.parse(str(pdf_path))
        assert len(pages) == 1
        page = pages[0]
        assert 'page_number' in page
        assert 'text' in page
        assert 'render_mode' in page
        assert 'tables' in page
