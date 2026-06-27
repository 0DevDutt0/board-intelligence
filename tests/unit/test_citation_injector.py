# tests/unit/test_citation_injector.py
import pytest
from src.generation.citation_injector import validate_citations, extract_citation_tags


def test_extract_valid_tags():
    text = 'Revenue grew by 15% [Page 5]. Costs increased [Page 3].'
    tags = extract_citation_tags(text)
    assert tags == [5, 3]


def test_extract_no_tags():
    text = 'This sentence has no citations at all.'
    tags = extract_citation_tags(text)
    assert tags == []


def test_validate_citations_all_valid():
    text = 'Revenue was $120M [Page 5]. Costs were $80M [Page 3].'
    valid_pages = {3, 5, 7}
    result = validate_citations(text, valid_pages)
    assert '[unverifiable citation]' not in result
    assert '[Page 5]' in result
    assert '[Page 3]' in result


def test_validate_citations_invalid_page():
    text = 'Something happened [Page 99].'
    valid_pages = {1, 2, 3}
    result = validate_citations(text, valid_pages)
    assert '[Page 99]' not in result
    assert '[unverifiable citation]' in result


def test_validate_citations_page_zero():
    text = 'A claim [Page 0].'
    valid_pages = {1, 2, 3}
    result = validate_citations(text, valid_pages)
    assert '[unverifiable citation]' in result


def test_validate_mixed_valid_invalid():
    text = 'Valid claim [Page 2]. Invalid claim [Page 500].'
    valid_pages = {1, 2, 3}
    result = validate_citations(text, valid_pages)
    assert '[Page 2]' in result
    assert '[Page 500]' not in result
    assert '[unverifiable citation]' in result
