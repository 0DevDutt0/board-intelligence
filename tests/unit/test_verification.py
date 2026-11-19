import pytest

from src.generation.citation_injector import ABSTENTION, extract_claims
from src.generation.hallucination_guard import HallucinationGuard
from src.utils.config import loopback_url


def test_claim_extraction_includes_prose_and_preserves_citations_after_punctuation():
    assert extract_claims(
        '**Decision:**\n\nThe Board approved phase one. [Page 3] It deferred phase two.\n'
        '1. The committee reviewed the proposal. [Page 1]'
    ) == [
        'The Board approved phase one. [Page 3]',
        'It deferred phase two.',
        'The committee reviewed the proposal. [Page 1]',
    ]


@pytest.mark.parametrize('url', ['https://example.com:11436', 'http://localhost@example.com', 'http://127.0.0.1/path'])
def test_inference_rejects_non_loopback_origins_and_paths(url):
    with pytest.raises(ValueError, match='loopback'):
        loopback_url(url)


@pytest.mark.asyncio
async def test_entailment_requires_support_on_each_cited_page():
    guard = object.__new__(HallucinationGuard)
    guard._threshold = 0.7
    guard._nli_score = lambda claim, premise: float('approved phase one' in premise)
    chunks = [
        {'page_number': 1, 'text': 'The Board approved phase one.'},
        {'page_number': 2, 'text': 'The Board deferred phase two.'},
    ]
    outcome = await guard.verify('- The Board approved phase one. [Page 1] [Page 2]', chunks)
    assert outcome['status'] == 'needs_review'
    assert [warning['kind'] for warning in outcome['warnings']] == ['low_support']
    assert 'Page 2' in outcome['warnings'][0]['message']


@pytest.mark.asyncio
async def test_empty_and_uncited_answers_never_receive_support():
    guard = object.__new__(HallucinationGuard)
    assert (await guard.verify('**Decision:**', []))['status'] == 'needs_review'
    result = await guard.verify('The Board approved phase one.', [])
    assert result['status'] == 'needs_review'
    assert result['warnings'][0]['kind'] == 'missing_citation'
    assert (await guard.verify(ABSTENTION, []))['status'] == 'abstained'
