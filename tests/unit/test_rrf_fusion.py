# tests/unit/test_rrf_fusion.py
import pytest
from src.retrieval.rrf_fusion import reciprocal_rank_fusion


def test_rrf_basic_merge():
    dense = [{'chunk_index': 0}, {'chunk_index': 1}, {'chunk_index': 2}]
    sparse = [{'chunk_index': 1}, {'chunk_index': 0}, {'chunk_index': 3}]
    result = reciprocal_rank_fusion([dense, sparse], k=60)
    ids = [r['chunk_index'] for r in result]
    # chunk_index 0 and 1 appear in both lists, should rank highest
    assert ids[0] in (0, 1)
    assert ids[1] in (0, 1)


def test_rrf_empty_inputs():
    result = reciprocal_rank_fusion([[], []], k=60)
    assert result == []


def test_rrf_single_list():
    dense = [{'chunk_index': 2}, {'chunk_index': 5}, {'chunk_index': 8}]
    result = reciprocal_rank_fusion([dense], k=60)
    assert [r['chunk_index'] for r in result] == [2, 5, 8]


def test_rrf_k_parameter():
    dense = [{'chunk_index': i} for i in range(10)]
    sparse = [{'chunk_index': i} for i in range(10)]
    result_60 = reciprocal_rank_fusion([dense, sparse], k=60)
    result_1 = reciprocal_rank_fusion([dense, sparse], k=1)
    # Both should produce same ordering (items in same rank both lists)
    assert [r['chunk_index'] for r in result_60] == \
           [r['chunk_index'] for r in result_1]


def test_rrf_scores_sum_correctly():
    # Item at rank 0 in both lists: score = 1/(60+1) + 1/(60+1) = 2/61
    dense = [{'chunk_index': 99}]
    sparse = [{'chunk_index': 99}]
    result = reciprocal_rank_fusion([dense, sparse], k=60)
    assert len(result) == 1
    expected = 2 / 61
    assert abs(result[0]['rrf_score'] - expected) < 1e-9
