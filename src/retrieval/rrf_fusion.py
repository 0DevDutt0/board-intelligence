# src/retrieval/rrf_fusion.py


def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    scores = {}
    chunk_map = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            key = chunk.get('chunk_index')
            if key is None:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = chunk

    sorted_keys = sorted(scores.keys(), key=lambda k_: scores[k_], reverse=True)
    result = []
    for key in sorted_keys:
        chunk = dict(chunk_map[key])
        chunk['rrf_score'] = scores[key]
        result.append(chunk)
    return result
