# tests/rag_eval/eval_runner.py
# Evaluates RAG quality against the golden set using deepeval.
# Usage: python tests/rag_eval/eval_runner.py

import asyncio
import json
import os
import sys
import time
from pathlib import Path

API_BASE = os.environ.get('BIS_API_URL', 'http://127.0.0.1:8000')
GOLDEN_SET = Path(__file__).parent / 'golden_set.json'
TEST_PDF = Path(__file__).parent.parent / 'fixtures' / 'sample_board_doc.pdf'


async def ingest_pdf(client) -> str:
    import httpx
    print(f'Ingesting {TEST_PDF}...')
    with open(TEST_PDF, 'rb') as f:
        resp = await client.post(
            '/ingest',
            files={'file': ('sample_board_doc.pdf', f, 'application/pdf')},
        )
    resp.raise_for_status()
    data = resp.json()
    print(f'Ingested: {data["page_count"]} pages, {data["chunk_count"]} chunks')
    return data['session_id']


async def query_sse(client, question: str, session_id: str) -> tuple:
    import httpx
    text_parts = []
    citation_chunks = []
    async with client.stream(
        'POST',
        '/query',
        json={'question': question, 'use_hyde': True},
        headers={'X-Session-ID': session_id},
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if 'text' in obj:
                text_parts.append(obj['text'])
            if 'chunks' in obj:
                citation_chunks = obj['chunks']
    return ''.join(text_parts), citation_chunks


def citation_coverage(answer: str, expected_answer: str) -> float:
    import re
    tags = re.findall(r'\[Page \d+\]', answer)
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]
    if not sentences:
        return 1.0
    cited = sum(1 for s in sentences if '[Page' in s)
    return cited / len(sentences)


async def run_eval():
    import httpx
    if not GOLDEN_SET.exists():
        print(f'ERROR: golden_set.json not found at {GOLDEN_SET}')
        sys.exit(1)
    if not TEST_PDF.exists():
        print(f'ERROR: sample PDF not found at {TEST_PDF}')
        sys.exit(1)

    with open(GOLDEN_SET) as f:
        golden = json.load(f)

    print(f'Loaded {len(golden)} golden Q&A pairs')
    print('')

    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            ContextualRecallMetric,
        )
        from deepeval.test_case import LLMTestCase
        use_deepeval = True
    except ImportError:
        print('WARNING: deepeval not installed. Running citation coverage only.')
        use_deepeval = False

    async with httpx.AsyncClient(base_url=API_BASE, timeout=180.0) as client:
        session_id = await ingest_pdf(client)

        results = []
        for i, item in enumerate(golden):
            question = item['question']
            expected = item.get('expected_answer', '')
            context = item.get('context', [])

            print(f'[{i+1}/{len(golden)}] {question[:60]}...')
            t0 = time.perf_counter()
            answer, chunks = await query_sse(client, question, session_id)
            elapsed = time.perf_counter() - t0

            coverage = citation_coverage(answer, expected)
            result = {
                'question': question,
                'answer': answer,
                'citation_coverage': coverage,
                'latency_s': round(elapsed, 2),
            }

            if use_deepeval:
                retrieved_ctx = [c['text'] for c in chunks]
                test_case = LLMTestCase(
                    input=question,
                    actual_output=answer,
                    expected_output=expected,
                    retrieval_context=retrieved_ctx,
                )
                result['test_case'] = test_case
            results.append(result)
            print(f'   citation_coverage={coverage:.2f}, latency={elapsed:.2f}s')

    print('')
    print('=' * 60)
    print('EVALUATION RESULTS')
    print('=' * 60)

    avg_coverage = sum(r['citation_coverage'] for r in results) / len(results)
    avg_latency = sum(r['latency_s'] for r in results) / len(results)
    print(f'Citation Coverage (avg): {avg_coverage:.3f}  (target: 1.0)')
    print(f'Average Latency:         {avg_latency:.2f}s')

    if use_deepeval:
        from deepeval import evaluate
        test_cases = [r['test_case'] for r in results]
        metrics = [
            AnswerRelevancyMetric(threshold=0.75),
            FaithfulnessMetric(threshold=0.80),
            ContextualRecallMetric(threshold=0.70),
        ]
        evaluate(test_cases, metrics)

    pass_coverage = avg_coverage >= 1.0
    print('')
    if pass_coverage:
        print('PASS: Citation coverage meets target (1.0)')
    else:
        print(f'FAIL: Citation coverage {avg_coverage:.3f} < 1.0')

    return 0 if pass_coverage else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(run_eval()))
