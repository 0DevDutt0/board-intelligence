"""Evaluate the local API on fictional board packs and replay labeled governance claims."""

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

import httpx

from examples.generate_board_packs import build_pdf, load_board_packs
from src.generation.citation_injector import extract_citation_tags, extract_claims, is_abstention
from src.generation.hallucination_guard import governance_findings
from src.utils.config import loopback_url

HERE = Path(__file__).resolve().parent
GOLDEN_SET = HERE / 'golden_set.json'


def ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def score_answer(case, answer, chunks):
    facts = case['facts']
    abstained = is_abstention(answer)
    claims = [] if abstained else extract_claims(answer)
    expected_pages = {page for fact in facts for page in fact['pages']}
    retrieved_pages = {chunk['page_number'] for chunk in chunks}
    fact_hits = set()
    claim_scores = []
    for claim in claims:
        tags = extract_citation_tags(claim)
        plain = re.sub(r'\[Page \d+\]|\[unverifiable citation\]|[*_]', '', claim)
        matching_facts = [i for i, fact in enumerate(facts) if re.search(fact['match'], plain, re.I)]
        cited_support = {page for i in matching_facts for page in facts[i]['pages']}
        fact_hits.update(i for i in matching_facts if set(facts[i]['pages']).intersection(tags))
        invalid_tag = '[unverifiable citation]' in claim
        claim_scores.append({
            'claim': claim,
            'pages': tags,
            'has_citation': bool(tags) or invalid_tag,
            'annotated_support': bool(tags) and not invalid_tag and set(tags).issubset(cited_support),
        })

    return {
        'expected_abstention': not facts,
        'abstained': abstained,
        'facts_found': len(fact_hits),
        'facts_expected': len(facts),
        'retrieved_gold_pages': len(expected_pages & retrieved_pages),
        'gold_pages': len(expected_pages),
        'claims': claim_scores,
    }


def replay_governance(packs, cases):
    rows = []
    for case in cases:
        page = packs[case['pack']]['pages'][case['page'] - 1]
        chunks = [{'text': '\n'.join(page['lines']), 'page_number': case['page'], 'section_heading': page['heading']}]
        findings = governance_findings(case['claim'], chunks)
        rows.append({**case, 'warnings': findings, 'with_guard': bool(findings), 'without_guard': False})
    metrics = {}
    for mode in ('without_guard', 'with_guard'):
        positive = sum(row['should_warn'] for row in rows)
        predicted = sum(row[mode] for row in rows)
        true_positive = sum(row[mode] and row['should_warn'] for row in rows)
        metrics[mode] = {
            'true_positives': true_positive,
            'false_positives': predicted - true_positive,
            'false_negatives': positive - true_positive,
            'true_negatives': len(rows) - positive - (predicted - true_positive),
            'precision': ratio(true_positive, predicted),
            'recall': ratio(true_positive, positive),
        }
    return {'metrics': metrics, 'cases': rows}


async def ask(client, session_id, question):
    events = {}
    token_time = None
    started = perf_counter()
    async with client.stream('POST', '/query', json={'question': question}, headers={'X-Session-ID': session_id}) as res:
        res.raise_for_status()
        event = None
        async for line in res.aiter_lines():
            if line.startswith('event: '):
                event = line[7:]
            elif event and line.startswith('data: '):
                if event == 'chunk':
                    if token_time is None:
                        token_time = perf_counter() - started
                else:
                    events[event] = json.loads(line[6:])
    if 'error' in events or 'corrected' not in events or 'verification' not in events:
        raise RuntimeError(f'Incomplete query stream: {sorted(events)}')
    return {
        'answer': events['corrected']['text'],
        'retrieved': events['citations']['chunks'],
        'verification': events['verification'],
        'first_stream_chunk_seconds': round(token_time, 3) if token_time is not None else None,
        'verified_response_seconds': round(perf_counter() - started, 3),
    }


async def evaluate_api(packs, cases, api_url):
    results = []
    async with httpx.AsyncClient(base_url=api_url, timeout=300, trust_env=False) as client:
        for case in cases:
            pdf = build_pdf(packs[case['pack']])
            res = await client.post('/ingest', content=pdf, headers={
                'Content-Type': 'application/pdf', 'X-Filename': f'{case["pack"]}.pdf',
            })
            res.raise_for_status()
            session_id = res.json()['session_id']
            try:
                result = await ask(client, session_id, case['question'])
            finally:
                (await client.delete(f'/session/{session_id}')).raise_for_status()
            scores = score_answer(case, result['answer'], result['retrieved'])
            results.append({**case, **result, **scores})
            print(f'{case["id"]}: {result["verification"]["status"]}', flush=True)

    claims = [claim for item in results for claim in item['claims']]
    answerable = [item for item in results if not item['expected_abstention']]
    unanswerable = [item for item in results if item['expected_abstention']]
    first_chunks = [r['first_stream_chunk_seconds'] for r in results if r['first_stream_chunk_seconds'] is not None]
    return {
        'metrics': {
            'questions': len(results),
            'full_credited_answers': sum(item['facts_found'] == item['facts_expected'] for item in answerable),
            'answerable_questions': len(answerable),
            'retrieved_page_recall': ratio(
                sum(item['retrieved_gold_pages'] for item in results), sum(item['gold_pages'] for item in results),
            ),
            'claim_citation_coverage': ratio(sum(claim['has_citation'] for claim in claims), len(claims)),
            'annotated_claim_support': ratio(sum(claim['annotated_support'] for claim in claims), len(claims)),
            'correct_abstentions': sum(item['abstained'] for item in unanswerable),
            'unanswerable_questions': len(unanswerable),
            'false_abstentions': sum(item['abstained'] for item in answerable),
            'first_stream_chunk_median_seconds': round(median(first_chunks), 3) if first_chunks else None,
            'verified_response_median_seconds': round(median(r['verified_response_seconds'] for r in results), 3),
        },
        'cases': results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    parser.add_argument('--guard-only', action='store_true', help='Replay labeled claims without a running model')
    parser.add_argument('--output', type=Path, default=HERE / 'results' / 'latest.json')
    args = parser.parse_args()

    golden = json.loads(GOLDEN_SET.read_text(encoding='utf-8'))
    packs = load_board_packs()
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'governance_replay': replay_governance(packs, golden['governance_replay']),
    }
    if not args.guard_only:
        report['api_evaluation'] = asyncio.run(evaluate_api(packs, golden['questions'], loopback_url(args.api)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value['metrics'] for key, value in report.items() if key != 'created_at'}, indent=2))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
