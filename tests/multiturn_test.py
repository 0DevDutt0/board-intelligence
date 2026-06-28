# tests/multiturn_test.py
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

API_BASE = os.environ.get('BIS_API_URL', 'http://127.0.0.1:8000')

# Multi-turn conversation scenarios.
# Each scenario is a list of turns; each turn has a question and checks.
# checks:
#   context_words  -- words that MUST appear in the response (case-insensitive)
#   absent_words   -- words that must NOT appear (wrong-topic hallucination)
#   min_citations  -- minimum [Page N] citations expected
#   references_prior -- True if the question relies on prior context (pronoun/ellipsis)
SCENARIOS = [
    {
        'name': 'Committee follow-up with pronoun resolution',
        'turns': [
            {
                'question': 'What are the responsibilities of the HSSC?',
                'checks': {
                    'context_words': ['health', 'safety', 'sustainability'],
                    'min_citations': 3,
                    'references_prior': False,
                },
            },
            {
                'question': 'How many meetings did it hold in 2025?',
                'checks': {
                    'context_words': ['four', 'meetings'],
                    'min_citations': 1,
                    'references_prior': True,
                },
            },
            {
                'question': 'Who are its members?',
                'checks': {
                    'context_words': ['Philippe', 'Leanne', 'Catrin'],
                    'min_citations': 1,
                    'references_prior': True,
                },
            },
        ],
    },
    {
        'name': 'Governance deep-dive across turns',
        'turns': [
            {
                'question': 'What is the GDA?',
                'checks': {
                    'context_words': ['Group Delegated Authorities', 'approval'],
                    'min_citations': 1,
                    'references_prior': False,
                },
            },
            {
                'question': 'What financial thresholds does it define?',
                'checks': {
                    'context_words': ['CHF', 'million'],
                    'min_citations': 1,
                    'references_prior': True,
                },
            },
        ],
    },
    {
        'name': 'Comparison across turns',
        'turns': [
            {
                'question': 'What does the Audit Committee oversee?',
                'checks': {
                    'context_words': ['audit', 'financial'],
                    'min_citations': 1,
                    'references_prior': False,
                },
            },
            {
                'question': 'How does that compare to the HSSC?',
                'checks': {
                    'context_words': ['HSSC', 'health', 'sustainability'],
                    'min_citations': 1,
                    'references_prior': True,
                },
            },
        ],
    },
    {
        'name': 'Context window stress (4 turns)',
        'turns': [
            {
                'question': 'What are Holcim\'s corporate governance objectives?',
                'checks': {'context_words': ['value', 'reputation'], 'min_citations': 1, 'references_prior': False},
            },
            {
                'question': 'What internal control frameworks support those objectives?',
                'checks': {'context_words': ['GIC', 'MCS'], 'min_citations': 1, 'references_prior': True},
            },
            {
                'question': 'Who is responsible for the GIC?',
                'checks': {'context_words': ['Group Internal Control'], 'min_citations': 1, 'references_prior': False},
            },
            {
                'question': 'What compliance programs does it run?',
                'checks': {'context_words': ['compliance', 'risk'], 'min_citations': 1, 'references_prior': True},
            },
        ],
    },
]


async def ingest_pdf(client, pdf_path: Path) -> str:
    import httpx
    print(f'Ingesting {pdf_path.name}...')
    with open(pdf_path, 'rb') as f:
        resp = await client.post(
            '/ingest',
            files={'file': (pdf_path.name, f, 'application/pdf')},
        )
    resp.raise_for_status()
    data = resp.json()
    print(f'Session: {data["session_id"][:8]}...  pages={data["page_count"]}  chunks={data["chunk_count"]}')
    return data['session_id']


async def query_sse(client, question: str, session_id: str) -> dict:
    import httpx
    text_parts = []
    corrected_text = None
    warnings = []
    done_data = {}

    async with client.stream(
        'POST', '/query',
        json={'question': question, 'use_hyde': True},
        headers={'X-Session-ID': session_id},
        timeout=120.0,
    ) as resp:
        event_type = None
        buffer = ''
        async for raw in resp.aiter_bytes():
            buffer += raw.decode('utf-8', errors='replace')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.rstrip('\r')
                if line.startswith('event:'):
                    event_type = line[6:].strip()
                elif line.startswith('data:'):
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    try:
                        obj = json.loads(payload)
                    except Exception:
                        continue
                    if event_type == 'chunk' and 'text' in obj:
                        text_parts.append(obj['text'])
                    elif event_type == 'corrected' and 'text' in obj:
                        corrected_text = obj['text']
                    elif event_type == 'verification' and 'warnings' in obj:
                        warnings = obj['warnings']
                    elif event_type == 'done':
                        done_data = obj
                    event_type = None

    text = corrected_text if corrected_text is not None else ''.join(text_parts)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return {'text': text, 'warnings': warnings, 'done': done_data}


def _check_turn(result: dict, checks: dict, turn_idx: int) -> list:
    issues = []
    text_lower = result['text'].lower()

    for word in checks.get('context_words', []):
        if word.lower() not in text_lower:
            issues.append(f'Missing expected word: "{word}"')

    for word in checks.get('absent_words', []):
        if word.lower() in text_lower:
            issues.append(f'Unexpected word present: "{word}"')

    citation_count = len(re.findall(r'\[Page \d+\]', result['text']))
    min_cit = checks.get('min_citations', 0)
    if citation_count < min_cit:
        issues.append(f'Too few citations: {citation_count} < {min_cit}')

    token_count = result['done'].get('total_tokens', 0)
    if token_count < 50:
        issues.append(f'Response suspiciously short: {token_count} tokens (context budget collapse?)')

    return issues


async def run_scenario(client, scenario: dict, pdf_path: Path) -> dict:
    print(f'\n  Scenario: {scenario["name"]}')
    print(f'  {"-" * 50}')

    session_id = await ingest_pdf(client, pdf_path)
    results = []

    for i, turn in enumerate(scenario['turns']):
        q = turn['question']
        prior_label = ' [relies on prior context]' if turn['checks'].get('references_prior') else ''
        print(f'\n  Turn {i+1}{prior_label}: {q}')

        t0 = time.perf_counter()
        try:
            result = await query_sse(client, q, session_id)
        except Exception as exc:
            print(f'    ERROR: {exc}')
            results.append({'turn': i+1, 'question': q, 'error': str(exc)})
            continue
        elapsed = time.perf_counter() - t0

        tokens = result['done'].get('total_tokens', 0)
        latency_ms = result['done'].get('latency_ms', 0)
        citations = len(re.findall(r'\[Page \d+\]', result['text']))
        warnings = len(result['warnings'])

        preview = result['text'][:120].replace('\n', ' ')
        print(f'    {tokens} tokens  {elapsed:.1f}s  {citations} citations  {warnings} warnings')
        print(f'    Preview: {preview}...')

        issues = _check_turn(result, turn['checks'], i)
        if issues:
            for issue in issues:
                print(f'    ISSUE: {issue}')
        else:
            print(f'    OK')

        results.append({
            'turn': i + 1,
            'question': q,
            'references_prior': turn['checks'].get('references_prior', False),
            'tokens': tokens,
            'latency_s': round(elapsed, 2),
            'citations': citations,
            'warnings': warnings,
            'issues': issues,
            'answer_preview': result['text'][:300],
        })

    return {'scenario': scenario['name'], 'turns': results}


async def main():
    import httpx

    if len(sys.argv) < 2:
        while True:
            raw = input('Enter path to board PDF: ').strip()
            p = Path(raw)
            if p.exists():
                pdf_path = p
                break
            print(f'File not found: {raw}')
    else:
        pdf_path = Path(sys.argv[1])
        if not pdf_path.exists():
            print(f'ERROR: {pdf_path} not found')
            sys.exit(1)

    print('Multi-Turn Conversation Test')
    print('=' * 60)
    print(f'PDF: {pdf_path.name}')
    print(f'API: {API_BASE}')

    async with httpx.AsyncClient(base_url=API_BASE, timeout=180.0) as client:
        try:
            health = await client.get('/health')
            health.raise_for_status()
            print('Backend: OK\n')
        except Exception as exc:
            print(f'ERROR: backend unreachable -- {exc}')
            sys.exit(1)

        all_results = []
        for scenario in SCENARIOS:
            result = await run_scenario(client, scenario, pdf_path)
            all_results.append(result)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)

    total_turns = 0
    total_prior_turns = 0
    prior_pass = 0
    all_issues = []

    for scenario_result in all_results:
        turns = [t for t in scenario_result['turns'] if 'error' not in t]
        issues_in_scenario = [i for t in turns for i in t.get('issues', [])]
        status = 'PASS' if not issues_in_scenario else f'FAIL ({len(issues_in_scenario)} issues)'
        print(f'  {scenario_result["scenario"][:45]:<45} {status}')

        for t in turns:
            total_turns += 1
            if t.get('references_prior'):
                total_prior_turns += 1
                if not t.get('issues'):
                    prior_pass += 1
        all_issues.extend(issues_in_scenario)

    print('')
    if total_prior_turns > 0:
        print(f'  Context-dependent turns passed: {prior_pass}/{total_prior_turns}')
    if all_issues:
        print(f'\n  Issues found:')
        for issue in all_issues:
            print(f'    - {issue}')
    else:
        print('  No issues found across all scenarios.')


if __name__ == '__main__':
    asyncio.run(main())
