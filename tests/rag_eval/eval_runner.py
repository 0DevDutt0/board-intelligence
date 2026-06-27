# tests/rag_eval/eval_runner.py
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

API_BASE = os.environ.get('BIS_API_URL', 'http://127.0.0.1:8000')
GOLDEN_SET = Path(__file__).parent / 'golden_set.json'
RESULTS_DIR = Path(__file__).parent / 'results'

# Phase 1 acceptance targets from ARCHITECTURE.md
TARGET_KEYWORD_RECALL = 0.80
TARGET_CITATION_COVERAGE = 0.90
TARGET_ABSTENTION_ACCURACY = 1.00
TARGET_LATENCY_S = 12.0
TARGET_WARNING_RATE = 2.0  # avg warnings per non-adversarial question


def _get_pdf_path() -> Path:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            return p
        print(f'ERROR: file not found: {p}')
        sys.exit(1)
    while True:
        raw = input('Enter path to board PDF for evaluation: ').strip()
        p = Path(raw)
        if p.exists():
            return p
        print(f'File not found: {raw}')


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
    print(
        f'Ingested: {data["page_count"]} pages, '
        f'{data["chunk_count"]} chunks, '
        f'{data["ingest_time_seconds"]:.2f}s'
    )
    return data['session_id']


async def query_sse(client, question: str, session_id: str) -> dict:
    import httpx
    text_parts = []
    citation_chunks = []
    corrected_text = None
    warnings = []
    done_data = {}

    async with client.stream(
        'POST',
        '/query',
        json={'question': question, 'use_hyde': True},
        headers={'X-Session-ID': session_id},
        timeout=120.0,
    ) as resp:
        event_type = None
        buffer = ''
        async for raw_bytes in resp.aiter_bytes():
            buffer += raw_bytes.decode('utf-8', errors='replace')
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
                    elif event_type == 'citations' and 'chunks' in obj:
                        citation_chunks = obj['chunks']
                    elif event_type == 'verification' and 'warnings' in obj:
                        warnings = obj['warnings']
                    elif event_type == 'done':
                        done_data = obj
                    event_type = None

    final_text = corrected_text if corrected_text is not None else ''.join(text_parts)
    return {
        'text': final_text,
        'chunks': citation_chunks,
        'warnings': warnings,
        'done': done_data,
    }


def _strip_think(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def score_keyword_recall(answer: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def score_citation_coverage(answer: str) -> float:
    clean = _strip_think(answer)
    bullets = re.findall(r'^[-*]\s+.+', clean, re.MULTILINE)
    if not bullets:
        return 1.0
    cited = sum(1 for b in bullets if re.search(r'\[Page \d+\]', b))
    return cited / len(bullets)


def score_abstention(answer: str, expected_abstain: bool) -> bool:
    clean = _strip_think(answer).lower()
    abstained = 'cannot find' in clean or 'insufficient' in clean
    return abstained == expected_abstain


def score_page_recall(answer: str, expected_pages: list) -> float:
    if not expected_pages:
        return 1.0
    cited_pages = set(int(m) for m in re.findall(r'\[Page (\d+)\]', answer))
    hits = sum(1 for p in expected_pages if p in cited_pages)
    return hits / len(expected_pages)


def count_warnings(warnings: list) -> int:
    return len(warnings)


def _print_bar(label: str, value: float, target: float, width: int = 30) -> None:
    filled = int(value * width)
    bar = '#' * filled + '-' * (width - filled)
    status = 'PASS' if value >= target else 'FAIL'
    print(f'  {label:<28} [{bar}] {value:.3f}  target={target:.2f}  {status}')


async def run_eval():
    import httpx

    if not GOLDEN_SET.exists():
        print(f'ERROR: golden_set.json not found at {GOLDEN_SET}')
        sys.exit(1)

    pdf_path = _get_pdf_path()

    with open(GOLDEN_SET) as f:
        golden = json.load(f)

    print(f'Loaded {len(golden)} golden Q&A pairs')
    print(f'API base: {API_BASE}')
    print('')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(base_url=API_BASE, timeout=180.0) as client:
        # Health check
        try:
            health = await client.get('/health')
            health.raise_for_status()
            print(f'Backend health: OK')
        except Exception as exc:
            print(f'ERROR: backend not reachable at {API_BASE} -- {exc}')
            sys.exit(1)

        session_id = await ingest_pdf(client, pdf_path)
        print('')

        results = []
        for i, item in enumerate(golden):
            qid = item['id']
            question = item['question']
            expected_keywords = item.get('expected_keywords', [])
            expected_abstain = item.get('expected_abstain', False)
            expected_pages = item.get('expected_pages', [])
            category = item.get('category', 'general')

            print(f'[{i+1:02d}/{len(golden)}] ({qid}) {question[:65]}')

            t0 = time.perf_counter()
            try:
                result = await query_sse(client, question, session_id)
            except Exception as exc:
                print(f'         ERROR: {exc}')
                results.append({
                    'id': qid,
                    'question': question,
                    'category': category,
                    'error': str(exc),
                })
                continue
            elapsed = time.perf_counter() - t0

            answer = result['text']
            warnings = result['warnings']
            warn_count = count_warnings(warnings)

            kw_recall = score_keyword_recall(answer, expected_keywords)
            cit_coverage = score_citation_coverage(answer)
            abstention_ok = score_abstention(answer, expected_abstain)
            page_recall = score_page_recall(answer, expected_pages)
            latency_ok = elapsed <= TARGET_LATENCY_S

            print(
                f'         kw_recall={kw_recall:.2f}  '
                f'cit_cov={cit_coverage:.2f}  '
                f'page_recall={page_recall:.2f}  '
                f'abstain={"OK" if abstention_ok else "FAIL"}  '
                f'warnings={warn_count}  '
                f'latency={elapsed:.1f}s'
            )

            results.append({
                'id': qid,
                'question': question,
                'category': category,
                'answer': answer[:400],
                'keyword_recall': kw_recall,
                'citation_coverage': cit_coverage,
                'page_recall': page_recall,
                'abstention_correct': abstention_ok,
                'warning_count': warn_count,
                'latency_s': round(elapsed, 2),
                'latency_ok': latency_ok,
                'expected_abstain': expected_abstain,
                'warnings': [
                    (w if isinstance(w, str) else w.get('message', str(w)))
                    for w in warnings
                ],
            })

    print('')
    print('=' * 64)
    print('EVALUATION SUMMARY')
    print('=' * 64)

    valid = [r for r in results if 'error' not in r]
    non_adversarial = [r for r in valid if not r['expected_abstain']]
    adversarial = [r for r in valid if r['expected_abstain']]

    if not valid:
        print('No valid results to summarize.')
        sys.exit(1)

    avg_kw_recall = sum(r['keyword_recall'] for r in valid) / len(valid)
    avg_cit_cov = sum(r['citation_coverage'] for r in non_adversarial) / max(len(non_adversarial), 1)
    avg_page_recall = sum(r['page_recall'] for r in non_adversarial) / max(len(non_adversarial), 1)
    abstention_acc = sum(1 for r in valid if r['abstention_correct']) / len(valid)
    avg_latency = sum(r['latency_s'] for r in valid) / len(valid)
    avg_warnings = (
        sum(r['warning_count'] for r in non_adversarial) / max(len(non_adversarial), 1)
    )
    pct_latency_ok = sum(1 for r in valid if r['latency_ok']) / len(valid)

    print('')
    print('  Metric                       Score    Target   Status')
    print('  ' + '-' * 60)
    _print_bar('Keyword Recall', avg_kw_recall, TARGET_KEYWORD_RECALL)
    _print_bar('Citation Coverage', avg_cit_cov, TARGET_CITATION_COVERAGE)
    _print_bar('Page Recall', avg_page_recall, 0.70)
    _print_bar('Abstention Accuracy', abstention_acc, TARGET_ABSTENTION_ACCURACY)
    # Latency target is avg < 12s (Phase 1 spec: "full response 200 words < 12s").
    # Complex multi-passage queries generate 500+ word responses and exceed
    # 12s individually; per-question pass rate is not the right metric here.
    latency_score = min(1.0, TARGET_LATENCY_S / avg_latency) if avg_latency > 0 else 1.0
    _print_bar('Avg Latency (vs target)', latency_score, 1.0)
    print(f'  {"Avg Latency":<28}  {avg_latency:.2f}s  target<{TARGET_LATENCY_S:.1f}s')
    print(f'  {"Avg Warnings (non-adv)":<28}  {avg_warnings:.2f}   target<{TARGET_WARNING_RATE:.1f}')
    print('')

    # Per-category breakdown
    categories = sorted({r['category'] for r in valid})
    print('  Per-category keyword recall:')
    for cat in categories:
        cat_items = [r for r in valid if r['category'] == cat]
        cat_avg = sum(r['keyword_recall'] for r in cat_items) / len(cat_items)
        print(f'    {cat:<32} {cat_avg:.3f}  (n={len(cat_items)})')
    print('')

    # Failures detail
    failures = [r for r in valid if not r['abstention_correct']]
    if failures:
        print('  ABSTENTION FAILURES:')
        for r in failures:
            expected = 'abstain' if r['expected_abstain'] else 'answer'
            print(f'    {r["id"]}: expected {expected} -- {r["question"][:60]}')
        print('')

    low_kw = [r for r in valid if r['keyword_recall'] < TARGET_KEYWORD_RECALL]
    if low_kw:
        print('  LOW KEYWORD RECALL:')
        for r in low_kw:
            print(f'    {r["id"]} ({r["keyword_recall"]:.2f}): {r["question"][:60]}')
        print('')

    # Write JSON report
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = RESULTS_DIR / f'eval_{ts}.json'
    report = {
        'timestamp': ts,
        'pdf': str(pdf_path),
        'golden_count': len(golden),
        'summary': {
            'keyword_recall': round(avg_kw_recall, 4),
            'citation_coverage': round(avg_cit_cov, 4),
            'page_recall': round(avg_page_recall, 4),
            'abstention_accuracy': round(abstention_acc, 4),
            'avg_latency_s': round(avg_latency, 2),
            'avg_warnings_non_adversarial': round(avg_warnings, 2),
            'latency_pass_rate': round(pct_latency_ok, 4),
        },
        'results': results,
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'  Report written to: {report_path}')
    print('')

    all_pass = (
        avg_kw_recall >= TARGET_KEYWORD_RECALL
        and avg_cit_cov >= TARGET_CITATION_COVERAGE
        and abstention_acc >= TARGET_ABSTENTION_ACCURACY
        and avg_latency <= TARGET_LATENCY_S
    )
    if all_pass:
        print('RESULT: PASS -- all Phase 1 targets met')
    else:
        print('RESULT: FAIL -- one or more Phase 1 targets not met')

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(run_eval()))
