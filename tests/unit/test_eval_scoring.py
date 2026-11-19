from examples.generate_board_packs import load_board_packs
from tests.rag_eval.eval_runner import score_answer, replay_governance


def test_scoring_counts_uncited_prose_and_rejects_extra_citations():
    case = {'facts': [{'match': 'approved phase one', 'pages': [2]}]}
    chunk = {'page_number': 2}
    prose = score_answer(case, 'The Board approved phase one.', [chunk])
    assert prose['facts_found'] == 0
    assert prose['claims'][0]['has_citation'] is False
    answer = score_answer(case, '- The Board approved phase one. [Page 2] [Page 3]', [chunk])
    assert answer['retrieved_gold_pages'] == 1
    assert answer['claims'][0]['annotated_support'] is False


def test_governance_replay_flags_authority_transfer_and_preserves_valid_approvals():
    cases = [
        {'pack': 'harbor-works', 'page': 1, 'claim': 'The Operations Committee approved the dock electrification proposal.', 'should_warn': True},
        {'pack': 'cedar-transit', 'page': 2, 'claim': 'The Audit Committee approved the 2027 internal audit plan.', 'should_warn': False},
    ]
    metrics = replay_governance(load_board_packs(), cases)['metrics']
    assert metrics['without_guard']['false_negatives'] == 1
    assert metrics['with_guard']['true_positives'] == 1
    assert metrics['with_guard']['false_positives'] == 0
