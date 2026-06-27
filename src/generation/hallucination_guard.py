# src/generation/hallucination_guard.py
import asyncio
import re
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

# Matches numbers including thousands separators (comma, period, space between digits)
# Examples: 12,249,093  12 249 093  12.249.093  1.5  0.80  2025
_NUMBER_RE = re.compile(r'\b\d+(?:[,.\s]\d{3})*(?:[.,]\d+)?\b')

# Executive/authority verbs that change legal meaning when falsely attributed to
# a committee. Governance documents distinguish advisory verbs (reviews, discusses)
# from executive verbs (approves, manages).
_EXEC_VERB_RE = re.compile(
    r'\b(approves?|approved|approving'
    r'|manages?|managed|managing'
    r'|oversees?|oversaw|overseeing'
    r'|develops?|developed|developing'
    r'|implements?|implemented|implementing'
    r'|decides?|decided|deciding'
    r'|authorizes?|authorized|authorizing'
    r'|directs?|directed|directing'
    r'|establishes?|established|establishing'
    r'|executes?|executed|executing'
    r'|maintains?|maintained|maintaining'
    r'|promotes?|promoted|promoting'
    r'|leads?|led|leading'
    r'|sets?|setting'
    r'|adopts?|adopted|adopting'
    r'|amends?|amended|amending'
    r'|ratifies?|ratified|ratifying'
    r')\b',
    re.IGNORECASE,
)


def _verb_pattern(forms: str) -> re.Pattern:
    return re.compile(forms, re.IGNORECASE)


_EXEC_VERB_PATTERNS = {
    'approve':      _verb_pattern(r'\bapprove[sd]?\b|\bapproving\b'),
    'approves':     _verb_pattern(r'\bapprove[sd]?\b|\bapproving\b'),
    'approved':     _verb_pattern(r'\bapprove[sd]?\b|\bapproving\b'),
    'approving':    _verb_pattern(r'\bapprove[sd]?\b|\bapproving\b'),
    'manage':       _verb_pattern(r'\bmanage[sd]?\b|\bmanaging\b'),
    'manages':      _verb_pattern(r'\bmanage[sd]?\b|\bmanaging\b'),
    'managed':      _verb_pattern(r'\bmanage[sd]?\b|\bmanaging\b'),
    'managing':     _verb_pattern(r'\bmanage[sd]?\b|\bmanaging\b'),
    'oversee':      _verb_pattern(r'\boversee[s]?\b|\boversaw\b|\boverseeing\b'),
    'oversees':     _verb_pattern(r'\boversee[s]?\b|\boversaw\b|\boverseeing\b'),
    'oversaw':      _verb_pattern(r'\boversee[s]?\b|\boversaw\b|\boverseeing\b'),
    'overseeing':   _verb_pattern(r'\boversee[s]?\b|\boversaw\b|\boverseeing\b'),
    'develop':      _verb_pattern(r'\bdevelop[s]?\b|\bdeveloped\b|\bdeveloping\b'),
    'develops':     _verb_pattern(r'\bdevelop[s]?\b|\bdeveloped\b|\bdeveloping\b'),
    'developed':    _verb_pattern(r'\bdevelop[s]?\b|\bdeveloped\b|\bdeveloping\b'),
    'developing':   _verb_pattern(r'\bdevelop[s]?\b|\bdeveloped\b|\bdeveloping\b'),
    'implement':    _verb_pattern(r'\bimplement[s]?\b|\bimplemented\b|\bimplementing\b'),
    'implements':   _verb_pattern(r'\bimplement[s]?\b|\bimplemented\b|\bimplementing\b'),
    'implemented':  _verb_pattern(r'\bimplement[s]?\b|\bimplemented\b|\bimplementing\b'),
    'implementing': _verb_pattern(r'\bimplement[s]?\b|\bimplemented\b|\bimplementing\b'),
    'decide':       _verb_pattern(r'\bdecide[sd]?\b|\bdeciding\b'),
    'decides':      _verb_pattern(r'\bdecide[sd]?\b|\bdeciding\b'),
    'decided':      _verb_pattern(r'\bdecide[sd]?\b|\bdeciding\b'),
    'deciding':     _verb_pattern(r'\bdecide[sd]?\b|\bdeciding\b'),
    'authorize':    _verb_pattern(r'\bauthorize[sd]?\b|\bauthorizing\b'),
    'authorizes':   _verb_pattern(r'\bauthorize[sd]?\b|\bauthorizing\b'),
    'authorized':   _verb_pattern(r'\bauthorize[sd]?\b|\bauthorizing\b'),
    'authorizing':  _verb_pattern(r'\bauthorize[sd]?\b|\bauthorizing\b'),
    'direct':       _verb_pattern(r'\bdirect[s]?\b|\bdirected\b|\bdirecting\b'),
    'directs':      _verb_pattern(r'\bdirect[s]?\b|\bdirected\b|\bdirecting\b'),
    'directed':     _verb_pattern(r'\bdirect[s]?\b|\bdirected\b|\bdirecting\b'),
    'directing':    _verb_pattern(r'\bdirect[s]?\b|\bdirected\b|\bdirecting\b'),
    'establish':    _verb_pattern(r'\bestablish(?:es|ed|ing)?\b'),
    'establishes':  _verb_pattern(r'\bestablish(?:es|ed|ing)?\b'),
    'established':  _verb_pattern(r'\bestablish(?:es|ed|ing)?\b'),
    'establishing': _verb_pattern(r'\bestablish(?:es|ed|ing)?\b'),
    'execute':      _verb_pattern(r'\bexecute[sd]?\b|\bexecuting\b'),
    'executes':     _verb_pattern(r'\bexecute[sd]?\b|\bexecuting\b'),
    'executed':     _verb_pattern(r'\bexecute[sd]?\b|\bexecuting\b'),
    'executing':    _verb_pattern(r'\bexecute[sd]?\b|\bexecuting\b'),
    'maintain':     _verb_pattern(r'\bmaintain[s]?\b|\bmaintained\b|\bmaintaining\b'),
    'maintains':    _verb_pattern(r'\bmaintain[s]?\b|\bmaintained\b|\bmaintaining\b'),
    'maintained':   _verb_pattern(r'\bmaintain[s]?\b|\bmaintained\b|\bmaintaining\b'),
    'maintaining':  _verb_pattern(r'\bmaintain[s]?\b|\bmaintained\b|\bmaintaining\b'),
    'promote':      _verb_pattern(r'\bpromote[sd]?\b|\bpromoting\b'),
    'promotes':     _verb_pattern(r'\bpromote[sd]?\b|\bpromoting\b'),
    'promoted':     _verb_pattern(r'\bpromote[sd]?\b|\bpromoting\b'),
    'promoting':    _verb_pattern(r'\bpromote[sd]?\b|\bpromoting\b'),
    'lead':         _verb_pattern(r'\bleads?\b|\bled\b|\bleading\b'),
    'leads':        _verb_pattern(r'\bleads?\b|\bled\b|\bleading\b'),
    'led':          _verb_pattern(r'\bleads?\b|\bled\b|\bleading\b'),
    'leading':      _verb_pattern(r'\bleads?\b|\bled\b|\bleading\b'),
    'set':          _verb_pattern(r'\bsets?\b|\bsetting\b'),
    'sets':         _verb_pattern(r'\bsets?\b|\bsetting\b'),
    'setting':      _verb_pattern(r'\bsets?\b|\bsetting\b'),
    'adopt':        _verb_pattern(r'\badopt[s]?\b|\badopted\b|\badopting\b'),
    'adopts':       _verb_pattern(r'\badopt[s]?\b|\badopted\b|\badopting\b'),
    'adopted':      _verb_pattern(r'\badopt[s]?\b|\badopted\b|\badopting\b'),
    'adopting':     _verb_pattern(r'\badopt[s]?\b|\badopted\b|\badopting\b'),
    'amend':        _verb_pattern(r'\bamend[s]?\b|\bamended\b|\bamending\b'),
    'amends':       _verb_pattern(r'\bamend[s]?\b|\bamended\b|\bamending\b'),
    'amended':      _verb_pattern(r'\bamend[s]?\b|\bamended\b|\bamending\b'),
    'amending':     _verb_pattern(r'\bamend[s]?\b|\bamended\b|\bamending\b'),
    'ratify':       _verb_pattern(r'\bratif(?:y|ies|ied|ying)\b'),
    'ratifies':     _verb_pattern(r'\bratif(?:y|ies|ied|ying)\b'),
    'ratified':     _verb_pattern(r'\bratif(?:y|ies|ied|ying)\b'),
    'ratifying':    _verb_pattern(r'\bratif(?:y|ies|ied|ying)\b'),
}

# Scope-broadening verbs -- softer than exec verbs but stronger than advisory.
# They imply the committee operationally handles something without using hard
# authority language. Flag as low-severity when source uses only advisory verbs.
# Examples: "addresses people roadmaps" > "discussed people roadmaps".
_SCOPE_VERB_RE = re.compile(
    r'\b(addresses?|addressed|addressing'
    r'|handles?|handled|handling'
    r'|coordinates?|coordinated|coordinating'
    r'|facilitates?|facilitated|facilitating'
    r'|drives?|drove|driving'
    r'|covers?|covered|covering'
    r'|updates?|updated|updating'
    r')\b',
    re.IGNORECASE,
)

_SCOPE_VERB_PATTERNS = {
    'address':      _verb_pattern(r'\baddresse?[sd]?\b|\baddressing\b'),
    'addresses':    _verb_pattern(r'\baddresse?[sd]?\b|\baddressing\b'),
    'addressed':    _verb_pattern(r'\baddresse?[sd]?\b|\baddressing\b'),
    'addressing':   _verb_pattern(r'\baddresse?[sd]?\b|\baddressing\b'),
    'handle':       _verb_pattern(r'\bhandle[sd]?\b|\bhandling\b'),
    'handles':      _verb_pattern(r'\bhandle[sd]?\b|\bhandling\b'),
    'handled':      _verb_pattern(r'\bhandle[sd]?\b|\bhandling\b'),
    'handling':     _verb_pattern(r'\bhandle[sd]?\b|\bhandling\b'),
    'coordinate':   _verb_pattern(r'\bcoordinate[sd]?\b|\bcoordinating\b'),
    'coordinates':  _verb_pattern(r'\bcoordinate[sd]?\b|\bcoordinating\b'),
    'coordinated':  _verb_pattern(r'\bcoordinate[sd]?\b|\bcoordinating\b'),
    'coordinating': _verb_pattern(r'\bcoordinate[sd]?\b|\bcoordinating\b'),
    'facilitate':   _verb_pattern(r'\bfacilitate[sd]?\b|\bfacilitating\b'),
    'facilitates':  _verb_pattern(r'\bfacilitate[sd]?\b|\bfacilitating\b'),
    'facilitated':  _verb_pattern(r'\bfacilitate[sd]?\b|\bfacilitating\b'),
    'facilitating': _verb_pattern(r'\bfacilitate[sd]?\b|\bfacilitating\b'),
    'drive':        _verb_pattern(r'\bdrives?\b|\bdrove\b|\bdriving\b'),
    'drives':       _verb_pattern(r'\bdrives?\b|\bdrove\b|\bdriving\b'),
    'drove':        _verb_pattern(r'\bdrives?\b|\bdrove\b|\bdriving\b'),
    'driving':      _verb_pattern(r'\bdrives?\b|\bdrove\b|\bdriving\b'),
    'cover':        _verb_pattern(r'\bcovers?\b|\bcovered\b|\bcovering\b'),
    'covers':       _verb_pattern(r'\bcovers?\b|\bcovered\b|\bcovering\b'),
    'covered':      _verb_pattern(r'\bcovers?\b|\bcovered\b|\bcovering\b'),
    'covering':     _verb_pattern(r'\bcovers?\b|\bcovered\b|\bcovering\b'),
    'update':       _verb_pattern(r'\bupdate[sd]?\b|\bupdating\b'),
    'updates':      _verb_pattern(r'\bupdate[sd]?\b|\bupdating\b'),
    'updated':      _verb_pattern(r'\bupdate[sd]?\b|\bupdating\b'),
    'updating':     _verb_pattern(r'\bupdate[sd]?\b|\bupdating\b'),
}

# Advisory verbs -- confirms source uses weaker language than response.
_ADVISORY_VERB_RE = re.compile(
    r'\b(reviews?|reviewed|reviewing'
    r'|discusses?|discussed|discussing'
    r'|considers?|considered|considering'
    r'|examines?|examined|examining'
    r'|advises?|advised|advising'
    r'|supports?|supported|supporting'
    r'|monitors?|monitored|monitoring'
    r'|assesses?|assessed|assessing'
    r'|evaluates?|evaluated|evaluating'
    r'|recommends?|recommended|recommending'
    r')\b',
    re.IGNORECASE,
)

# Identifies the committee being analyzed in source text.
_COMMITTEE_RE = re.compile(
    r'\b(HSSC'
    r'|Health,?\s+Safety\s+(?:&|and)\s+Sustainability\s+Committee'
    r'|the\s+Committee'
    r')\b',
    re.IGNORECASE,
)

# Governance bodies that hold executive authority distinct from the subject committee.
_OTHER_BODY_RE = re.compile(
    r'\b(Board\s+of\s+Directors'
    r'|Audit\s+Committee'
    r'|Nomination\s+(?:Compensation\s+(?:&|and)\s+)?(?:Governance\s+)?Committee'
    r'|shareholders?'
    r'|management'
    r'|executive\s+(?:team|committee|management)'
    r'|CEO'
    r'|CFO'
    r')\b',
    re.IGNORECASE,
)

# The governance pattern "supports/advises the Board on X ... It [exec verb]" means
# the pronoun "It" refers to the Board, not the committee. This is the most common
# coreference ambiguity in governance reports.
_ADVISES_BOARD_RE = re.compile(
    r'(?:supports?\s+and\s+advises?|advises?)\s+the\s+Board(?:\s+of\s+Directors)?',
    re.IGNORECASE,
)

# Citation tags like [Page 5] -- strip before verb checks.
_CITATION_RE = re.compile(r'\[Page\s+\d+\]', re.IGNORECASE)

_CLAUSE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+|;\s*')

# Approval verbs that trigger event attribution checking.
# Narrower than the full exec verb list -- only formal decision/approval actions.
_APPROVAL_VERB_RE = re.compile(
    r'\b(approves?|approved|approving'
    r'|adopts?|adopted|adopting'
    r'|ratifies?|ratified|ratifying'
    r'|decides?|decided|deciding'
    r'|authorizes?|authorized|authorizing'
    r')\b',
    re.IGNORECASE,
)

# Shareholder attribution patterns: indicate that a vote/approval belongs to
# shareholders or the AGM, not to any committee.
_SHAREHOLDER_ATTR_RE = re.compile(
    r'(?:shareholder|Annual\s+General\s+Meeting|AGM)\s*(?:approval|vote|support|majority|endorsed)'
    r'|shareholders?\s+(?:approved|voted|authorized|supported|endorsed)'
    r'|(?:approved|adopted|ratified|endorsed)\s+(?:at|by|with)\s+'
    r'(?:the\s+)?(?:AGM|Annual\s+General\s+Meeting|shareholders?)',
    re.IGNORECASE,
)

# Board resolution patterns: indicate approval/decision belongs to the Board.
_BOARD_RESOLUTION_RE = re.compile(
    r'Board\s+(?:of\s+Directors\s+)?(?:approved|resolved|decided|authorized)'
    r'|(?:approved|resolved|decided|authorized)\s+by\s+(?:the\s+)?Board',
    re.IGNORECASE,
)


_HEADING_RE = re.compile(r'^\*{1,2}[^*\n]+\*{1,2}:?\s*$')


def _split_sentences(text: str) -> list:
    # Split on sentence-ending punctuation and on bullet/numbered list markers
    # so each bullet gets its own NLI score rather than the whole formatted block.
    raw = re.split(r'(?<=[.!?])\s+|\n(?=\s*[-*•]\s)', text.strip())
    results = []
    for p in raw:
        p = p.strip()
        if not p:
            continue
        # Skip markdown headings -- they are structural labels, not factual claims.
        if _HEADING_RE.match(p):
            continue
        results.append(p)
    return results


def _normalize_digits(s: str) -> str:
    s = re.sub(r'(\d)[,\s](\d)', r'\1\2', s)
    s = re.sub(r'(\d)[,\s](\d)', r'\1\2', s)
    s = re.sub(r'(\d)[,\s](\d)', r'\1\2', s)
    return s


def _num_present_in_text(num: str, text: str) -> bool:
    if num in text:
        return True
    norm_num  = _normalize_digits(num)
    norm_text = _normalize_digits(text)
    if norm_num in norm_text:
        return True
    digits = re.sub(r'\D', '', num)
    if not digits:
        return True
    try:
        return bool(re.search(r'(?<!\d)' + re.escape(digits) + r'(?!\d)', norm_text))
    except re.error:
        return True


def _extract_numbers(text: str) -> list:
    return _NUMBER_RE.findall(text)


def _chunk_is_about_committee(chunk: dict) -> bool:
    heading = chunk.get('section_heading', '') or ''
    lead = heading + ' ' + (chunk.get('text', '') or '')[:300]
    return bool(_COMMITTEE_RE.search(lead))


def _verb_clearly_committee_in_text(pattern: re.Pattern, text: str) -> bool:
    '''
    Return True only if the exec verb in `text` is clearly attributed to the
    committee -- not to the Board via pronoun reference.

    Handles the governance coreference pattern:
      "The HSSC supports and advises the Board... It approves..."
    Here "It" resolves to the Board (the entity being advised), not the HSSC.

    PDF-extracted text uses single newlines for line-wrapping within sentences.
    We collapse those to spaces so that "The HSSC supports and\nadvises the Board"
    is treated as one clause rather than two, preserving the advisory context
    for pronoun resolution.
    '''
    # Collapse single newlines (line wraps) to spaces; preserve double newlines
    # (paragraph breaks) as proper sentence separators.
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    sentences = _CLAUSE_SPLIT_RE.split(text)
    prev_advises_board = False
    prev_subject = None  # 'committee' | 'other' | None

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Track subject of non-matching sentences for pronoun resolution.
        has_committee = bool(_COMMITTEE_RE.search(sent))
        has_other = bool(_OTHER_BODY_RE.search(sent))
        advises_board_here = bool(_ADVISES_BOARD_RE.search(sent))

        if not pattern.search(sent):
            prev_advises_board = advises_board_here
            if has_committee and not has_other:
                prev_subject = 'committee'
            elif has_other and not has_committee:
                prev_subject = 'other'
            continue

        # This sentence contains the exec verb.

        # Pronoun "It" or "They" after "advises the Board" -> Board is the actor.
        if re.match(r'^(?:it|they)\b', sent, re.IGNORECASE):
            if prev_advises_board:
                return False  # "It" = Board
            return prev_subject == 'committee'

        # Explicit committee mention in same sentence (and not overridden by other body).
        if has_committee and not has_other:
            return True

        # Explicit other body in same sentence.
        if has_other and not has_committee:
            return False

        # No clear subject -- fall back to tracked prior subject.
        return prev_subject == 'committee'

    return False


def _verb_in_committee_chunks(pattern: re.Pattern, chunks: list) -> bool:
    '''True if the exec verb is clearly attributed to the committee in any chunk.'''
    for chunk in chunks:
        if _chunk_is_about_committee(chunk):
            text = chunk.get('text', '')
            result = _verb_clearly_committee_in_text(pattern, text)
            if result:
                return True
    return False


def _verb_in_other_body_chunks(pattern: re.Pattern, chunks: list) -> str:
    '''Return name of another body that holds this verb, or empty string.'''
    for chunk in chunks:
        text = chunk.get('text', '')
        if not pattern.search(text):
            continue
        for sent in _CLAUSE_SPLIT_RE.split(text):
            if not pattern.search(sent):
                continue
            m = _OTHER_BODY_RE.search(sent)
            if m:
                return m.group(0)
    return ''


def _advisory_in_committee_chunks(chunks: list) -> bool:
    '''True if any committee-focused chunk uses advisory language.'''
    for chunk in chunks:
        if _chunk_is_about_committee(chunk):
            if _ADVISORY_VERB_RE.search(chunk.get('text', '')):
                return True
    return False


def _check_event_attribution(sentence: str, chunks: list) -> list:
    '''
    Detect when a response sentence attributes a formal approval/decision to the
    committee but the source shows it belongs to shareholders, the AGM, or the Board.

    Strategy: use specific percentage figures in the response as anchors. If a
    percentage from the response sentence also appears near a shareholder/Board
    attribution pattern in the source, the approval event belongs to that other
    actor, not the committee. Anchoring on percentages keeps false positives low
    without requiring topic matching.

    Example:
      Response: "Approves Climate Report with 89.75% shareholder approval"
      Source: "The report received 89.75% shareholder approval at the AGM"
      -> Event attribution: 89.75% in source belongs to shareholders, not committee
    '''
    clean = _CITATION_RE.sub('', sentence)

    # Only run on sentences with formal approval/decision verbs.
    if not _APPROVAL_VERB_RE.search(clean):
        return []

    # Extract percentage anchors from the response sentence.
    percentages = re.findall(r'\b\d+(?:\.\d+)?%', clean)
    if not percentages:
        return []

    all_chunk_text = ' '.join(c.get('text', '') for c in chunks)
    warnings = []

    for pct in percentages:
        pct_esc = re.escape(pct)

        # Check if this percentage appears within 150 chars of a shareholder attribution.
        shareholder_window = re.compile(
            r'(?:' + pct_esc + r'.{0,150}(?:shareholder|AGM|Annual\s+General\s+Meeting)'
            r'|(?:shareholder|AGM|Annual\s+General\s+Meeting).{0,150}' + pct_esc + r')',
            re.IGNORECASE | re.DOTALL,
        )
        if shareholder_window.search(all_chunk_text):
            warnings.append(
                f'"{pct}" -- in the source this figure is linked to '
                f'shareholder/AGM approval, not a committee decision'
            )
            break

        # Check if this percentage appears within 150 chars of a Board resolution.
        board_window = re.compile(
            r'(?:' + pct_esc + r'.{0,150}Board\s+of\s+Directors'
            r'|Board\s+of\s+Directors.{0,150}' + pct_esc + r')',
            re.IGNORECASE | re.DOTALL,
        )
        if board_window.search(all_chunk_text):
            warnings.append(
                f'"{pct}" -- in the source this figure is linked to '
                f'a Board of Directors decision, not the committee'
            )
            break

    return warnings


def _check_scope_broadening(sentence: str, chunks: list) -> list:
    '''
    Low-severity check: detect scope-broadening verbs (addresses, handles,
    coordinates, facilitates, drives, covers) when source uses only advisory
    language for the committee. These are softer than exec verbs but still
    imply more operational responsibility than the source assigns.
    '''
    clean = _CITATION_RE.sub('', sentence)
    scope_matches = _SCOPE_VERB_RE.findall(clean)
    if not scope_matches:
        return []

    all_chunk_text = ' '.join(c.get('text', '') for c in chunks)
    warnings = []

    for verb in set(v.lower() for v in scope_matches):
        pattern = _SCOPE_VERB_PATTERNS.get(verb)
        if pattern is None:
            continue
        if pattern.search(all_chunk_text):
            continue  # verb is in source -- acceptable paraphrase
        if _advisory_in_committee_chunks(chunks):
            warnings.append(
                f'"{verb}" -- implies more operational authority than the source '
                f'supports; retrieved passages use discusses/reviews for the committee'
            )

    return warnings


def _check_verb_inflation(sentence: str, chunks: list) -> list:
    '''
    Four-case governance verb check per sentence:

    1. Verb not in source at all AND advisory verbs in source
       -> Semantic inflation (advisory language inflated to executive)
    2. Verb not in source at all
       -> Semantic inflation (verb absent from source)
    3. Verb in source but clearly attributed to another body (Board, Audit Committee...)
       -> Attribution error
    4. Verb in source within a committee chunk but in an "advises Board... It [verb]"
       pronoun construction (Board is the actor, not the committee)
       -> Authority transfer (source describes Board's decision after committee advises)
    5. Verb in source AND clearly attributed to the committee
       -> OK, pass through
    '''
    clean = _CITATION_RE.sub('', sentence)
    exec_matches = _EXEC_VERB_RE.findall(clean)
    if not exec_matches:
        return []

    all_chunk_text = ' '.join(c.get('text', '') for c in chunks)
    warnings = []

    for verb in set(v.lower() for v in exec_matches):
        pattern = _EXEC_VERB_PATTERNS.get(verb)
        if pattern is None:
            continue

        verb_in_source = bool(pattern.search(all_chunk_text))

        if not verb_in_source:
            # Cases 1 and 2
            if _advisory_in_committee_chunks(chunks):
                warnings.append(
                    f'"{verb}" -- not found in the source; the retrieved passages '
                    f'use advisory language (reviews/discusses/advises) for the committee'
                )
            else:
                warnings.append(
                    f'"{verb}" -- not found in the retrieved passages'
                )
            continue

        # Verb IS somewhere in source -- check attribution.
        if _verb_in_committee_chunks(pattern, chunks):
            # Case 5: clearly the committee's verb in source -- no warning.
            continue

        # Verb in source but NOT clearly attributed to the committee.
        other_body = _verb_in_other_body_chunks(pattern, chunks)
        if other_body:
            # Case 3: verb explicitly belongs to another body.
            warnings.append(
                f'"{verb}" -- in the source this is attributed to the '
                f'{other_body}, not the committee'
            )
        elif _advisory_in_committee_chunks(chunks):
            # Case 4: committee chunks use advisory language; exec verb appears
            # elsewhere (likely via "advises Board... It [exec verb]" pattern).
            warnings.append(
                f'"{verb}" -- the source says the committee advises the Board '
                f'here; this verb likely belongs to the Board, not the committee'
            )
        else:
            warnings.append(
                f'"{verb}" -- not clearly attributed to the committee '
                f'in the retrieved passages'
            )

    return warnings


class HallucinationGuard:
    def __init__(self, model_path: str = None):
        from transformers import pipeline
        path = model_path or settings.nli_model
        self._pipe = pipeline(
            'zero-shot-classification',
            model=path,
            device=-1,
        )
        self._threshold = settings.nli_entailment_threshold
        logger.info(f'HallucinationGuard loaded NLI model from {path} on CPU')

    def _nli_score(self, sentence: str, premise: str) -> float:
        try:
            result = self._pipe(
                sentence,
                candidate_labels=['entailment', 'neutral', 'contradiction'],
                hypothesis_template='{}',
                multi_label=False,
            )
            scores = dict(zip(result['labels'], result['scores']))
            return scores.get('entailment', 0.0)
        except Exception:
            return 0.0

    def _verify_sentence(self, sentence: str, chunks: list) -> tuple:
        max_entailment = 0.0
        for chunk in chunks:
            score = self._nli_score(sentence, chunk['text'])
            if score > max_entailment:
                max_entailment = score

        verified = max_entailment >= self._threshold

        numeric_warnings = []
        if settings.numeric_verification:
            numbers = _extract_numbers(sentence)
            all_chunk_text = ' '.join(c['text'] for c in chunks)

            for num in numbers:
                digits_only = re.sub(r'\D', '', num)
                if len(digits_only) < 4:
                    continue
                if re.search(r'\[Page\s+' + re.escape(num) + r'\]', sentence):
                    continue
                if not _num_present_in_text(num, all_chunk_text):
                    numeric_warnings.append(
                        f'"{num}" -- this figure does not appear verbatim in the retrieved passages'
                    )

        verb_warnings = _check_verb_inflation(sentence, chunks)
        scope_warnings = _check_scope_broadening(sentence, chunks)
        event_warnings = _check_event_attribution(sentence, chunks)

        return verified, max_entailment, numeric_warnings + verb_warnings + scope_warnings + event_warnings

    def _check_citation_mismatches(self, response_text: str, chunks: list) -> list:
        '''
        Parse individual bullet-level claims from the response and verify that
        each [Page N] citation is supported by a retrieved chunk from that page.

        Two checks per citation:
        1. Page not in retrieved chunks at all -> strong mismatch
        2. Page in chunks but low lexical overlap with the specific claim ->
           soft mismatch, with hint about which page has better support

        Uses words of 5+ characters to avoid noise from common short words.
        One warning per suspicious page number to avoid repetition.
        '''
        # Build page -> chunk text map from retrieved chunks.
        page_map = {}
        for chunk in chunks:
            page = chunk.get('page_number')
            if page is not None:
                page_map.setdefault(int(page), []).append(chunk.get('text', ''))
        retrieved_pages = set(page_map.keys())

        # Parse individual bullets with their trailing [Page N] citations.
        bullet_re = re.compile(r'[-*]\s+(.+?)\s*\[Page\s+(\d+)\]', re.MULTILINE)
        warned_pages = set()
        warnings = []

        for match in bullet_re.finditer(response_text):
            claim_raw = match.group(1).strip()
            page_num = int(match.group(2))

            if page_num in warned_pages:
                continue

            if page_num not in retrieved_pages:
                warnings.append(
                    f'[Page {page_num}] cited but no matching passage was retrieved '
                    f'from that page -- this citation cannot be verified'
                )
                warned_pages.add(page_num)
                continue

            # Check lexical overlap between the claim and the page chunk.
            claim_words = set(re.findall(r'\b[a-z]{5,}\b', claim_raw.lower()))
            if len(claim_words) < 3:
                continue  # Claim too short to check reliably.

            page_text = ' '.join(page_map[page_num]).lower()
            page_words = set(re.findall(r'\b[a-z]{5,}\b', page_text))
            overlap = len(claim_words & page_words) / len(claim_words)

            if overlap < 0.25:
                # Find the page with strongest support.
                best_page = max(
                    (p for p in retrieved_pages if p != page_num),
                    key=lambda p: len(
                        claim_words & set(re.findall(
                            r'\b[a-z]{5,}\b',
                            ' '.join(page_map[p]).lower(),
                        ))
                    ),
                    default=None,
                )
                if best_page is not None:
                    best_overlap = len(
                        claim_words & set(re.findall(
                            r'\b[a-z]{5,}\b',
                            ' '.join(page_map[best_page]).lower(),
                        ))
                    ) / len(claim_words)
                    if best_overlap > overlap + 0.30:
                        warnings.append(
                            f'[Page {page_num}] has weak overlap with this claim '
                            f'({overlap:.0%} shared keywords); '
                            f'page {best_page} appears to be a better source '
                            f'({best_overlap:.0%} overlap)'
                        )
                        warned_pages.add(page_num)

        return warnings

    async def verify(self, response_text: str, chunks: list) -> list:
        '''
        Returns a list of warning dicts: {message: str, snippet: str}.
        snippet is the first 80 chars of the sentence that triggered the warning
        (empty string for bulk citation warnings).
        '''
        loop = asyncio.get_event_loop()
        sentences = _split_sentences(response_text)
        warnings = []

        for sentence in sentences:
            verified, score, extra_warns = await loop.run_in_executor(
                None, self._verify_sentence, sentence, chunks
            )
            snippet = sentence[:80]
            if not verified:
                warnings.append({
                    'message': (
                        f'"{sentence[:60]}..." -- '
                        f'low support ({score:.2f}); this claim is not clearly '
                        f'backed by the retrieved passages'
                    ),
                    'snippet': snippet,
                })
            for w in extra_warns:
                warnings.append({'message': w, 'snippet': snippet})

        # Bulk citation check -- operates at bullet level, not sentence level.
        citation_warnings = await loop.run_in_executor(
            None, self._check_citation_mismatches, response_text, chunks
        )
        for w in citation_warnings:
            warnings.append({'message': w, 'snippet': ''})

        return warnings
