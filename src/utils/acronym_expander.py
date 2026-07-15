# src/utils/acronym_expander.py
import re

# Governance and Holcim-specific acronym map.
# Keys: uppercase acronym as it appears in queries and chunks.
# Values: full form appended to the BM25 query so cross-form matches are found.
#
# Pattern: if a user asks about "GDA", BM25 will also search for tokens from
# "Group Delegated Authorities", matching chunks that spell out the full name
# without the abbreviation. The reverse path (full form -> acronym) handles
# queries like "What is Group Internal Control?" finding "GIC" chunks.
_ACRONYM_MAP = {
    'GDA':  'Group Delegated Authorities',
    'GIC':  'Group Internal Control',
    'GIA':  'Group Internal Audit',
    'MCS':  'Minimum Control Standards',
    'HSSC': 'Health Safety Sustainability Committee',
    'ERM':  'Enterprise Risk Management',
    'ESG':  'Environmental Social Governance',
    'AGM':  'Annual General Meeting',
    'KPI':  'Key Performance Indicator',
    'KPIS': 'Key Performance Indicators',
    'CEO':  'Chief Executive Officer',
    'CFO':  'Chief Financial Officer',
    'CRO':  'Chief Risk Officer',
    'COO':  'Chief Operating Officer',
    'IT':   'Information Technology',
    'HSE':  'Health Safety Environment',
    'BOD':  'Board of Directors',
    'NCG':  'Nomination Compensation Governance Committee',
    'AC':   'Audit Committee',
    'CHF':  'Swiss Franc',
}

# Reverse map: canonical lowercase full form -> acronym.
_FULL_TO_ACRONYM = {v.lower(): k for k, v in _ACRONYM_MAP.items()}

_WORD_RE = re.compile(r"[A-Za-z']+")

# Governance framework terms whose PURPOSE/ROLE queries should also retrieve
# chunks describing their oversight relationships with the Board, Audit Committee,
# and Executive Committee. Without this expansion, "What is the purpose of ERM?"
# doesn't surface chunks that describe ERM reporting upward through governance.
# Word boundaries are required: bare substring checks made 'gic' match
# 'strategic', 'erm' match 'long-term', and 'ics' match 'logistics'.
_FRAMEWORK_TERM_RE = re.compile(
    r'\b(erm|enterprise risk management'
    r'|gic|group internal control'
    r'|gia|group internal audit'
    r'|gda|group delegated authorities'
    r'|mcs|minimum control standards'
    r'|ics|internal control system)\b',
)

# Query intent patterns that signal the user wants to understand how a framework
# fits into the governance structure, not just what it does in isolation.
_RELATIONSHIP_RE = re.compile(
    r'\b(purpose|role|function|interact|report(?:s|ing)?\s+to|oversight'
    r'|relationship|how\s+does|what\s+does|governed\s+by|accountabl\w*'
    r'|responsib\w*|structure|how\s+is)\b',
    re.IGNORECASE,
)

# Terms added to BM25 when a governance-relationship query is detected.
# Surfaces chunks that name the oversight chain explicitly.
_GOVERNANCE_CHAIN_EXPANSION = (
    'Board of Directors Audit Committee Executive Committee oversight reporting'
)


def expand_query(query: str) -> str:
    additions = []
    seen = set()

    for word in _WORD_RE.findall(query):
        upper = word.upper()
        if upper in _ACRONYM_MAP and upper not in seen:
            additions.append(_ACRONYM_MAP[upper])
            seen.add(upper)

    query_lower = query.lower()
    for full_form, acronym in _FULL_TO_ACRONYM.items():
        if full_form in query_lower and acronym not in seen:
            additions.append(acronym)
            seen.add(acronym)

    # When the query asks about the PURPOSE or ROLE of a governance framework,
    # also expand with oversight body names so BM25 retrieves chunks that
    # describe the governance chain (e.g. ERM reports to Audit Committee).
    framework_hit = bool(_FRAMEWORK_TERM_RE.search(query_lower))
    if framework_hit and _RELATIONSHIP_RE.search(query):
        if 'BOD' not in seen:
            additions.append(_GOVERNANCE_CHAIN_EXPANSION)
            seen.add('BOD')

    if not additions:
        return query

    return query + ' ' + ' '.join(additions)
