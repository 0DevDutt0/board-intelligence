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

    if not additions:
        return query

    expanded = query + ' ' + ' '.join(additions)
    return expanded
