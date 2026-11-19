# src/generation/citation_injector.py
import re


_CITATION_RE = re.compile(r'\[Page (\d+)\]')
_CLAIM_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])|(?<=\])\s+(?=[A-Z0-9])')
_LIST_PREFIX_RE = re.compile(r'^(?:[-*•]|\d+[.)])\s+')
_HEADING_ONLY_RE = re.compile(r'^(?:#{1,6}\s+.+|\*\*[^*]+\*\*:?|[-*_]{3,})$')
ABSTENTION = 'I cannot find sufficient information in the document.'


def is_abstention(text: str) -> bool:
    return text.strip().rstrip('.').casefold() == ABSTENTION.rstrip('.').casefold()


def extract_claims(text: str) -> list[str]:
    claims = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _HEADING_ONLY_RE.fullmatch(line):
            continue
        bullet = _LIST_PREFIX_RE.match(line)
        if bullet:
            claims.append(line[bullet.end():])
        else:
            claims.extend(_CLAIM_BOUNDARY_RE.split(line))
    return claims


def extract_citation_tags(text: str) -> list:
    matches = _CITATION_RE.findall(text)
    return [int(m) for m in matches]


def validate_citations(text: str, valid_pages: set) -> str:
    def replace_tag(match):
        page_num = int(match.group(1))
        if page_num in valid_pages and page_num > 0:
            return match.group(0)
        return '[unverifiable citation]'

    return _CITATION_RE.sub(replace_tag, text)
