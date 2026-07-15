# src/generation/citation_injector.py
import re


_CITATION_RE = re.compile(r'\[Page (\d+)\]')


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
