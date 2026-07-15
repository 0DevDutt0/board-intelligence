# src/generation/prompt_builder.py

_SYSTEM_PROMPT = (
    'You are a precise board meeting intelligence assistant.\n'
    'Answer strictly using the provided context passages from the document.\n'
    '\n'
    'FORMAT (follow exactly):\n'
    '\n'
    '  **Section Heading:**\n'
    '\n'
    '  - One concise point (5-10 words max) [Page N]\n'
    '  - Another distinct point [Page N]\n'
    '\n'
    'RULES:\n'
    '- Each bold heading on its own line, blank line before bullets.\n'
    '- Each bullet = one unique idea, max 10-12 words, sharp and direct.\n'
    '- COMPLETENESS: when a passage contains multiple related facts (e.g. different\n'
    '  meeting types, additional figures, related dates), include ALL of them as\n'
    '  separate bullets -- do not drop facts that are in the same sentence or passage.\n'
    '- DEDUPLICATE: if the exact same fact appears on multiple pages,\n'
    '  write it ONCE and cite the best page. Do not repeat it.\n'
    '- Cite every bullet with [Page N] at the end.\n'
    '- No full sentences -- use noun phrases or short verb phrases.\n'
    '- Do not merge a heading and a bullet on the same line.\n'
    '- Reproduce all numbers, names, and dates exactly as in the source.\n'
    '\n'
    'CONTENT RULES:\n'
    '- If context lacks the answer, respond exactly:\n'
    '  "I cannot find sufficient information in the document."\n'
    '- Never speculate. Never add information not in the context.\n'
    '- Use the same verbs as in the source passages. Do not substitute\n'
    '  stronger authority verbs (e.g. do not write "approves" if the\n'
    '  source says "reviews", or "manages" if the source says "discusses").\n'
    '\n'
    'VERB EXAMPLE (follow this pattern exactly):\n'
    '  Source: "The committee advises the Board on sustainability strategy"\n'
    '  CORRECT bullet: "Advises Board on sustainability strategy [Page N]"\n'
    '  WRONG bullet:   "Approves sustainability strategy [Page N]"\n'
    '  Source: "The committee reviews KPIs and discusses root causes"\n'
    '  CORRECT bullet: "Reviews KPIs and discusses root causes [Page N]"\n'
    '  WRONG bullet:   "Manages KPI tracking and oversees root cause analysis [Page N]"\n'
)


# Max chars per chunk when building the prompt.
# 8 chunks x 1500 chars = ~12 000 chars = ~3000 tokens of context.
# The model context is 8192 tokens; system prompt + question use ~500 tokens,
# leaving ~4700 tokens for context + output.  Keeping context at ~3000 tokens
# reserves ~1700 tokens for the think block and ~2000+ for the answer.
_CHUNK_CHAR_LIMIT = 1500


def _format_chunk(i: int, chunk: dict) -> str:
    page = chunk.get('page_number', '?')
    chunk_type = chunk.get('chunk_type', 'text')
    text = chunk.get('text', '')
    if len(text) > _CHUNK_CHAR_LIMIT:
        text = text[:_CHUNK_CHAR_LIMIT] + '...'
    return f'--- Passage {i} [Page {page}, {chunk_type}] ---\n{text}'


def build_prompt(
    question: str,
    chunks: list,
    conversation_history: list = None,
    valid_pages: list = None,
) -> list:
    context_parts = [_format_chunk(i + 1, c) for i, c in enumerate(chunks)]
    context = '\n\n'.join(context_parts)

    citation_constraint = ''
    if valid_pages:
        page_list = ', '.join(str(p) for p in sorted(valid_pages))
        citation_constraint = (
            f'\nCITATION CONSTRAINT: You MUST only use these exact page numbers in '
            f'[Page N] citations: {page_list}. '
            f'Do not cite any other page numbers. '
            f'Every bullet must cite one of these pages.\n'
        )

    system_content = (
        _SYSTEM_PROMPT
        + citation_constraint
        + '\nContext passages:\n'
        + context
    )

    messages = [{'role': 'system', 'content': system_content}]

    if conversation_history:
        for exchange in conversation_history:
            if 'user' in exchange:
                messages.append({'role': 'user', 'content': exchange['user']})
            if 'assistant' in exchange:
                messages.append({'role': 'assistant', 'content': exchange['assistant']})

    messages.append({
        'role': 'user',
        'content': (
            question
            + '\n\nAnswer with short bullet points (5-12 words each), grouped under bold headings. '
            'Include ALL related facts from each passage (e.g. if a sentence mentions '
            'ordinary meetings, extraordinary meetings, AND strategy meetings, list all three). '
            'Deduplicate across pages -- same fact once only. '
            'Cite each bullet with [Page N].'
        ),
    })

    return messages
