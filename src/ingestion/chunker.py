# src/ingestion/chunker.py
import re
from typing import Optional
from src.utils.config import settings
from src.utils.logger import get_logger
from src.ingestion.table_extractor import TableExtractor

logger = get_logger(__name__)

_APPROX_CHARS_PER_TOKEN = 4


def _count_tokens(text: str) -> int:
    return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)


def _split_sentences(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def _heading_from_text(text: str) -> str:
    lines = text.strip().split('\n')
    for line in lines:
        stripped = line.strip().lstrip('#').strip()
        if stripped and len(stripped) < 120:
            return stripped
    return ''


class SemanticChunker:
    def __init__(
        self,
        target_tokens: int = None,
        min_tokens: int = None,
        overlap_tokens: int = None,
        table_max_tokens: int = None,
    ):
        self._target = target_tokens or settings.chunk_target_tokens
        self._min = min_tokens or settings.chunk_min_tokens
        self._overlap = overlap_tokens or settings.chunk_overlap_tokens
        self._table_extractor = TableExtractor(table_max_tokens)
        self._embed_model = None

    def _get_embed_model(self):
        if self._embed_model is None:
            from FlagEmbedding import BGEM3FlagModel
            self._embed_model = BGEM3FlagModel(
                settings.embedding_model,
                use_fp16=True,
                device=settings.device,
            )
        return self._embed_model

    def _cosine_sim(self, a, b) -> float:
        import numpy as np
        a = a / (float(np.linalg.norm(a)) + 1e-9)
        b = b / (float(np.linalg.norm(b)) + 1e-9)
        return float(np.dot(a, b))

    def _detect_topic_shift(self, sentences: list, embedder=None) -> list:
        if len(sentences) <= 1:
            return []
        if embedder is not None:
            embeddings = embedder.encode_batch(sentences, batch_size=32, max_length=128)
        else:
            model = self._get_embed_model()
            output = model.encode(sentences, batch_size=32, max_length=128)
            embeddings = output['dense_vecs']

        shifts = []
        for i in range(1, len(embeddings)):
            sim = self._cosine_sim(embeddings[i - 1], embeddings[i])
            if sim < 0.7:
                shifts.append(i)
        return shifts

    def _chunk_text(
        self,
        text: str,
        page_number: int,
        doc_hash: str,
        section_heading: str,
        chunk_index_start: int,
        embedder=None,
    ) -> list:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        shift_indices = set()
        try:
            shift_indices = set(self._detect_topic_shift(sentences, embedder=embedder))
        except Exception as exc:
            logger.warning(f'Semantic shift detection failed: {exc}. Using fixed chunking.')

        chunks = []
        current = []
        current_tokens = 0
        idx = chunk_index_start

        def flush(current_sents, overlap_sents, force=False):
            nonlocal idx
            chunk_text = ' '.join(current_sents)
            token_count = _count_tokens(chunk_text)
            if token_count < self._min and not force:
                # Under the minimum: keep accumulating instead of dropping.
                return current_sents
            prefix = f'[Page {page_number}, Section: {section_heading}]'
            chunks.append({
                'text': chunk_text,
                'page_number': page_number,
                'chunk_index': idx,
                'section_heading': section_heading,
                'chunk_type': 'text',
                'doc_hash': doc_hash,
                'token_count': token_count,
                'embedding_prefix': prefix,
            })
            idx += 1
            overlap_token_budget = self._overlap
            overlap = []
            for s in reversed(current_sents):
                t = _count_tokens(s)
                if overlap_token_budget - t >= 0:
                    overlap.insert(0, s)
                    overlap_token_budget -= t
                else:
                    break
            return overlap

        overlap_carry = []

        for i, sent in enumerate(sentences):
            sent_tokens = _count_tokens(sent)
            at_shift = i in shift_indices

            if at_shift and current_tokens >= self._min:
                overlap_carry = flush(current, overlap_carry)
                current = list(overlap_carry)
                current_tokens = sum(_count_tokens(s) for s in current)

            if current_tokens + sent_tokens > self._target and current_tokens >= self._min:
                overlap_carry = flush(current, overlap_carry)
                current = list(overlap_carry)
                current_tokens = sum(_count_tokens(s) for s in current)

            current.append(sent)
            current_tokens += sent_tokens

        # Force the final flush: a page shorter than the minimum (or a sub-min
        # tail) must still be indexed, otherwise its content and page number
        # disappear from the index entirely.
        if current:
            flush(current, overlap_carry, force=True)

        return chunks

    def chunk_pages(self, pages: list, doc_hash: str, embedder=None) -> list:
        all_chunks = []
        chunk_index = 0
        current_heading = 'Document'

        for page in pages:
            page_no = page['page_number']
            page_text = page.get('text', '')
            tables = page.get('tables', [])

            heading = _heading_from_text(page_text) or current_heading
            current_heading = heading

            text_chunks = self._chunk_text(
                page_text, page_no, doc_hash, heading, chunk_index, embedder=embedder
            )
            all_chunks.extend(text_chunks)
            chunk_index += len(text_chunks)

            for table_item in tables:
                table_mds = self._table_extractor.table_to_markdown(table_item)
                for table_md in table_mds:
                    if not table_md.strip():
                        continue
                    token_count = _count_tokens(table_md)
                    prefix = f'[Page {page_no}, Section: {heading}]'
                    all_chunks.append({
                        'text': table_md,
                        'page_number': page_no,
                        'chunk_index': chunk_index,
                        'section_heading': heading,
                        'chunk_type': 'table',
                        'doc_hash': doc_hash,
                        'token_count': token_count,
                        'embedding_prefix': prefix,
                    })
                    chunk_index += 1

        logger.info(f'Produced {len(all_chunks)} chunks from {len(pages)} pages')
        return all_chunks
