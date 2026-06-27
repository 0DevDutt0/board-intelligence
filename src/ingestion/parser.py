# src/ingestion/parser.py
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Pages with fewer than this many chars on average are treated as scanned
_SCANNED_CHAR_THRESHOLD = 50


def _detect_render_mode(pdf_path: str) -> str:
    import pymupdf
    doc = pymupdf.open(pdf_path)
    total_chars = sum(len(p.get_text()) for p in doc)
    avg = total_chars / max(len(doc), 1)
    doc.close()
    return 'native' if avg >= _SCANNED_CHAR_THRESHOLD else 'scanned'


def _pymupdf_plain(pdf_path: str) -> list:
    '''Guaranteed fallback: extract raw text page-by-page with pymupdf.'''
    import pymupdf
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        pages.append({
            'page_number': i + 1,
            'text': page.get_text() or '',
            'render_mode': 'pymupdf_plain',
            'tables': [],
        })
    doc.close()
    return pages


def _parse_with_pymupdf(pdf_path: str) -> list:
    '''Try pymupdf4llm markdown extraction; fall back to plain text.'''
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(pdf_path)
    page_count = len(doc)
    doc.close()

    pages = []
    try:
        md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        if not md_pages:
            raise ValueError('pymupdf4llm returned empty result')
        for i, chunk in enumerate(md_pages):
            text = chunk.get('text', '') if isinstance(chunk, dict) else str(chunk or '')
            pages.append({
                'page_number': i + 1,
                'text': text,
                'render_mode': 'pymupdf4llm',
                'tables': [],
            })
    except Exception as exc:
        logger.warning(f'pymupdf4llm failed ({exc}), using plain page.get_text()')
        pages = _pymupdf_plain(pdf_path)

    # Pad any missing pages (can happen when pymupdf4llm skips blank pages)
    seen = {p['page_number'] for p in pages}
    for page_no in range(1, page_count + 1):
        if page_no not in seen:
            pages.append({
                'page_number': page_no,
                'text': '',
                'render_mode': 'pymupdf_plain',
                'tables': [],
            })

    pages.sort(key=lambda p: p['page_number'])
    return pages


def _docling_artifacts_valid(artifacts_path: str) -> bool:
    '''Return True only when the path exists and contains Docling model files.'''
    p = Path(artifacts_path)
    if not p.is_dir():
        return False
    # Docling models directory must contain at least one subdirectory with weights
    children = list(p.iterdir())
    return len(children) > 0


def _parse_with_docling(pdf_path: str, artifacts_path: str) -> list:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    # Only set artifacts_path when the directory is genuinely populated;
    # otherwise Docling will attempt to download models at runtime.
    if _docling_artifacts_valid(artifacts_path):
        pipeline_options.artifacts_path = artifacts_path
    else:
        logger.warning(
            f'Docling artifacts path {artifacts_path!r} missing or empty -- '
            'Docling will attempt to download models at runtime.'
        )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(pdf_path)
    doc = result.document

    import pymupdf
    mupdf_doc = pymupdf.open(pdf_path)
    page_count = len(mupdf_doc)
    mupdf_doc.close()

    page_texts = {}
    page_tables = {}

    for item, _ in doc.iterate_items():
        try:
            from docling.datamodel.document import TextItem, TableItem
            page_no = item.prov[0].page_no if item.prov else 1
            if isinstance(item, TableItem):
                page_tables.setdefault(page_no, []).append(item)
            elif isinstance(item, TextItem):
                page_texts.setdefault(page_no, []).append(item.text)
        except Exception:
            continue

    pages = []
    for page_no in range(1, page_count + 1):
        pages.append({
            'page_number': page_no,
            'text': '\n'.join(page_texts.get(page_no, [])),
            'render_mode': 'docling',
            'tables': page_tables.get(page_no, []),
        })

    return pages


class DocumentParser:
    def __init__(self, docling_artifacts: str):
        self._artifacts = docling_artifacts

    def parse(self, pdf_path: str) -> list:
        render_mode = _detect_render_mode(pdf_path)
        logger.info(f'Render mode: {render_mode} for {pdf_path}')

        # Native PDF: PyMuPDF is fast and accurate -- Docling adds no value here.
        # _parse_with_pymupdf is now bulletproof (inner plain fallback).
        if render_mode == 'native':
            pages = _parse_with_pymupdf(pdf_path)
            logger.info(f'Native parse: {len(pages)} pages')
            return pages

        # Scanned PDF: try Docling (OCR) first, then PyMuPDF plain as last resort.
        try:
            pages = _parse_with_docling(pdf_path, self._artifacts)
            logger.info(f'Docling parse: {len(pages)} pages')
            return pages
        except Exception as exc:
            logger.warning(f'Docling failed ({exc}), falling back to PyMuPDF')
            pages = _pymupdf_plain(pdf_path)
            logger.info(f'PyMuPDF plain fallback: {len(pages)} pages')
            return pages
