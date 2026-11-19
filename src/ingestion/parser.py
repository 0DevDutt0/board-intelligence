# src/ingestion/parser.py
from pathlib import Path
from io import BytesIO
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Pages with fewer than this many chars on average are treated as scanned
_SCANNED_CHAR_THRESHOLD = 50


def _open_pdf(source: str | bytes):
    import pymupdf
    if isinstance(source, bytes):
        return pymupdf.open(stream=source, filetype='pdf')
    return pymupdf.open(source)


def _detect_render_mode(source: str | bytes) -> str:
    with _open_pdf(source) as doc:
        total_chars = sum(len(p.get_text()) for p in doc)
        avg = total_chars / max(len(doc), 1)
    return 'native' if avg >= _SCANNED_CHAR_THRESHOLD else 'scanned'


def _pymupdf_plain(source: str | bytes) -> list:
    '''Guaranteed fallback: extract raw text page-by-page with pymupdf.'''
    with _open_pdf(source) as doc:
        return [
            {
                'page_number': i + 1,
                'text': page.get_text() or '',
                'render_mode': 'pymupdf_plain',
                'tables': [],
            }
            for i, page in enumerate(doc)
        ]


def _parse_with_pymupdf(source: str | bytes) -> list:
    '''Try pymupdf4llm markdown extraction; fall back to plain text.'''
    import pymupdf4llm

    try:
        with _open_pdf(source) as doc:
            md_pages = pymupdf4llm.to_markdown(doc, page_chunks=True, use_ocr=False)
            if len(md_pages) != len(doc):
                return _pymupdf_plain(source)
        return [
            {
                'page_number': page_num,
                'text': chunk['text'],
                'render_mode': 'pymupdf4llm',
                'tables': [],
            }
            for page_num, chunk in enumerate(md_pages, start=1)
        ]
    except Exception:
        logger.warning('Layout extraction failed, using plain PDF text')
        return _pymupdf_plain(source)


def _docling_artifacts_valid(artifacts_path: str) -> bool:
    '''Return True only when the path exists and contains Docling model files.'''
    p = Path(artifacts_path)
    if not p.is_dir():
        return False
    # Docling models directory must contain at least one subdirectory with weights
    children = list(p.iterdir())
    return len(children) > 0


def _parse_with_docling(source: str | bytes, artifacts_path: str) -> list:
    if not _docling_artifacts_valid(artifacts_path):
        raise FileNotFoundError('Docling artifacts are missing. Pre-stage them before processing scans.')

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.datamodel.base_models import DocumentStream, InputFormat

    pipeline_options = PdfPipelineOptions(
        artifacts_path=artifacts_path,
        enable_remote_services=False,
        allow_external_plugins=False,
        do_ocr=True,
        ocr_options=RapidOcrOptions(backend='onnxruntime'),
        do_table_structure=True,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    document_input = (
        DocumentStream(name='document.pdf', stream=BytesIO(source))
        if isinstance(source, bytes) else source
    )
    result = converter.convert(document_input)
    with _open_pdf(source) as doc:
        page_count = len(doc)

    doc = result.document
    from docling_core.types.doc import TextItem, TableItem

    page_texts = {}
    page_tables = {}

    for item, _ in doc.iterate_items():
        if not isinstance(item, (TextItem, TableItem)) or not item.prov:
            continue
        page_no = item.prov[0].page_no
        if isinstance(item, TableItem):
            page_tables.setdefault(page_no, []).append(item)
        else:
            page_texts.setdefault(page_no, []).append(item.text)

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

    def parse(self, source: str | bytes) -> list:
        render_mode = _detect_render_mode(source)
        logger.info('Render mode: %s', render_mode)

        # Native PDF: PyMuPDF is fast and accurate -- Docling adds no value here.
        # _parse_with_pymupdf is now bulletproof (inner plain fallback).
        if render_mode == 'native':
            pages = _parse_with_pymupdf(source)
            logger.info(f'Native parse: {len(pages)} pages')
            return pages

        # Scanned PDF: try Docling (OCR) first, then PyMuPDF plain as last resort.
        try:
            pages = _parse_with_docling(source, self._artifacts)
            logger.info(f'Docling parse: {len(pages)} pages')
            return pages
        except Exception:
            logger.warning('Local OCR unavailable or failed, falling back to embedded PDF text')
            pages = _pymupdf_plain(source)
            logger.info(f'PyMuPDF plain fallback: {len(pages)} pages')
            return pages
