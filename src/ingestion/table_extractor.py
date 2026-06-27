# src/ingestion/table_extractor.py
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_APPROX_CHARS_PER_TOKEN = 4


def _count_tokens_approx(text: str) -> int:
    return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)


class TableExtractor:
    def __init__(self, table_max_tokens: int = None):
        self._max_tokens = table_max_tokens or settings.table_max_tokens

    def table_to_markdown(self, table_item) -> list:
        try:
            md = table_item.export_to_markdown()
        except Exception:
            try:
                md = str(table_item)
            except Exception:
                return []

        if _count_tokens_approx(md) <= self._max_tokens:
            return [md]

        return self._split_table_by_rows(table_item)

    def _split_table_by_rows(self, table_item) -> list:
        try:
            df = table_item.export_to_dataframe()
        except Exception:
            return [str(table_item)]

        if df is None or df.empty:
            return []

        header = '| ' + ' | '.join(str(c) for c in df.columns) + ' |'
        separator = '| ' + ' | '.join('---' for _ in df.columns) + ' |'

        chunks = []
        current_rows = []
        current_tokens = _count_tokens_approx(header + '\n' + separator)

        for _, row in df.iterrows():
            row_md = '| ' + ' | '.join(str(v) for v in row.values) + ' |'
            row_tokens = _count_tokens_approx(row_md)

            if current_tokens + row_tokens > self._max_tokens and current_rows:
                chunk_md = '\n'.join([header, separator] + current_rows)
                chunks.append(chunk_md)
                current_rows = []
                current_tokens = _count_tokens_approx(header + '\n' + separator)

            current_rows.append(row_md)
            current_tokens += row_tokens

        if current_rows:
            chunk_md = '\n'.join([header, separator] + current_rows)
            chunks.append(chunk_md)

        return chunks
