"""Generate PDFs from the fictional minutes used by the demo and evaluation."""

import argparse
import json
from pathlib import Path

import pymupdf

CORPUS = Path(__file__).with_name('board_packs.json')


def load_board_packs() -> dict:
    return json.loads(CORPUS.read_text(encoding='utf-8'))


def build_pdf(pack: dict) -> bytes:
    with pymupdf.open() as document:
        for number, item in enumerate(pack['pages'], 1):
            page = document.new_page()
            page.insert_text((54, 68), 'FICTIONAL BOARD MATERIAL', fontsize=10, color=(0.16, 0.32, 0.60))
            page.insert_text((54, 103), pack['title'], fontname='hebo', fontsize=21)
            page.draw_line((54, 123), (541, 123), color=(0.73, 0.78, 0.85), width=0.7)
            page.insert_text((54, 157), item['heading'], fontname='hebo', fontsize=15)
            for index, line in enumerate(item['lines']):
                page.insert_text((54, 195 + 30 * index), line, fontsize=11)
            page.insert_text((54, 783), f'Page {number} of {len(pack["pages"])}', fontsize=9)
        document.set_metadata({'title': pack['title'], 'author': 'Board Intelligence examples'})
        return document.tobytes(no_new_id=True, deflate=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=CORPUS.parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, pack in load_board_packs().items():
        output = 'board-minutes.pdf' if name == 'harbor-works' else f'{name}.pdf'
        (args.output_dir / output).write_bytes(build_pdf(pack))


if __name__ == '__main__':
    main()
