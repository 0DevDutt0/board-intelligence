# scripts/download_models.py
# Downloads all required models to the /models/ directory.
# Run once before first startup.
# Requires: huggingface_hub, internet connection (one-time only)
# Usage: python3 scripts/download_models.py

import os
import sys
import hashlib
from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent / 'models'

MODELS = [
    {
        'name': 'Qwen3-14B-AWQ',
        'repo': 'Qwen/Qwen3-14B-AWQ',
        'local_dir': MODELS_DIR / 'Qwen3-14B-AWQ',
        'description': 'Main LLM -- Qwen3 14B AWQ quantized for vLLM Marlin',
        'size_hint': '~9.5 GB',
    },
    {
        'name': 'BGE-M3',
        'repo': 'BAAI/bge-m3',
        'local_dir': MODELS_DIR / 'bge-m3',
        'description': 'Embedding model -- dense + sparse + multi-vector',
        'size_hint': '~1.1 GB',
    },
    {
        'name': 'BGE-Reranker-v2-m3',
        'repo': 'BAAI/bge-reranker-v2-m3',
        'local_dir': MODELS_DIR / 'bge-reranker-v2-m3',
        'description': 'Cross-encoder reranker',
        'size_hint': '~0.6 GB',
    },
    {
        'name': 'NLI-DeBERTa-v3-small',
        'repo': 'cross-encoder/nli-deberta-v3-small',
        'local_dir': MODELS_DIR / 'nli-deberta-v3-small',
        'description': 'Hallucination guard NLI model',
        'size_hint': '~180 MB',
    },
]

DOCLING_MODELS = {
    'name': 'Docling layout models',
    'local_dir': MODELS_DIR / 'docling-models',
    'description': 'DocLayNet + TableFormer weights for offline PDF parsing',
}


def download_hf_model(repo_id, local_dir, ignore_patterns=None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print('ERROR: huggingface_hub not installed.')
        print('Run: pip install huggingface_hub')
        sys.exit(1)

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f'  Downloading {repo_id} -> {local_dir}')
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=ignore_patterns or ['*.msgpack', '*.h5', 'flax_model*', 'tf_model*'],
    )
    print(f'  Done: {local_dir}')


def download_docling_models(local_dir):
    try:
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except ImportError:
        print('  WARNING: docling not installed, skipping docling model download')
        print('  Install with: pip install docling')
        return

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f'  Downloading Docling models (DocLayNet + TableFormer) -> {local_dir}')
    pipeline_options = PdfPipelineOptions(artifacts_path=str(local_dir))
    try:
        _ = StandardPdfPipeline(pipeline_options=pipeline_options)
        print(f'  Done: {local_dir}')
    except Exception as e:
        print(f'  WARNING: Docling model download encountered an error: {e}')
        print('  Models may download on first run instead.')


def check_disk_space(required_gb=30):
    import shutil
    free = shutil.disk_usage(MODELS_DIR.parent).free / (1024 ** 3)
    if free < required_gb:
        print(f'WARNING: Only {free:.1f} GB free, need ~{required_gb} GB')
        print('Proceeding anyway -- ensure sufficient space before running.')
    else:
        print(f'Disk space: {free:.1f} GB free (need ~{required_gb} GB) -- OK')


def main():
    print('\nBoard Intelligence System -- Model Downloader')
    print('=' * 60)
    print(f'Models directory: {MODELS_DIR}')
    print('')

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    check_disk_space()
    print('')

    total = len(MODELS) + 1
    for i, model in enumerate(MODELS, 1):
        already_exists = Path(model['local_dir']).exists() and \
            any(Path(model['local_dir']).iterdir())

        print(f'[{i}/{total}] {model["name"]} ({model["size_hint"]})')
        print(f'  {model["description"]}')

        if already_exists:
            print(f'  SKIPPING -- already exists at {model["local_dir"]}')
            print(f'  Delete the directory to re-download.')
        else:
            download_hf_model(model['repo'], model['local_dir'])
        print('')

    print(f'[{total}/{total}] {DOCLING_MODELS["name"]}')
    print(f'  {DOCLING_MODELS["description"]}')
    already_exists = Path(DOCLING_MODELS['local_dir']).exists() and \
        any(Path(DOCLING_MODELS['local_dir']).iterdir())
    if already_exists:
        print(f'  SKIPPING -- already exists')
    else:
        download_docling_models(DOCLING_MODELS['local_dir'])
    print('')

    print('=' * 60)
    print('Download complete. Verify with:')
    print('  python3 scripts/verify_gpu.py')
    print('')
    print('Then start the system:')
    print('  scripts\\start_all.ps1')
    print('')


if __name__ == '__main__':
    main()
