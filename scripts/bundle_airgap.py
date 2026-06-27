# scripts/bundle_airgap.py
import os
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BUNDLE_DIR   = PROJECT_ROOT / 'dist'

SERVICES = ['vllm', 'backend', 'frontend']

INCLUDE_FILES = [
    'docker-compose.yml',
    'Dockerfile.vllm',
    'Dockerfile.backend',
    'Dockerfile.frontend',
    '.env.example',
    'requirements.txt',
    'CLAUDE.md',
    'ARCHITECTURE.md',
    'README.md',
]

INCLUDE_DIRS = [
    'scripts',
    'src',
    'frontend/src',
    'frontend/package.json',
    'frontend/vite.config.js',
    'frontend/tailwind.config.js',
]


def run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    print(f'  $ {" ".join(str(c) for c in cmd)}')
    return subprocess.run(cmd, check=check, capture_output=False)


def check_models():
    models_dir = PROJECT_ROOT / 'models'
    required = [
        'Qwen3-14B-AWQ',
        'bge-m3',
        'bge-reranker-v2-m3',
        'nli-deberta-v3-small',
        'docling-models',
    ]
    print('Checking model weights...')
    missing = []
    for m in required:
        p = models_dir / m
        if not p.exists():
            missing.append(m)
        else:
            size_gb = sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e9
            print(f'  {m}: {size_gb:.1f} GB')
    if missing:
        print(f'ERROR: missing models: {missing}')
        print('Run python scripts/download_models.py first.')
        sys.exit(1)
    print('All models present.')


def save_docker_images(out_dir: Path):
    images_dir = out_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    for svc in SERVICES:
        image_tag = f'bis-{svc}:latest'
        out_file   = images_dir / f'{svc}.tar'
        print(f'Saving Docker image {image_tag} -> {out_file.name}')
        run(['docker', 'save', '-o', str(out_file), image_tag])
        size_gb = out_file.stat().st_size / 1e9
        print(f'  Saved: {size_gb:.2f} GB')


def copy_source_files(out_dir: Path):
    import shutil
    src_dir = out_dir / 'source'
    src_dir.mkdir(parents=True, exist_ok=True)
    for rel in INCLUDE_FILES:
        src = PROJECT_ROOT / rel
        if src.exists():
            dst = src_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for rel in INCLUDE_DIRS:
        src = PROJECT_ROOT / rel
        if src.exists():
            dst = src_dir / rel
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    print(f'Source files copied to {src_dir}')


def write_load_script(out_dir: Path):
    script = out_dir / 'load_and_run.sh'
    script.write_text(
        '#!/usr/bin/env bash\n'
        '# Load Docker images and start the Board Intelligence System.\n'
        '# Run this script on the air-gapped target machine.\n'
        'set -euo pipefail\n'
        '\n'
        'echo "Loading Docker images..."\n'
        'for img in images/*.tar; do\n'
        '  echo "  Loading $img"\n'
        '  docker load -i "$img"\n'
        'done\n'
        '\n'
        'echo "Copying source..."\n'
        'cp -r source/* .\n'
        '\n'
        'echo "Setting up environment..."\n'
        'if [ ! -f .env ]; then\n'
        '  cp .env.example .env\n'
        '  echo "IMPORTANT: Edit .env and set SESSION_SECRET_KEY before starting."\n'
        'fi\n'
        '\n'
        'echo "Starting services..."\n'
        'docker compose up -d\n'
        'echo "Board Intelligence System started. Open http://localhost:5173"\n'
    )
    script.chmod(0o755)
    print(f'Load script written: {script.name}')


def create_tarball(out_dir: Path, bundle_path: Path):
    print(f'Creating tarball: {bundle_path.name}')
    with tarfile.open(bundle_path, 'w:gz') as tar:
        tar.add(out_dir, arcname='board-intelligence-bundle')
    size_gb = bundle_path.stat().st_size / 1e9
    print(f'Bundle ready: {bundle_path} ({size_gb:.2f} GB)')


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir     = BUNDLE_DIR / f'bundle_{ts}'
    bundle_path = BUNDLE_DIR / f'board-intelligence-{ts}.tar.gz'

    print('Board Intelligence -- Air-Gap Bundle Script')
    print('=' * 50)
    print(f'Output: {bundle_path}')
    print('')

    check_models()
    print('')

    print('Building Docker images...')
    run(['docker', 'compose', 'build'])
    print('')

    out_dir.mkdir(parents=True, exist_ok=True)
    save_docker_images(out_dir)
    print('')

    copy_source_files(out_dir)
    write_load_script(out_dir)
    print('')

    create_tarball(out_dir, bundle_path)
    print('')
    print('Done. Transfer the .tar.gz to the target machine and run:')
    print('  tar -xzf board-intelligence-*.tar.gz')
    print('  cd board-intelligence-bundle')
    print('  ./load_and_run.sh')


if __name__ == '__main__':
    main()
