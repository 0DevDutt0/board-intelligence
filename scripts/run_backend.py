# scripts/run_backend.py
# Native Windows entrypoint for the FastAPI backend.
#
# Why this exists instead of calling uvicorn directly:
# On this Windows venv (conda-forge Python 3.13), importing pyarrow AFTER the
# app's import graph (asyncio, cryptography, pydantic, fastapi) is loaded
# causes a hard access violation inside pyarrow's native module (exit code
# 0xC0000005). pyarrow is pulled in transitively by FlagEmbedding -> datasets
# -> pandas at model-load time, which is after uvicorn has already imported
# asyncio -- so 'python -m uvicorn src.api.main:app' always crashes at startup.
# Importing pyarrow first, before anything else, reliably avoids the conflict.

import pyarrow  # must stay the first import in this file

import os
import sys

# Make the repo root importable and the CWD, regardless of where the script
# is invoked from ('src.api.main' import and .env discovery both need it).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import uvicorn

from src.utils.config import settings


def main():
    uvicorn.run(
        'src.api.main:app',
        host=settings.api_host,
        port=settings.api_port,
        workers=1,
        log_level=settings.log_level.lower(),
    )


if __name__ == '__main__':
    main()
