# src/api/routes/health.py
from fastapi import APIRouter
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get('/health')
async def health():
    status = {
        'status': 'ok',
        'vllm_ready': False,
        'models_loaded': False,
        'gpu_memory_used_mb': 0,
    }

    try:
        from src.api.main import app_state
        status['models_loaded'] = app_state.get('models_loaded', False)
    except Exception:
        pass

    try:
        import httpx
        from src.utils.config import settings
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(settings.vllm_base_url + '/health')
            status['vllm_ready'] = resp.status_code == 200
    except Exception:
        status['vllm_ready'] = False

    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(0)
            status['gpu_memory_used_mb'] = mem // (1024 * 1024)
    except Exception:
        pass

    return status
