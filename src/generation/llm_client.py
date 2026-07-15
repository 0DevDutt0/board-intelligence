# src/generation/llm_client.py
from typing import AsyncIterator, Optional
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self, base_url: str = None, model: str = None):
        import openai
        self._base_url = base_url or settings.vllm_base_url
        self._model = model or settings.vllm_model
        self._client = openai.AsyncOpenAI(
            base_url=self._base_url + '/v1',
            api_key='not-required',
            timeout=settings.vllm_timeout_seconds,
        )
        logger.info(f'LLMClient pointed at {self._base_url}, model={self._model}')

    async def stream_completion(
        self,
        messages: list,
        max_tokens: int = None,
        temperature: float = None,
    ) -> AsyncIterator[str]:
        mt = max_tokens or settings.vllm_max_tokens
        temp = temperature if temperature is not None else settings.vllm_temperature

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=mt,
            temperature=temp,
            stream=True,
        )

        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            delta = choice.delta
            if delta and delta.content:
                yield delta.content
            if choice.finish_reason == 'stop':
                break

    async def complete(
        self,
        messages: list,
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        tokens = []
        async for token in self.stream_completion(messages, max_tokens, temperature):
            tokens.append(token)
        return ''.join(tokens)

    async def health_check(self) -> bool:
        try:
            models = await self._client.models.list()
            return len(models.data) > 0
        except Exception:
            return False
