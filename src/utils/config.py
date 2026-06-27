# src/utils/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
    )

    # vLLM
    vllm_base_url: str = Field(default='http://vllm:11436')
    vllm_model: str = Field(default='qwen3-14b')
    vllm_max_tokens: int = Field(default=4096)
    vllm_temperature: float = Field(default=0.1)
    vllm_timeout_seconds: int = Field(default=300)

    # Model paths
    embedding_model: str = Field(default='/models/bge-m3')
    reranker_model: str = Field(default='/models/bge-reranker-v2-m3')
    nli_model: str = Field(default='/models/nli-deberta-v3-small')
    docling_artifacts: str = Field(default='/models/docling-models')

    # Hardware
    device: str = Field(default='cuda')
    embedding_batch_size: int = Field(default=64)
    embedding_dtype: str = Field(default='float16')

    # Retrieval
    dense_top_k: int = Field(default=50)
    sparse_top_k: int = Field(default=50)
    rrf_k: int = Field(default=60)
    reranker_top_k: int = Field(default=12)
    reranker_min_score: float = Field(default=0.10)
    bm25_k1: float = Field(default=1.2)
    bm25_b: float = Field(default=0.5)
    use_hyde: bool = Field(default=True)
    hyde_min_query_words: int = Field(default=6)
    hyde_concat_query: bool = Field(default=True)

    # Chunking
    chunk_target_tokens: int = Field(default=500)
    chunk_min_tokens: int = Field(default=100)
    chunk_overlap_tokens: int = Field(default=50)
    table_max_tokens: int = Field(default=1024)

    # Hallucination guard
    nli_entailment_threshold: float = Field(default=0.70)
    numeric_verification: bool = Field(default=True)

    # Session
    session_secret_key: str = Field(default='change-me-in-production')
    session_timeout_minutes: int = Field(default=30)
    max_conversation_history: int = Field(default=3)
    temp_session_dir: str = Field(default='/tmp/bis-sessions')
    max_upload_mb: int = Field(default=100)

    # API
    api_host: str = Field(default='127.0.0.1')
    api_port: int = Field(default=8000)
    log_level: str = Field(default='INFO')
    log_queries: bool = Field(default=False)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def temp_session_path(self) -> Path:
        return Path(self.temp_session_dir)


settings = Settings()
