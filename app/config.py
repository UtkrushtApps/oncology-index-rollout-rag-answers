from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='/root/task/.env', extra='ignore')

    database_url: str = 'postgresql://raguser:ragpass@localhost:5432/ragdb'
    qdrant_url: str = 'http://localhost:6333'
    redis_url: str = 'redis://localhost:6379/0'
    qdrant_collection: str = 'oncology_protocol_chunks'
    corpus_id: str = 'oncology_phase3_protocols_v1'
    eval_dataset_id: str = 'qa_protocol_eval_v1'
    llm_provider: str = 'openai'
    llm_model: str = 'gpt-4o-mini'
    openai_api_key: str = ''
    anthropic_api_key: str = ''
    embedding_model: str = 'BAAI/bge-small-en-v1.5'
    embedding_version: str = 'emb-v1'
    active_manifest_id: str = 'blue-2024-06'


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
