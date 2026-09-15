from functools import lru_cache
from fastembed import TextEmbedding
from app.config import get_settings


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=get_settings().embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [list(v) for v in _model().embed(texts)]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
