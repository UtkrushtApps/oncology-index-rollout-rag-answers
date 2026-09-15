from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import get_settings
from app.models import Citation, QueryScope


class ProviderNotConfigured(RuntimeError):
    pass


def build_prompt(question: str, scope: QueryScope, context: str) -> str:
    return (
        'You answer clinical protocol questions using only the supplied passages. '
        'If the passages are not sufficient, say that the protocol evidence is insufficient. '
        'Include concise citations by chunk label.\n\n'
        f'Scope: {scope.model_dump(mode="json")}\n'
        f'Question: {question}\n\nPassages:\n{context}\n'
    )


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.2, max=2))
def generate_answer(question: str, scope: QueryScope, context: str, citations: list[Citation]) -> tuple[str, str]:
    settings = get_settings()
    prompt = build_prompt(question, scope, context)
    if settings.llm_provider.lower() == 'anthropic':
        if not settings.anthropic_api_key:
            raise ProviderNotConfigured('Anthropic provider key is not configured')
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        msg = client.messages.create(
            model=settings.llm_model,
            max_tokens=500,
            temperature=0,
            messages=[{'role': 'user', 'content': prompt}],
        )
        text = ''.join(block.text for block in msg.content if getattr(block, 'type', '') == 'text')
        return text, settings.llm_model
    if not settings.openai_api_key:
        raise ProviderNotConfigured('OpenAI provider key is not configured')
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return resp.choices[0].message.content or '', settings.llm_model
