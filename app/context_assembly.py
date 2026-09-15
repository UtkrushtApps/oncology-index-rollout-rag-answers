from __future__ import annotations

from app.models import Citation, Evidence


def build_context(evidence: list[Evidence]) -> tuple[str, list[Citation]]:
    blocks: list[str] = []
    citations: list[Citation] = []
    for i, ev in enumerate(evidence, start=1):
        label = f'[{i}] {ev.trial_id} {ev.amendment_id} {ev.section} {ev.chunk_id}'
        blocks.append(f'{label}\n{ev.text}')
        quote = ev.text[:220]
        citations.append(
            Citation(
                chunk_id=ev.chunk_id,
                document_id=ev.document_id,
                section=ev.section,
                quote=quote,
                char_start=ev.char_start,
                char_end=min(ev.char_end, ev.char_start + len(quote)),
            )
        )
    return '\n\n'.join(blocks), citations
