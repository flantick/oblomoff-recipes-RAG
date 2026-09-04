"""The end-to-end question -> answer pipeline: retrieval + generation (Step 5).

    from src.generation.answer import answer
    res = answer("как приготовить сочный стейк")

The same function is called by FastAPI in Step 6.
"""
from __future__ import annotations

from loguru import logger
from pydantic import ValidationError

from src.config import RETRIEVAL_MIN_SCORE, RETRIEVAL_PER_VIDEO, RETRIEVAL_TOP_VIDEOS
from src.generation.llm import LLMClient
from src.generation.prompts import build_messages
from src.generation.schemas import LLMRecipe, RecipeAnswer, SourceRef
from src.retrieval.retriever import Retriever, get_retriever


def _sources_from_citations(citations: list[dict]) -> list[SourceRef]:
    return [
        SourceRef(
            n=c["n"],
            video_id=c.get("video_id", ""),
            title=c.get("title", ""),
            url=c.get("url", ""),
            timecode=c.get("timecode", ""),
        )
        for c in citations
    ]


def answer(
    query: str,
    *,
    retriever: Retriever | None = None,
    llm: LLMClient | None = None,
    top_videos: int = RETRIEVAL_TOP_VIDEOS,
    per_video: int = RETRIEVAL_PER_VIDEO,
    use_intent_filter: bool = False,
    temperature: float | None = None,
) -> RecipeAnswer:
    retriever = retriever or get_retriever()
    llm = llm or LLMClient()

    rr = retriever.retrieve(
        query,
        top_videos=top_videos,
        per_video=per_video,
        use_intent_filter=use_intent_filter,
    )
    sources = _sources_from_citations(rr.citations)

    if not rr.citations:
        return RecipeAnswer(query=query, found=False, model=llm.model, used_reranker=rr.used_reranker)

    # Nothing used to stop an obviously irrelevant context from reaching the LLM.
    # The reranker's top score puts a floor under that: junk scores far below any
    # real query, so the call is skipped and ~25 s of generation saved.
    top_score = max((v.score for v in rr.videos), default=0.0)
    if top_score < RETRIEVAL_MIN_SCORE:
        logger.info("Слабое соответствие ({:.4f}) — пропускаю вызов LLM", top_score)
        return RecipeAnswer(query=query, found=False, sources=sources, model=llm.model,
                            used_reranker=rr.used_reranker)

    messages = build_messages(query, rr.context)
    kw = {} if temperature is None else {"temperature": temperature}
    obj = llm.chat_json(messages, **kw)

    try:
        rec = LLMRecipe.model_validate(obj)
    except ValidationError as exc:
        logger.warning("Ответ LLM не по схеме ({}), помечаю found=false", exc.errors()[:1])
        return RecipeAnswer(query=query, found=False, sources=sources, model=llm.model,
                            used_reranker=rr.used_reranker)

    by_n = {s.n: s for s in sources}
    primary = by_n.get(rec.source_n) if rec.source_n is not None else None
    if primary is None and rec.found:
        primary = sources[0]

    return RecipeAnswer(
        query=query,
        found=rec.found,
        dish=rec.dish,
        ingredients=rec.ingredients,
        steps=rec.steps,
        notes=rec.notes,
        source=primary,
        sources=sources,
        model=llm.model,
        used_reranker=rr.used_reranker,
    )
