"""FastAPI backend of the recipe RAG system (Step 6).

Endpoints:
    GET  /health          — status of Qdrant and vLLM
    GET  /search?q=...     — raw retrieval (no LLM), fast
    POST /ask              — question -> a structured recipe (RecipeAnswer)

The heavy models (bge-m3, reranker) are loaded once in the lifespan.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from src.config import (
    LLM_MODEL,
    QDRANT_COLLECTION,
    RETRIEVAL_PER_VIDEO,
    RETRIEVAL_TOP_VIDEOS,
)
from src.generation.answer import answer
from src.generation.llm import LLMClient
from src.generation.schemas import RecipeAnswer
from src.retrieval.retriever import Retriever

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Инициализация retriever (bge-m3 + reranker)…")
    STATE["retriever"] = Retriever()          # EMBED_DEVICE / RERANK_DEVICE come from the environment
    STATE["llm"] = LLMClient()
    logger.info("API готов")
    yield
    STATE.clear()


app = FastAPI(title="oblomoff RAG", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    # the defaults come from config, otherwise the retriever settings are dead:
    # hardcoded 3/2 used to sit here and overrode RETRIEVAL_* on every call
    top_videos: int = Field(default=RETRIEVAL_TOP_VIDEOS, ge=1, le=8)
    per_video: int = Field(default=RETRIEVAL_PER_VIDEO, ge=1, le=12)
    use_intent_filter: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=1.5)


class SearchHit(BaseModel):
    title: str
    url: str
    timecode: str
    section: str
    score: float
    text: str


class HealthResponse(BaseModel):
    status: str
    qdrant: dict
    llm: dict


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    retriever: Retriever = STATE["retriever"]
    llm: LLMClient = STATE["llm"]

    qdrant_ok, points, qerr = True, None, None
    try:
        points = retriever.store.count()
    except Exception as exc:  # noqa: BLE001
        qdrant_ok, qerr = False, str(exc)

    llm_ok = llm.health()
    status = "ok" if (qdrant_ok and points and llm_ok) else "degraded"
    return HealthResponse(
        status=status,
        qdrant={"ok": qdrant_ok, "collection": QDRANT_COLLECTION, "points": points, "error": qerr},
        llm={"ok": llm_ok, "model": LLM_MODEL},
    )


@app.get("/search", response_model=list[SearchHit])
def search(
    q: str = Query(min_length=2, max_length=500),
    k: int = Query(default=5, ge=1, le=20),
    mode: str = Query(default="hybrid", pattern="^(hybrid|dense|sparse)$"),
) -> list[SearchHit]:
    retriever: Retriever = STATE["retriever"]
    emb = retriever.embedder.encode_queries([q])[0]
    points = retriever.store.search(emb, k=k, mode=mode)
    out: list[SearchHit] = []
    for p in points:
        pl = p.payload or {}
        out.append(
            SearchHit(
                title=pl.get("title", ""),
                url=pl.get("url", ""),
                timecode=pl.get("timecode", ""),
                section=pl.get("section", ""),
                score=float(p.score),
                text=pl.get("text", ""),
            )
        )
    return out


@app.post("/ask", response_model=RecipeAnswer)
def ask(req: AskRequest) -> RecipeAnswer:
    retriever: Retriever = STATE["retriever"]
    llm: LLMClient = STATE["llm"]
    t0 = time.time()
    try:
        res = answer(
            req.query,
            retriever=retriever,
            llm=llm,
            top_videos=req.top_videos,
            per_video=req.per_video,
            use_intent_filter=req.use_intent_filter,
            temperature=req.temperature,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка в /ask")
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc
    logger.info("/ask '{}' -> found={} за {:.1f}s", req.query, res.found, time.time() - t0)
    return res


@app.get("/")
def root() -> dict:
    return {"service": "oblomoff RAG", "docs": "/docs", "health": "/health"}
