"""Shared fakes and data factories for the whole test suite.

Project rule: the outside world is replaced by small explicit
fakes injected through the seams the code already has, not by MagicMocks. Only
fakes and factories needed by more than one test package live here; fixtures
used by a single module belong in that module's test file.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest
from loguru import logger

from src.index.embedder import Embedding


# --- resetting global state ------------------------------------------
@pytest.fixture(autouse=True)
def _restore_loguru_sink():
    """Puts a LIVE loguru sink back after a test that reconfigured logging.

    Every CLI entry point in this project opens with logger.remove(), which
    stops the process-wide handlers. Stashing the handler dict and putting it
    back does NOT undo that: the objects restored are the same ones that were
    already stopped, so from then on the whole suite logs into the void and any
    later test that inspects a warning silently depends on file order.

    So the registry is rebuilt rather than replayed, and only when a test
    actually disturbed it — loguru's own default is a plain stderr sink.
    """
    core = logger._core
    saved = dict(core.handlers)
    yield
    disturbed = core.handlers != saved or any(
        getattr(h, "_stopped", False) for h in core.handlers.values()
    )
    if disturbed:
        logger.remove()
        logger.add(sys.stderr)
@pytest.fixture(autouse=True)
def _reset_retriever_singleton():
    """get_retriever() caches a Retriever in a module-level variable — without
    a reset the tests would start depending on execution order."""
    import src.retrieval.retriever as mod

    saved = mod._DEFAULT
    mod._DEFAULT = None
    yield
    mod._DEFAULT = saved


# --- data factories ---------------------------------------------------
def make_payload(
    *,
    chunk_id: str | None = None,
    video_id: str = "vid1",
    chunk_index: int = 1,
    n_chunks: int = 5,
    title: str = "Как приготовить стейк",
    text: str = "Обжарьте стейк на сильном огне.",
    section: str = "steps",
    start: float = 60.0,
    end: float = 120.0,
    timecode: str = "01:00",
    url: str | None = None,
    **extra: Any,
) -> dict:
    """A chunk payload shaped the way it is stored in Qdrant."""
    payload = {
        "chunk_id": chunk_id or f"{video_id}::{chunk_index:03d}",
        "video_id": video_id,
        "title": title,
        "url": url or f"https://youtu.be/{video_id}?t={int(start)}",
        "timecode": timecode,
        "start": start,
        "end": end,
        "chunk_index": chunk_index,
        "n_chunks": n_chunks,
        "section": section,
        "text": text,
    }
    payload.update(extra)
    return payload


@dataclass
class FakePoint:
    """Stands in for qdrant_client.models.ScoredPoint: only id/score/payload
    are ever read."""
    id: str
    score: float
    payload: dict | None


def make_point(score: float = 0.9, **payload_kw: Any) -> FakePoint:
    payload = make_payload(**payload_kw)
    return FakePoint(id=payload["chunk_id"], score=score, payload=payload)


# --- fakes for the outside world -------------------------------------
@dataclass
class FakeStore:
    """Stands in for src.index.store.VectorStore.

    points      — what search() returns, in the given order;
    corpus      — payloads fetch_chunks() picks the neighbours from;
    fetch_error — the exception fetch_chunks() raises (the except branch).
    """
    points: list[FakePoint] = field(default_factory=list)
    corpus: list[dict] = field(default_factory=list)
    count_value: int = 0
    fetch_error: Exception | None = None
    search_calls: list[dict] = field(default_factory=list)
    fetch_calls: list[tuple[str, list[int]]] = field(default_factory=list)

    def search(self, emb, *, k: int = 5, mode: str = "hybrid",
               query_filter=None, overfetch: int = 5) -> list[FakePoint]:
        self.search_calls.append(
            {"emb": emb, "k": k, "mode": mode, "query_filter": query_filter,
             "overfetch": overfetch}
        )
        return self.points[:k]

    def fetch_chunks(self, video_id: str, indices: list[int]) -> list[dict]:
        self.fetch_calls.append((video_id, list(indices)))
        if self.fetch_error is not None:
            raise self.fetch_error
        wanted = set(indices)
        return [
            p for p in self.corpus
            if p.get("video_id") == video_id and p.get("chunk_index") in wanted
        ]

    def count(self) -> int:
        return self.count_value


@dataclass
class FakeEmbedder:
    """Stands in for src.index.embedder.BGEM3Embedder."""
    dense: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    sparse_indices: list[int] = field(default_factory=lambda: [7, 42])
    sparse_values: list[float] = field(default_factory=lambda: [0.5, 0.25])
    encoded: list[list[str]] = field(default_factory=list)

    def _encode(self, texts: list[str]) -> list[Embedding]:
        self.encoded.append(list(texts))
        return [
            Embedding(dense=list(self.dense),
                      sparse_indices=list(self.sparse_indices),
                      sparse_values=list(self.sparse_values))
            for _ in texts
        ]

    encode_queries = _encode
    encode_passages = _encode


@dataclass
class FakeReranker:
    """Stands in for src.retrieval.rerank.Reranker.

    scores — either a ready list of scores matching the order of the texts, or
    a callable (query, text) -> float. By default the scores decrease with
    position, which makes the reranker's order differ from the search order.
    """
    scores: list[float] | Callable[[str, str], float] | None = None
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        if not texts:
            return []
        if callable(self.scores):
            return [float(self.scores(query, t)) for t in texts]
        if self.scores is not None:
            return [float(x) for x in self.scores[:len(texts)]]
        return [round(1.0 - i / max(len(texts), 1), 4) for i in range(len(texts))]


@dataclass
class FakeLLM:
    """Stands in for src.generation.llm.LLMClient.

    payload      — what chat_json() returns;
    error        — the exception chat_json() raises;
    fail_on_call — calling chat_json() is itself a test failure (for the
                   branches where the LLM must NOT be reached);
    calls        — one (messages, kwargs) pair per chat_json() call, so that a
                   test can check both the prompt and what was forwarded
                   alongside it (temperature and the like).
    """
    model: str = "fake-model"
    payload: dict | None = None
    error: Exception | None = None
    fail_on_call: bool = False
    healthy: bool = True
    calls: list[tuple[list[dict], dict]] = field(default_factory=list)

    def chat_json(self, messages: list[dict], **kw) -> dict:
        if self.fail_on_call:
            raise AssertionError("chat_json should not have been called")
        self.calls.append((messages, kw))
        if self.error is not None:
            raise self.error
        return dict(self.payload or {"found": False})

    def health(self) -> bool:
        return self.healthy


@dataclass
class FakeRetriever:
    """Stands in for src.retrieval.retriever.Retriever.

    citations/videos/context/used_reranker come back as given, wrapped into a
    RetrievalResult; calls records the arguments each retrieve() got.

    retrieve() takes **kw rather than named parameters on purpose: the real
    Retriever.retrieve gives every parameter a default, and the callers differ
    — answer() passes top_videos/per_video/use_intent_filter, while
    src/eval/run.py calls retrieve(query) bare and retrieve(query, rerank=True).
    """
    citations: list[dict] = field(default_factory=list)
    videos: list = field(default_factory=list)
    context: str = ""
    used_reranker: bool = False
    mode: str = "hybrid"
    calls: list[dict] = field(default_factory=list)

    def retrieve(self, query: str, **kw):
        from src.retrieval.schemas import RetrievalResult

        self.calls.append({"query": query, **kw})
        return RetrievalResult(
            query=query,
            videos=self.videos,
            context=self.context,
            citations=self.citations,
            used_reranker=self.used_reranker,
            mode=self.mode,
        )


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
