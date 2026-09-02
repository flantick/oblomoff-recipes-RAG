"""Retriever: hybrid search -> rerank -> dedup by video -> stitching of adjacent
chunks -> context assembly within a token budget (Step 4).

Returns a RetrievalResult ready for generation (Step 5).
"""
from __future__ import annotations

import math

from loguru import logger

from src.config import (
    RETRIEVAL_FIRST_CHUNK_WEIGHT,
    RETRIEVAL_LAST_CHUNK_WEIGHT,
    RETRIEVAL_MAX_PASSAGE_WORDS,
    RETRIEVAL_NEIGHBOR_RADIUS,
    RETRIEVAL_OVERFETCH,
    RETRIEVAL_PER_VIDEO,
    RETRIEVAL_RERANK_TOP,
    RETRIEVAL_SECTION_WEIGHTS,
    RETRIEVAL_TOKEN_BUDGET,
    RETRIEVAL_TOP_VIDEOS,
)
from src.chunking.tokens import TokenCounter
from src.index.embedder import BGEM3Embedder
from src.index.store import VectorStore
from src.retrieval.filters import build_intent_filter
from src.retrieval.rerank import Reranker, try_load_reranker
from src.retrieval.schemas import (
    RetrievalResult,
    RetrievedChunk,
    RetrievedPassage,
    RetrievedVideo,
)


def _truncate_words(text: str, max_words: int) -> str:
    w = text.split()
    return text if len(w) <= max_words else " ".join(w[:max_words]) + " …"


def _stitch(a: str, b: str, max_overlap_words: int = 120) -> str:
    """Stitches two overlapping chunks: drops the longest suffix of a that
    matches a prefix of b (word-wise)."""
    aw, bw = a.split(), b.split()
    limit = min(max_overlap_words, len(aw), len(bw))
    for n in range(limit, 0, -1):
        if [w.lower() for w in aw[-n:]] == [w.lower() for w in bw[:n]]:
            return " ".join(aw + bw[n:])
    return a + " " + b


class Retriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: BGEM3Embedder | None = None,
        reranker: Reranker | None | str = "auto",
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.store = store or VectorStore()
        self.embedder = embedder or BGEM3Embedder()
        if reranker == "auto":
            self.reranker = try_load_reranker()
        else:
            self.reranker = reranker  # None or a ready object
        self.counter = token_counter or TokenCounter()

    # --- the main method -------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        top_videos: int = RETRIEVAL_TOP_VIDEOS,
        per_video: int = RETRIEVAL_PER_VIDEO,
        overfetch: int = RETRIEVAL_OVERFETCH,
        rerank: bool = True,
        rerank_top: int = RETRIEVAL_RERANK_TOP,
        token_budget: int = RETRIEVAL_TOKEN_BUDGET,
        use_intent_filter: bool = False,
        neighbor_radius: int = RETRIEVAL_NEIGHBOR_RADIUS,
        section_weights: bool = True,
    ) -> RetrievalResult:
        flt, matched = (build_intent_filter(query) if use_intent_filter else (None, []))

        pool = max(top_videos * per_video * overfetch, rerank_top)
        emb = self.embedder.encode_queries([query])[0]
        points = self.store.search(emb, k=pool, mode=mode, query_filter=flt)

        cands: list[RetrievedChunk] = []
        for p in points:
            pl = p.payload or {}
            cands.append(
                RetrievedChunk(
                    chunk_id=pl.get("chunk_id", str(p.id)),
                    video_id=pl.get("video_id", ""),
                    title=pl.get("title", ""),
                    url=pl.get("url", ""),
                    timecode=pl.get("timecode", ""),
                    start=float(pl.get("start", 0.0)),
                    end=float(pl.get("end", 0.0)),
                    chunk_index=int(pl.get("chunk_index", 0)),
                    n_chunks=int(pl.get("n_chunks", 0)),
                    section=pl.get("section", "other"),
                    text=pl.get("text", ""),
                    score=float(p.score),
                    retriever_score=float(p.score),
                )
            )

        used_reranker = False
        if rerank and self.reranker is not None and cands:
            head = cands[:rerank_top]
            scores = self.reranker.score(query, [c.text for c in head])
            for c, s in zip(head, scores):
                c.score = s
            head.sort(key=lambda c: c.score, reverse=True)
            # the tail (not reranked) must not overtake the reranked passages
            for c in cands[rerank_top:]:
                c.score = 0.0
            cands = head + cands[rerank_top:]
            used_reranker = True

        # The opening ("сегодня готовим борщ", an enumeration of the dish
        # varieties) and the closing ("ставьте лайк") are the champions in dish
        # name density and therefore float to the top even though they contain no
        # recipe. We push them down to free up slots for the middle of the video,
        # which is where the cooking actually happens.
        if section_weights:
            for c in cands:
                c.score *= self._position_weight(c)
            cands.sort(key=lambda c: c.score, reverse=True)

        videos = self._group(cands, top_videos, per_video, neighbor_radius)
        context, citations = self._build_context(videos, token_budget)

        return RetrievalResult(
            query=query,
            videos=videos,
            context=context,
            citations=citations,
            used_reranker=used_reranker,
            mode=mode,
            debug={
                "candidates": len(cands),
                "pool": pool,
                "intent_filter": matched,
                "filtered": flt is not None,
                "neighbor_radius": neighbor_radius,
                "passages": sum(len(v.passages) for v in videos),
            },
        )

    @staticmethod
    def _position_weight(c: RetrievedChunk) -> float:
        """A penalty for the edges of a video: position beats the section label.

        On an ASR corpus the section labelling is degenerate (78% of the chunks
        are steps), and the first chunk is tagged intro for only a fifth of the
        videos. chunk_index, on the other hand, cannot lie: chunk zero is always
        the greeting and the last one is always the farewell.
        """
        w = RETRIEVAL_SECTION_WEIGHTS.get(c.section, 1.0)
        if c.chunk_index == 0:
            w *= RETRIEVAL_FIRST_CHUNK_WEIGHT
        elif c.n_chunks and c.chunk_index >= c.n_chunks - 1:
            w *= RETRIEVAL_LAST_CHUNK_WEIGHT
        return w

    # --- grouping by video + stitching of adjacent chunks ----
    def _group(
        self,
        cands: list[RetrievedChunk],
        top_videos: int,
        per_video: int,
        neighbor_radius: int = 0,
    ) -> list[RetrievedVideo]:
        by_video: dict[str, list[RetrievedChunk]] = {}
        order: list[str] = []
        for c in cands:
            if c.video_id not in by_video:
                by_video[c.video_id] = []
                order.append(c.video_id)
            by_video[c.video_id].append(c)

        # The video rank is computed BEFORE the neighbours are fetched: the
        # neighbours are filler and must not influence which video won.
        ranked = sorted(
            ((vid, max(c.score for c in by_video[vid])) for vid in order),
            key=lambda t: t[1],
            reverse=True,
        )[:top_videos]

        videos: list[RetrievedVideo] = []
        for vid, vscore in ranked:
            chunks = sorted(by_video[vid], key=lambda c: c.score, reverse=True)[:per_video]
            if neighbor_radius > 0:
                chunks = self._with_neighbors(vid, chunks, neighbor_radius)
            chunks.sort(key=lambda c: c.chunk_index)
            videos.append(
                RetrievedVideo(
                    video_id=vid,
                    title=chunks[0].title,
                    score=vscore,
                    passages=self._merge_adjacent(chunks),
                )
            )
        return videos

    def _with_neighbors(
        self, video_id: str, chunks: list[RetrievedChunk], radius: int
    ) -> list[RetrievedChunk]:
        """Fetches the chunks surrounding the retrieved ones from Qdrant.

        Ranking picks isolated points out of a recipe, while the recipe itself is
        continuous: a step started in chunk i is finished in i+1, which did not
        make the top. The neighbours arrive with score=0 — they are filler and
        affect neither the video rank nor the passage order, but they let
        _merge_adjacent close the gaps.
        """
        have = {c.chunk_index for c in chunks}
        wanted: set[int] = set()
        for c in chunks:
            for d in range(1, radius + 1):
                wanted.add(c.chunk_index - d)
                wanted.add(c.chunk_index + d)
        wanted = {i for i in wanted if i >= 0 and i not in have}
        if not wanted:
            return chunks

        try:
            payloads = self.store.fetch_chunks(video_id, sorted(wanted))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Дозабор соседей для {} не удался: {}", video_id, exc)
            return chunks

        extra = [
            RetrievedChunk(
                chunk_id=pl.get("chunk_id", ""),
                video_id=pl.get("video_id", video_id),
                title=pl.get("title", ""),
                url=pl.get("url", ""),
                timecode=pl.get("timecode", ""),
                start=float(pl.get("start", 0.0)),
                end=float(pl.get("end", 0.0)),
                chunk_index=int(pl.get("chunk_index", 0)),
                n_chunks=int(pl.get("n_chunks", 0)),
                section=pl.get("section", "other"),
                text=pl.get("text", ""),
                score=0.0,
                retriever_score=0.0,
            )
            for pl in payloads
        ]
        return chunks + extra

    @staticmethod
    def _merge_adjacent(
        chunks: list[RetrievedChunk], max_words: int = RETRIEVAL_MAX_PASSAGE_WORDS
    ) -> list[RetrievedPassage]:
        """Stitches adjacent chunks together, but never beyond max_words.

        Without a limit, a stitch of 6+ chunks produced a single 2000-word
        passage which _build_context then truncated from the head — that is, down
        to the introduction, where there is no recipe. Returning several passages
        is better: they arrive whole.
        """
        passages: list[RetrievedPassage] = []
        cur: list[RetrievedChunk] = []
        cur_words = 0

        def flush() -> None:
            if not cur:
                return
            text = cur[0].text
            for nxt in cur[1:]:
                text = _stitch(text, nxt.text)
            passages.append(
                RetrievedPassage(
                    video_id=cur[0].video_id,
                    title=cur[0].title,
                    url=cur[0].url,
                    timecode=cur[0].timecode,
                    start=cur[0].start,
                    end=cur[-1].end,
                    text=text,
                    chunk_ids=[c.chunk_id for c in cur],
                    score=max(c.score for c in cur),
                )
            )

        for c in chunks:
            words = len(c.text.split())
            adjacent = bool(cur) and c.chunk_index == cur[-1].chunk_index + 1
            if adjacent and cur_words + words <= max_words:
                cur.append(c)
                cur_words += words
            else:
                flush()
                cur = [c]
                cur_words = words
        flush()
        passages.sort(key=lambda p: p.start)
        return passages

    # --- context assembly within the budget ------------------
    def _build_context(self, videos: list[RetrievedVideo], budget: int) -> tuple[str, list[dict]]:
        """Selection goes by relevance, the order in the prompt goes by time.

        These are two different jobs that used to be solved by a single sort by
        score: the budget must be spent on the most relevant passages, but in the
        prompt they must appear in cooking order. Otherwise the LLM received
        "leave it overnight" before "boil the potatoes" and restored the order as
        best it could.
        """
        ranked = [(vi, p) for vi, v in enumerate(videos) for p in v.passages]

        chosen: list[tuple[int, RetrievedPassage, str]] = []
        used = 0
        for vi, p in sorted(ranked, key=lambda t: t[1].score, reverse=True):
            text = _truncate_words(p.text, RETRIEVAL_MAX_PASSAGE_WORDS)
            cost = self.counter.count(text) + 24          # +24 for the block header
            if chosen and used + cost > budget:
                continue                                  # not break: smaller ones may still fit
            chosen.append((vi, p, text))
            used += cost

        chosen.sort(key=lambda t: (t[0], t[1].start))     # videos by rank, passages by time

        blocks: list[str] = []
        citations: list[dict] = []
        for n, (_vi, p, text) in enumerate(chosen, 1):
            citations.append(
                {
                    "n": n,
                    "video_id": p.video_id,
                    "title": p.title,
                    "url": p.url,
                    "timecode": p.timecode,
                }
            )
            blocks.append(f"[{n}] {p.title} ({p.timecode})\n{text}")
        return "\n\n".join(blocks), citations


_DEFAULT: Retriever | None = None


def get_retriever() -> Retriever:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Retriever()
    return _DEFAULT
