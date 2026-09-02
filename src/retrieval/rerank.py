"""A reranker on top of the hybrid search candidates (Step 4).

BAAI/bge-reranker-v2-m3 is a cross-encoder: it sorts the top-N by the
(query, chunk) pair more accurately than a pure bi-encoder / fusion does.

RERANK_DEVICE=cpu is for when the GPU is taken by vLLM (Step 5). On CPU fp16 is
switched off.
"""
from __future__ import annotations

from loguru import logger

from src.config import RERANK_DEVICE, RERANK_MODEL


class Reranker:
    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        *,
        device: str | None = RERANK_DEVICE,
        use_fp16: bool | None = None,
    ) -> None:
        from FlagEmbedding import FlagReranker

        on_cpu = (device or "").lower() == "cpu"
        if use_fp16 is None:
            use_fp16 = not on_cpu
        kwargs: dict = {"use_fp16": use_fp16}
        if device:
            kwargs["devices"] = device
        self.model = FlagReranker(model_name, **kwargs)
        logger.info("Reranker загружен: {} (device={}, fp16={})", model_name, device or "auto", use_fp16)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [[query, t] for t in texts]
        raw = self.model.compute_score(pairs, normalize=True)
        if isinstance(raw, float):
            raw = [raw]
        return [float(x) for x in raw]


def try_load_reranker() -> Reranker | None:
    try:
        return Reranker()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reranker недоступен ({}) — работаем без него", exc)
        return None
