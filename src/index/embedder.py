"""Embedding chunks and queries with BGE-M3 (Step 3).

From a single model BGE-M3 returns:
  - dense   — a 1024-dimensional dense vector (cosine),
  - lexical — a sparse bag of weighted tokens (for hybrid search).
The model needs NO query/passage prefixes.
"""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from src.config import EMBED_DEVICE, EMBED_MAX_LENGTH, EMBED_MODEL, EMBED_USE_FP16


@dataclass
class Embedding:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class BGEM3Embedder:
    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        *,
        device: str | None = EMBED_DEVICE,
        use_fp16: bool = EMBED_USE_FP16,
        max_length: int = EMBED_MAX_LENGTH,
        batch_size: int = 12,
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        if (device or "").lower() == "cpu":
            use_fp16 = False  # fp16 is not supported on CPU
        kwargs: dict = {"use_fp16": use_fp16}
        if device:
            kwargs["devices"] = device
        self.model = BGEM3FlagModel(model_name, **kwargs)
        self.max_length = max_length
        self.batch_size = batch_size
        logger.info("BGE-M3 загружен: {} (fp16={}, max_len={})", model_name, use_fp16, max_length)

    def _encode(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        lexical = out["lexical_weights"]
        embs: list[Embedding] = []
        for d, lw in zip(dense, lexical):
            idx: list[int] = []
            val: list[float] = []
            for k, v in lw.items():
                fv = float(v)
                if fv <= 0.0:
                    continue
                idx.append(int(k))
                val.append(fv)
            embs.append(Embedding(dense=[float(x) for x in d], sparse_indices=idx, sparse_values=val))
        return embs

    # BGE-M3 is symmetric: queries and passages are encoded the same way.
    encode_passages = _encode
    encode_queries = _encode
