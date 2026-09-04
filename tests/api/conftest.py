"""Guardrails for the API tests.

The API is the one place where a unit test can accidentally build the whole
production stack: the lifespan constructs a real Retriever, and a Retriever
constructs the embedder, the vector store and the reranker.
"""
from __future__ import annotations

import pytest

import src.retrieval.retriever as retriever_mod


@pytest.fixture(autouse=True)
def _no_real_retriever_dependencies(monkeypatch):
    """Makes building the real retrieval stack fail loudly and instantly.

    The lifespan test substitutes main.Retriever, which is enough today. But
    that substitution only holds while the lifespan resolves Retriever as a
    module global: rewrite it as a function-local import and the patch is
    bypassed, bge-m3 and the reranker download for real, and the suite goes
    from two seconds to several minutes with no explanation.

    These three names are what Retriever.__init__ actually reaches for, so
    tripping them turns any such bypass into an immediate, readable failure.
    """
    def forbidden(name):
        def _raise(*args, **kwargs):
            raise AssertionError(
                f"{name} was constructed for real in an API unit test — "
                "the lifespan substitution was bypassed"
            )
        return _raise

    monkeypatch.setattr(retriever_mod, "BGEM3Embedder", forbidden("BGEM3Embedder"))
    monkeypatch.setattr(retriever_mod, "VectorStore", forbidden("VectorStore"))
    monkeypatch.setattr(retriever_mod, "try_load_reranker", forbidden("try_load_reranker"))
