"""Замер качества на golden-наборе (Шаг 7).

    # только retrieval, без LLM — быстро, гоняется на каждую правку ранжирования
    python -m src.eval.run --mode retrieval

    # сквозной прогон с генерацией (медленно: ~30 с на запрос)
    python -m src.eval.run --mode answer

    # в докере, где модели уже подняты:
    docker compose exec -T api python -m src.eval.run --mode retrieval

Метрики retrieval (по видео, негативные запросы исключены):
    hit@k  — хотя бы одно релевантное видео в top-k
    MRR    — 1/позиция первого релевантного видео
    ctx_precision — доля пассажей контекста из релевантных видео: показывает,
                    сколько бюджета токенов ушло на посторонние ролики
    edge_rate     — доля пассажей, начатых крайним чанком ролика (вступление
                    или прощание). Прямой индикатор той проблемы, из-за которой
                    в контекст попадали «сегодня готовим борщ» вместо рецепта.

Метрики answer:
    found_acc     — совпал ли found с ожиданием (для negative ожидается false)
    attribution   — source.video_id попал в relevant
    steps/ingr    — средняя детализация ответа
    with_amounts  — доля ингредиентов, где есть число (граммовки)
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.eval.dataset import DEFAULT_GOLDEN, GoldenItem, dump_report, load_golden

_HAS_DIGIT = re.compile(r"\d")
RANK_KS = (1, 3, 5)


def _chunk_index(chunk_id: str) -> int:
    """chunk_id имеет вид '<video_id>::007'."""
    try:
        return int(chunk_id.rsplit("::", 1)[1])
    except (IndexError, ValueError):
        return -1


def _mean(xs: list[float]) -> float:
    return round(statistics.mean(xs), 4) if xs else 0.0


# --------------------------------------------------------------------
def eval_retrieval(items: list[GoldenItem], retriever, rank_k: int, budget: int | None) -> dict:
    positives = [i for i in items if not i.is_negative]
    hits: dict[int, list[float]] = {k: [] for k in RANK_KS}
    rr: list[float] = []
    ctx_prec: list[float] = []
    edge: list[float] = []
    rows: list[dict] = []

    for it in tqdm(positives, desc="retrieval", unit="q"):
        # проход 1 — ранжирование: широкий top_videos, иначе hit@5 не измерить
        wide = retriever.retrieve(it.query, top_videos=rank_k, rerank=True)
        order = [v.video_id for v in wide.videos]
        rank = next((i + 1 for i, v in enumerate(order) if v in it.relevant), None)
        for k in RANK_KS:
            hits[k].append(1.0 if rank is not None and rank <= k else 0.0)
        rr.append(1.0 / rank if rank else 0.0)

        # проход 2 — боевая конфигурация: что реально уедет в промпт
        kw = {"token_budget": budget} if budget else {}
        prod = retriever.retrieve(it.query, **kw)
        passages = [p for v in prod.videos for p in v.passages]
        if passages:
            ctx_prec.append(
                sum(1 for p in passages if p.video_id in it.relevant) / len(passages)
            )
            edge.append(
                sum(1 for p in passages if _chunk_index(p.chunk_ids[0]) == 0) / len(passages)
            )

        rows.append(
            {
                "id": it.id, "query": it.query, "kind": it.kind,
                "rank": rank, "top": order[:3],
                "ctx_precision": round(ctx_prec[-1], 3) if ctx_prec else None,
            }
        )

    summary = {"n": len(positives), "mrr": _mean(rr)}
    summary.update({f"hit@{k}": _mean(v) for k, v in hits.items()})
    summary["ctx_precision"] = _mean(ctx_prec)
    summary["edge_rate"] = _mean(edge)

    by_kind: dict[str, dict] = {}
    for kind in sorted({i.kind for i in positives}):
        sub = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {
            "n": len(sub),
            "hit@1": _mean([1.0 if r["rank"] == 1 else 0.0 for r in sub]),
            "hit@3": _mean([1.0 if r["rank"] and r["rank"] <= 3 else 0.0 for r in sub]),
        }
    return {"summary": summary, "by_kind": by_kind, "rows": rows}


# --------------------------------------------------------------------
def eval_answer(items: list[GoldenItem], retriever, llm) -> dict:
    from src.generation.answer import answer

    found_ok: list[float] = []
    attribution: list[float] = []
    steps: list[float] = []
    ingr: list[float] = []
    amounts: list[float] = []
    rows: list[dict] = []

    for it in tqdm(items, desc="answer", unit="q"):
        t0 = time.time()
        try:
            res = answer(it.query, retriever=retriever, llm=llm)
        except Exception as exc:  # noqa: BLE001
            logger.warning("{}: {}", it.id, exc)
            rows.append({"id": it.id, "query": it.query, "error": str(exc)[:200]})
            continue
        dt = time.time() - t0

        found_ok.append(1.0 if res.found == it.expect_found else 0.0)
        if res.found and not it.is_negative:
            attribution.append(1.0 if res.source and res.source.video_id in it.relevant else 0.0)
            steps.append(len(res.steps))
            ingr.append(len(res.ingredients))
            if res.ingredients:
                amounts.append(
                    sum(1 for x in res.ingredients if _HAS_DIGIT.search(x)) / len(res.ingredients)
                )
        rows.append(
            {
                "id": it.id, "query": it.query, "kind": it.kind,
                "found": res.found, "expect_found": it.expect_found,
                "steps": len(res.steps), "ingredients": len(res.ingredients),
                "source": res.source.video_id if res.source else None,
                "seconds": round(dt, 1),
            }
        )

    return {
        "summary": {
            "n": len(rows),
            "found_acc": _mean(found_ok),
            "attribution": _mean(attribution),
            "avg_steps": _mean(steps),
            "avg_ingredients": _mean(ingr),
            "with_amounts": _mean(amounts),
        },
        "rows": rows,
    }


# --------------------------------------------------------------------
def _print(title: str, summary: dict) -> None:
    print(f"\n=== {title} ===")
    for k, v in summary.items():
        print(f"  {k:16s} {v}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Оценка RAG на golden-наборе")
    p.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    p.add_argument("--mode", choices=["retrieval", "answer", "both"], default="retrieval")
    p.add_argument("--kinds", default=None, help="через запятую: exact,paraphrase,descriptive,negative")
    p.add_argument("--rank-k", type=int, default=10, help="глубина списка видео для hit@k / MRR")
    p.add_argument("--budget", type=int, default=None, help="переопределить бюджет токенов контекста")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--out", default=None, help="куда сложить подробный JSON-отчёт")
    args = p.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="WARNING", format="{level: <7} | {message}")
    sys.stdout.reconfigure(encoding="utf-8")

    kinds = [k.strip() for k in args.kinds.split(",")] if args.kinds else None
    items = load_golden(args.golden, kinds)
    if not items:
        print("golden-набор пуст", file=sys.stderr)
        return 1
    print(f"golden: {len(items)} запросов ({sum(1 for i in items if i.is_negative)} негативных)")

    from src.retrieval.retriever import Retriever

    retriever = Retriever(reranker=None if args.no_rerank else "auto")
    report: dict = {"golden": str(args.golden), "rank_k": args.rank_k}

    if args.mode in ("retrieval", "both"):
        report["retrieval"] = eval_retrieval(items, retriever, args.rank_k, args.budget)
        _print("RETRIEVAL", report["retrieval"]["summary"])
        print("\n  по типам запроса:")
        for kind, m in report["retrieval"]["by_kind"].items():
            print(f"    {kind:12s} n={m['n']:2d}  hit@1={m['hit@1']:.2f}  hit@3={m['hit@3']:.2f}")
        miss = [r for r in report["retrieval"]["rows"] if r["rank"] is None]
        if miss:
            print(f"\n  промахи ({len(miss)}):")
            for r in miss:
                print(f"    {r['id']} {r['query']}")

    if args.mode in ("answer", "both"):
        from src.generation.llm import LLMClient

        llm = LLMClient()
        if not llm.health():
            print("!! LLM недоступен: docker compose up -d vllm", file=sys.stderr)
            return 2
        report["answer"] = eval_answer(items, retriever, llm)
        _print("ANSWER", report["answer"]["summary"])
        wrong = [
            r for r in report["answer"]["rows"]
            if "error" not in r and r["found"] != r["expect_found"]
        ]
        if wrong:
            print(f"\n  расхождения по found ({len(wrong)}):")
            for r in wrong:
                print(f"    {r['id']} {r['query']}  found={r['found']} ожидалось={r['expect_found']}")

    if args.out:
        dump_report(args.out, report)
        print(f"\nотчёт: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
