"""Đánh giá độc lập các cấu hình lexical, dense và hybrid retrieval."""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .utils import save_json, setup_logging

if TYPE_CHECKING:
    from .retrieval import Retriever

LOGGER = logging.getLogger("rag_knowledge_assistant.evaluate")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data/evaluation/questions.json"


@dataclass(frozen=True)
class BenchmarkCase:
    """Một câu hỏi đánh giá với nguồn đúng và nhãn dev/test."""

    question: str
    expected_sources: tuple[str, ...]
    split: str


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK_PATH) -> list[BenchmarkCase]:
    """Đọc và xác thực benchmark từ JSON để tránh hard-code trong mã nguồn."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []
    for index, item in enumerate(payload):
        question = str(item.get("question", "")).strip()
        sources = tuple(
            str(value).strip() for value in item.get("expected_sources", [])
        )
        split = str(item.get("split", "test")).strip().lower()
        if not question or not sources or split not in {"dev", "test"}:
            raise ValueError(f"Benchmark case #{index} không hợp lệ: {item}")
        cases.append(BenchmarkCase(question, sources, split))
    if not cases:
        raise ValueError("Benchmark không được rỗng.")
    return cases


def calculate_retrieval_metrics(
    ranked_sources: list[list[str]],
    expected_sources: list[set[str]],
    latencies: list[float],
) -> dict[str, float | int]:
    """Tính Recall@k, Hit@1, MRR và phân vị latency."""
    if (
        not ranked_sources
        or len(ranked_sources) != len(expected_sources)
        or len(latencies) != len(ranked_sources)
    ):
        raise ValueError(
            "Kết quả, ground truth và latency phải cùng số mẫu, không rỗng."
        )

    reciprocal_ranks: list[float] = []
    hits_at_one = 0
    for sources, expected in zip(ranked_sources, expected_sources, strict=True):
        rank = next(
            (
                position
                for position, source in enumerate(sources, start=1)
                if source in expected
            ),
            None,
        )
        hits_at_one += int(rank == 1)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    latency_ms = sorted(value * 1000 for value in latencies)
    p95_index = min(len(latency_ms) - 1, max(0, int(0.95 * len(latency_ms)) - 1))
    return {
        "recall_at_k": round(
            sum(value > 0 for value in reciprocal_ranks) / len(reciprocal_ranks), 4
        ),
        "hit_rate_at_1": round(hits_at_one / len(reciprocal_ranks), 4),
        "mrr": round(statistics.fmean(reciprocal_ranks), 4),
        "avg_latency_ms": round(statistics.fmean(latency_ms), 2),
        "p95_latency_ms": round(latency_ms[p95_index], 2),
        "total_queries": len(reciprocal_ranks),
    }


def evaluate_configuration(
    retriever: Retriever,
    cases: list[BenchmarkCase],
    *,
    dense_weight: float,
    top_k: int,
) -> dict[str, float | int]:
    """Đánh giá một trọng số retrieval trên đúng tập câu hỏi được cung cấp."""
    ranked_sources: list[list[str]] = []
    expected_sources: list[set[str]] = []
    latencies: list[float] = []
    for case in cases:
        started = time.perf_counter()
        results = retriever.search(case.question, k=top_k, dense_weight=dense_weight)
        latencies.append(time.perf_counter() - started)
        ranked_sources.append([str(item.get("source", "")) for item in results])
        expected_sources.append(set(case.expected_sources))
    return calculate_retrieval_metrics(ranked_sources, expected_sources, latencies)


def run_evaluation(
    model_dir: str | Path | None = None,
    benchmark_path: str | Path = DEFAULT_BENCHMARK_PATH,
    output_report_path: str | Path | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Chọn hybrid weight trên dev và so sánh baseline một lần trên test."""
    setup_logging()
    from .retrieval import Retriever

    cases = load_benchmark(benchmark_path)
    dev_cases = [case for case in cases if case.split == "dev"]
    test_cases = [case for case in cases if case.split == "test"]
    if not dev_cases or not test_cases:
        raise ValueError("Benchmark phải có cả split dev và test.")

    retriever = Retriever(model_dir=model_dir)
    candidate_weights = (0.5, 0.7, 0.85)
    dev_results = {
        str(weight): evaluate_configuration(
            retriever, dev_cases, dense_weight=weight, top_k=top_k
        )
        for weight in candidate_weights
    }
    best_weight = max(
        candidate_weights,
        key=lambda value: (
            dev_results[str(value)]["mrr"],
            dev_results[str(value)]["hit_rate_at_1"],
        ),
    )

    configurations = {
        "lexical_only": 0.0,
        "dense_only": 1.0,
        "hybrid_selected_on_dev": best_weight,
    }
    test_results = {
        name: {
            **evaluate_configuration(
                retriever, test_cases, dense_weight=weight, top_k=top_k
            ),
            "dense_weight": weight,
        }
        for name, weight in configurations.items()
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "selection_split": "dev",
        "evaluation_split": "test",
        "top_k": top_k,
        "selected_dense_weight": best_weight,
        "dev_tuning": dev_results,
        "test_baseline_comparison": test_results,
        "limitations": [
            "Benchmark mẫu còn nhỏ; không suy rộng sang tài liệu doanh nghiệp khác.",
            "Chỉ đánh giá retrieval, chưa đánh giá faithfulness của LLM.",
        ],
    }
    report_path = Path(output_report_path or PROJECT_ROOT / "reports/test_metrics.json")
    save_json(report_path, report)
    LOGGER.info("Đã lưu benchmark retrieval tại %s", report_path.resolve())
    return report


if __name__ == "__main__":
    run_evaluation()
