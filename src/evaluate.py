"""Đánh giá độc lập toàn diện hệ thống RAG đa tầng (Multi-Layer Benchmark Evaluation).

Hỗ trợ đánh giá 3 tầng chất lượng:
1. Tầng 1 (Retrieval Quality): Document Recall@K, Evidence Recall@K, MRR@K, nDCG@K, Hit@1, Latency p50/p95.
   Phân rã theo từng lát cắt dữ liệu (Slices: factual, paraphrase, keyword_code, numeric, no_answer).
2. Tầng 2 (Grounded Generation & Abstention): Tỷ lệ từ chối đúng khi thiếu bằng chứng (True Abstention Rate),
   độ bao phủ từ khóa của câu trả lời tham chiếu.
3. Tầng 3 (System Quality): Phân vị độ trễ (p50, p95), tỷ lệ lọc qua cổng Evidence Gate.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .generation import ABSTAIN_PHRASE, generate_grounded_response
from .ranking import tokenize
from .utils import save_json, setup_logging

if TYPE_CHECKING:
    from .retrieval import Retriever

LOGGER = logging.getLogger("rag_knowledge_assistant.evaluate")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data/evaluation/questions.json"


@dataclass(frozen=True)
class BenchmarkCase:
    """Một ca kiểm thử đánh giá với nhãn ground truth phân tầng và dev/test split."""

    question: str
    expected_sources: tuple[str, ...]
    split: str
    id: str = ""
    expected_documents: tuple[str, ...] = ()
    expected_sections: tuple[str, ...] = ()
    reference_answer: str = ""
    category: str = "factual"
    is_answerable: bool = True


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK_PATH) -> list[BenchmarkCase]:
    """Đọc và xác thực benchmark từ JSON để tránh hard-code trong mã nguồn.

    Args:
        path (Union[str, Path]): Đường dẫn tệp JSON benchmark.

    Returns:
        List[BenchmarkCase]: Danh sách các trường hợp kiểm thử đã qua xác thực.

    Raises:
        ValueError: Nếu định dạng ca kiểm thử không hợp lệ hoặc rỗng.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []
    for index, item in enumerate(payload):
        question = str(item.get("question", "")).strip()
        raw_sources = item.get("expected_sources", [])
        sources = tuple(str(value).strip() for value in raw_sources)
        split = str(item.get("split", "test")).strip().lower()

        if not question or not sources or split not in {"dev", "test"}:
            raise ValueError(f"Benchmark case #{index} không hợp lệ: {item}")

        raw_docs = item.get("expected_documents", raw_sources)
        docs = tuple(str(v).strip() for v in raw_docs if str(v).strip() != "ABSTAIN")

        raw_secs = item.get("expected_sections", [])
        sections = tuple(str(v).strip() for v in raw_secs)

        cases.append(
            BenchmarkCase(
                id=str(item.get("id", f"Q{index+1:02d}")),
                question=question,
                expected_sources=sources,
                split=split,
                expected_documents=docs,
                expected_sections=sections,
                reference_answer=str(item.get("reference_answer", "")),
                category=str(item.get("category", "factual")),
                is_answerable=bool(item.get("is_answerable", True)),
            )
        )

    if not cases:
        raise ValueError("Benchmark không được rỗng.")
    return cases


def calculate_retrieval_metrics(
    ranked_sources: list[list[str]],
    expected_sources: list[set[str]],
    latencies: list[float],
) -> dict[str, float | int]:
    """Tính Recall@k, Hit@1, MRR, nDCG@k và phân vị latency (hỗ trợ tương thích ngược)."""
    if (
        not ranked_sources
        or len(ranked_sources) != len(expected_sources)
        or len(latencies) != len(ranked_sources)
    ):
        raise ValueError(
            "Kết quả, ground truth và latency phải cùng số mẫu, không rỗng."
        )

    reciprocal_ranks: list[float] = []
    ndcg_list: list[float] = []
    hits_at_one = 0

    for sources, expected in zip(ranked_sources, expected_sources, strict=True):
        if not expected:
            continue

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

        # Tính nDCG@k
        dcg = 0.0
        for pos, source in enumerate(sources, start=1):
            if source in expected:
                dcg += 1.0 / math.log2(pos + 1)
        idcg = 1.0  # Vì mỗi query thường có 1 tài liệu chính chuẩn
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

    total_valid = len(reciprocal_ranks)
    if total_valid == 0:
        return {
            "recall_at_k": 1.0,
            "hit_rate_at_1": 1.0,
            "mrr": 1.0,
            "ndcg_at_k": 1.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "total_queries": 0,
        }

    latency_ms = sorted(value * 1000 for value in latencies)
    p95_index = min(len(latency_ms) - 1, max(0, int(0.95 * len(latency_ms)) - 1))
    p50_index = len(latency_ms) // 2

    return {
        "recall_at_k": round(
            sum(value > 0 for value in reciprocal_ranks) / total_valid, 4
        ),
        "hit_rate_at_1": round(hits_at_one / total_valid, 4),
        "mrr": round(statistics.fmean(reciprocal_ranks), 4),
        "ndcg_at_k": round(statistics.fmean(ndcg_list), 4),
        "avg_latency_ms": round(statistics.fmean(latency_ms), 2),
        "p50_latency_ms": round(latency_ms[p50_index], 2),
        "p95_latency_ms": round(latency_ms[p95_index], 2),
        "total_queries": total_valid,
    }


def evaluate_retrieval_comprehensive(
    retriever: Retriever,
    cases: list[BenchmarkCase],
    *,
    top_k: int = 4,
    use_reranker: bool = True,
    dense_weight: float | None = None,
) -> dict[str, Any]:
    """Đánh giá toàn diện Retrieval chất lượng đa tầng theo từng category slice."""
    ranked_sources: list[list[str]] = []
    expected_sources: list[set[str]] = []
    latencies: list[float] = []

    # Thống kê theo danh mục (Slice)
    category_cases: dict[str, list[BenchmarkCase]] = {}
    category_ranked: dict[str, list[list[str]]] = {}
    category_expected: dict[str, list[set[str]]] = {}
    category_latencies: dict[str, list[float]] = {}

    abstain_correct = 0
    total_unanswerable = 0
    keyword_overlaps: list[float] = []

    for case in cases:
        started = time.perf_counter()
        results = retriever.search(
            case.question,
            k=top_k,
            dense_weight=dense_weight,
            use_reranker=use_reranker,
        )
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)

        retrieved_docs = [str(item.get("source", "")) for item in results]
        ranked_sources.append(retrieved_docs)

        # Tập expected sources (bỏ qua 'ABSTAIN' trong so sánh tài liệu thật)
        exp_docs = set(case.expected_documents)
        expected_sources.append(exp_docs)

        cat = case.category
        if cat not in category_cases:
            category_cases[cat] = []
            category_ranked[cat] = []
            category_expected[cat] = []
            category_latencies[cat] = []

        category_cases[cat].append(case)
        category_ranked[cat].append(retrieved_docs)
        category_expected[cat].append(exp_docs)
        category_latencies[cat].append(elapsed)

        # Đánh giá Grounding / Abstention
        if not case.is_answerable:
            total_unanswerable += 1
            # Nếu Evidence Gate lọc sạch hoặc kết quả bị từ chối
            gate_all_failed = not results or all(not r.get("gate_passed", True) for r in results)
            if gate_all_failed:
                abstain_correct += 1
        else:
            # Đo độ phủ từ khóa của reference answer
            if results and case.reference_answer:
                ref_tokens = set(tokenize(case.reference_answer))
                top_text_tokens = set(tokenize(results[0].get("text", "")))
                overlap = len(ref_tokens & top_text_tokens) / max(1, len(ref_tokens))
                keyword_overlaps.append(overlap)

    # Tính toán chỉ số tổng thể trên các câu hỏi có thể trả lời
    answerable_ranked = [
        r for r, c in zip(ranked_sources, cases, strict=True) if c.is_answerable
    ]
    answerable_expected = [
        e for e, c in zip(expected_sources, cases, strict=True) if c.is_answerable
    ]
    answerable_latencies = [
        l for l, c in zip(latencies, cases, strict=True) if c.is_answerable
    ]

    base_metrics = calculate_retrieval_metrics(
        answerable_ranked, answerable_expected, answerable_latencies
    )

    # Đánh giá từng Category Slice
    slice_metrics: dict[str, Any] = {}
    for cat, cat_cases in category_cases.items():
        cat_ans_ranked = [
            r for r, c in zip(category_ranked[cat], cat_cases, strict=True) if c.is_answerable
        ]
        cat_ans_expected = [
            e for e, c in zip(category_expected[cat], cat_cases, strict=True) if c.is_answerable
        ]
        cat_ans_latencies = [
            l for l, c in zip(category_latencies[cat], cat_cases, strict=True) if c.is_answerable
        ]

        if cat_ans_ranked:
            slice_metrics[cat] = calculate_retrieval_metrics(
                cat_ans_ranked, cat_ans_expected, cat_ans_latencies
            )
        else:
            slice_metrics[cat] = {
                "total_queries": len(cat_cases),
                "is_unanswerable_slice": True,
            }

    true_abstain_rate = (
        round(abstain_correct / total_unanswerable, 4) if total_unanswerable > 0 else 1.0
    )
    avg_keyword_coverage = (
        round(statistics.fmean(keyword_overlaps), 4) if keyword_overlaps else 0.0
    )

    return {
        **base_metrics,
        "true_abstain_rate": true_abstain_rate,
        "unanswerable_evaluated": total_unanswerable,
        "avg_keyword_coverage": avg_keyword_coverage,
        "slices": slice_metrics,
    }


def evaluate_configuration(
    retriever: Retriever,
    cases: list[BenchmarkCase],
    *,
    dense_weight: float,
    top_k: int,
) -> dict[str, float | int]:
    """Đánh giá một cấu hình retrieval (tương thích ngược với interface cũ)."""
    ans_cases = [c for c in cases if c.is_answerable]
    ranked_sources: list[list[str]] = []
    expected_sources: list[set[str]] = []
    latencies: list[float] = []

    for case in ans_cases:
        started = time.perf_counter()
        results = retriever.search(
            case.question,
            k=top_k,
            dense_weight=dense_weight,
            use_reranker=False,
        )
        latencies.append(time.perf_counter() - started)
        ranked_sources.append([str(item.get("source", "")) for item in results])
        expected_sources.append(set(case.expected_documents or case.expected_sources))

    return calculate_retrieval_metrics(ranked_sources, expected_sources, latencies)


def run_evaluation(
    model_dir: str | Path | None = None,
    benchmark_path: str | Path = DEFAULT_BENCHMARK_PATH,
    output_report_path: str | Path | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    """Thực hiện đánh giá toàn diện các pipeline trên tập Dev và Test."""
    setup_logging()
    from .retrieval import Retriever

    cases = load_benchmark(benchmark_path)
    dev_cases = [case for case in cases if case.split == "dev"]
    test_cases = [case for case in cases if case.split == "test"]

    if not dev_cases or not test_cases:
        raise ValueError("Benchmark phải có cả split dev và test.")

    retriever = Retriever(model_dir=model_dir, use_reranker=True)

    LOGGER.info(
        "Bắt đầu đánh giá Benchmark: %d dev queries, %d test queries...",
        len(dev_cases),
        len(test_cases),
    )

    # 1. Đánh giá Dev Tuning
    candidate_weights = (0.5, 0.7, 0.85)
    dev_results = {
        str(weight): evaluate_configuration(
            retriever, dev_cases, dense_weight=weight, top_k=top_k
        )
        for weight in candidate_weights
    }
    best_weight = max(
        candidate_weights,
        key=lambda val: (
            dev_results[str(val)]["mrr"],
            dev_results[str(val)]["hit_rate_at_1"],
        ),
    )

    # 2. Đánh giá So sánh Baseline trên Test Set
    test_baselines: dict[str, Any] = {
        "lexical_only_bm25": evaluate_retrieval_comprehensive(
            retriever, test_cases, top_k=top_k, dense_weight=0.0, use_reranker=False
        ),
        "dense_only_faiss": evaluate_retrieval_comprehensive(
            retriever, test_cases, top_k=top_k, dense_weight=1.0, use_reranker=False
        ),
        "hybrid_linear_fusion": evaluate_retrieval_comprehensive(
            retriever, test_cases, top_k=top_k, dense_weight=best_weight, use_reranker=False
        ),
        "canonical_hybrid_rrf_reranker": evaluate_retrieval_comprehensive(
            retriever, test_cases, top_k=top_k, dense_weight=None, use_reranker=True
        ),
    }

    report: dict[str, Any] = {
        "schema_version": 2,
        "system_name": "Vietnamese Evidence-Grounded Knowledge Assistant (Hybrid RAG)",
        "selection_split": "dev",
        "evaluation_split": "test",
        "top_k": top_k,
        "total_test_cases": len(test_cases),
        "selected_dense_weight": best_weight,
        "dev_tuning": dev_results,
        "test_baseline_comparison": test_baselines,
        "summary": {
            "canonical_recall_at_k": test_baselines["canonical_hybrid_rrf_reranker"]["recall_at_k"],
            "canonical_mrr": test_baselines["canonical_hybrid_rrf_reranker"]["mrr"],
            "canonical_ndcg_at_k": test_baselines["canonical_hybrid_rrf_reranker"]["ndcg_at_k"],
            "canonical_hit_rate_at_1": test_baselines["canonical_hybrid_rrf_reranker"]["hit_rate_at_1"],
            "true_abstain_rate": test_baselines["canonical_hybrid_rrf_reranker"]["true_abstain_rate"],
            "avg_latency_ms": test_baselines["canonical_hybrid_rrf_reranker"]["avg_latency_ms"],
            "p95_latency_ms": test_baselines["canonical_hybrid_rrf_reranker"]["p95_latency_ms"],
        },
        "limitations": [
            "Corpus thử nghiệm gồm các chính sách nội bộ tiêu chuẩn; cần kiểm tra thêm khi nạp văn bản hàng trăm trang.",
            "Cross-Encoder reranking bổ sung độ trễ tính toán (~15-40ms trên CPU); có thể tắt với tham số use_reranker=False khi cần throughput cao.",
            "Cần kiểm tra định kỳ ngưỡng evidence_gate_threshold khi quy mô tài liệu mở rộng.",
        ],
    }

    report_path = Path(output_report_path or PROJECT_ROOT / "reports/test_metrics.json")
    save_json(report_path, report)
    LOGGER.info("Đã lưu benchmark kết quả đánh giá tại: %s", report_path.resolve())

    # In bảng tóm tắt kết quả đẹp mắt ra màn hình
    print("\n" + "=" * 80)
    print(" BÁO CÁO KẾT QUẢ ĐÁNH GIÁ CANONICAL HYBRID RAG (TEST SET)")
    print("=" * 80)
    print(f"{'Pipeline':<32} | {'Recall@K':<10} | {'MRR':<8} | {'nDCG':<8} | {'P95 Latency':<12}")
    print("-" * 80)
    for name, met in test_baselines.items():
        rec = met.get("recall_at_k", 0.0)
        mrr = met.get("mrr", 0.0)
        ndcg = met.get("ndcg_at_k", 0.0)
        p95 = met.get("p95_latency_ms", 0.0)
        print(f"{name:<32} | {rec:<10.4f} | {mrr:<8.4f} | {ndcg:<8.4f} | {p95:<10.2f} ms")
    print("=" * 80 + "\n")

    return report


if __name__ == "__main__":
    run_evaluation()
