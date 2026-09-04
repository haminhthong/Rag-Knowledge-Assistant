"""Unit tests bổ sung cho module ranking và API routes."""

from __future__ import annotations

import pytest
from src.evaluate import calculate_retrieval_metrics, load_benchmark
from src.ranking import bm25_score_single, hybrid_score, tokenize


def test_tokenize_vietnamese_text():
    """Kiểm tra việc tách từ tiếng Việt có dấu."""
    text = "Nhân viên có 12 ngày phép năm!"
    tokens = tokenize(text)
    assert "nhân" in tokens
    assert "viên" in tokens
    assert "12" in tokens
    assert "phép" in tokens


def test_bm25_score_calculation():
    """Kiểm tra tính toán điểm BM25 cho câu chứa từ khóa."""
    query_tokens = ["bảo", "mật"]
    doc_tokens_relevant = ["hướng", "dẫn", "bảo", "mật", "thông", "tin"]
    doc_tokens_irrelevant = ["chính", "sách", "nghỉ", "phép", "năm"]

    score_rel = bm25_score_single(query_tokens, doc_tokens_relevant)
    score_irrel = bm25_score_single(query_tokens, doc_tokens_irrelevant)

    assert score_rel > score_irrel
    assert score_irrel == 0.0


def test_hybrid_score_boundaries():
    """Kiểm tra biên của điểm số hybrid_score."""
    with pytest.raises(ValueError, match="dense_weight"):
        hybrid_score(0.8, 0.5, dense_weight=1.5)

    with pytest.raises(ValueError, match="dense_weight"):
        hybrid_score(0.8, 0.5, dense_weight=-0.1)

    # dense_score = 1.0 (Cosine tối đa) -> Normalized = 1.0
    # lexical_score = 1.0
    assert hybrid_score(1.0, 1.0, dense_weight=0.8) == 1.0


def test_benchmark_contains_independent_dev_and_test_splits():
    """Benchmark phải tách tập chọn tham số khỏi tập báo cáo cuối."""
    cases = load_benchmark()
    assert {case.split for case in cases} == {"dev", "test"}
    assert all(case.question and case.expected_sources for case in cases)


def test_retrieval_metrics_use_first_relevant_rank():
    """MRR sử dụng đúng vị trí đầu tiên chứa một trong các nguồn hợp lệ."""
    metrics = calculate_retrieval_metrics(
        [["wrong.txt", "right.txt"], ["right.txt"]],
        [{"right.txt"}, {"right.txt"}],
        [0.01, 0.02],
    )
    assert metrics["recall_at_k"] == 1.0
    assert metrics["hit_rate_at_1"] == 0.5
    assert metrics["mrr"] == 0.75
