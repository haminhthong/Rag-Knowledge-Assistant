"""Unit tests cho module ranking, BM25Index, RRF và Cross-Encoder."""

from __future__ import annotations

import pytest
from src.evaluate import calculate_retrieval_metrics, load_benchmark
from src.ranking import (
    BM25Index,
    CrossEncoderReranker,
    bm25_score_single,
    hybrid_score,
    reciprocal_rank_fusion,
    tokenize,
)


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


def test_bm25_index_corpus_search():
    """Kiểm tra BM25Index tìm kiếm chính xác từ khóa trong corpus."""
    docs = [
        "Chính sách nghỉ phép nhân viên 12 ngày một năm",
        "Quy định bật xác thực 2FA cho mọi tài khoản công ty",
        "Hóa đơn đỏ VAT là bắt buộc khi hoàn ứng chi phí công tác",
    ]
    bm25 = BM25Index.from_texts(docs)

    # Tìm kiếm từ khóa chính xác "VAT"
    hits_vat = bm25.search("hóa đơn VAT", top_k=2)
    assert len(hits_vat) > 0
    assert hits_vat[0][0] == 2  # Doc index 2 phải đứng đầu

    # Tìm kiếm "2FA"
    hits_2fa = bm25.search("xác thực 2FA", top_k=2)
    assert len(hits_2fa) > 0
    assert hits_2fa[0][0] == 1


def test_reciprocal_rank_fusion_math():
    """Kiểm tra tính toán điểm Reciprocal Rank Fusion (RRF)."""
    # Doc 0: Rank 1 ở cả Dense và BM25
    # Doc 1: Rank 2 ở Dense, không xuất hiện ở BM25
    # Doc 2: Rank 1 ở BM25, không xuất hiện ở Dense
    dense_ranks = {0: 1, 1: 2}
    bm25_ranks = {0: 1, 2: 1}

    rrf_scores = reciprocal_rank_fusion(dense_ranks, bm25_ranks, k=60)

    # RRF(0) = 1/(60+1) + 1/(60+1) = 2/61 ~ 0.03278
    # RRF(1) = 1/(60+2) = 1/62 ~ 0.01612
    # RRF(2) = 1/(60+1) = 1/61 ~ 0.01639
    assert rrf_scores[0] > rrf_scores[2] > rrf_scores[1]


def test_cross_encoder_reranker_fallback_scoring():
    """Kiểm tra CrossEncoderReranker hoạt động trơn tru ở chế độ fallback."""
    reranker = CrossEncoderReranker(enabled=False)
    candidates = [
        {"text": "Chính sách nghỉ phép 12 ngày", "rrf_score": 0.03},
        {"text": "Bảo mật thông tin và mật khẩu", "rrf_score": 0.01},
    ]
    reranked = reranker.rerank("nghỉ phép", candidates, top_k=2)

    assert len(reranked) == 2
    assert "rerank_score" in reranked[0]
    assert "retrieval_score" in reranked[0]
    assert reranked[0]["text"].startswith("Chính sách nghỉ phép")


def test_hybrid_score_boundaries():
    """Kiểm tra biên của điểm số hybrid_score."""
    with pytest.raises(ValueError, match="dense_weight"):
        hybrid_score(0.8, 0.5, dense_weight=1.5)

    with pytest.raises(ValueError, match="dense_weight"):
        hybrid_score(0.8, 0.5, dense_weight=-0.1)

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
    assert metrics["ndcg_at_k"] > 0.0
