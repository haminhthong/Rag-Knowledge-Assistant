"""Unit test cho chức năng chia nhỏ đoạn văn (Chunking) và xác thực cấu hình."""

from __future__ import annotations

import pytest
from src.config import IndexConfig
from src.ingestion import chunk_text
from src.ranking import hybrid_score, lexical_overlap


def test_chunking_keeps_source_and_metadata():
    """Kiểm tra việc chia chunk bảo toàn đúng metadata nguồn và độ dài."""
    text = " ".join(["từ_mẫu"] * 600)
    chunks = chunk_text(
        text, source="doc_test.txt", page=1, chunk_words=100, overlap_words=10
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.source == "doc_test.txt"
        assert chunk.page == 1
        assert chunk.word_count > 0


def test_chunking_rejects_invalid_overlap():
    """Kiểm tra ngoại lệ khi tham số overlap không hợp lệ."""
    with pytest.raises(ValueError, match="overlap_words"):
        chunk_text(
            "Nội dung thử nghiệm", source="test.txt", chunk_words=10, overlap_words=10
        )


def test_index_config_validation():
    """Kiểm tra việc validate thông số IndexConfig."""
    with pytest.raises(ValueError, match="overlap_words"):
        IndexConfig(chunk_words=100, overlap_words=100).validate()

    with pytest.raises(ValueError, match="chunk_words"):
        IndexConfig(chunk_words=0, overlap_words=0).validate()


def test_hybrid_ranking_rewards_exact_query_terms():
    """Kiểm tra thuật toán Lexical Overlap ưu tiên các đoạn chứa chính xác từ khóa."""
    query = "báo sự cố bảo mật"
    doc_match = "Nhân viên phải báo sự cố bảo mật trong vòng 30 phút"
    doc_unrelated = "Quy trình thanh toán và nộp hoàn ứng chi phí công tác"

    exact_score = lexical_overlap(query, doc_match)
    unrelated_score = lexical_overlap(query, doc_unrelated)

    assert exact_score > unrelated_score
    assert hybrid_score(0.5, exact_score, dense_weight=0.8) > hybrid_score(
        0.5, unrelated_score, dense_weight=0.8
    )
