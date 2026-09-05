"""Smoke test kiểm tra import module, health check endpoints và citation validation."""

from __future__ import annotations

import json

from src.api import INDEX_DIR, health, health_live
from src.generation import validate_citations


def test_health_reports_sanitized_status():
    """Kiểm tra hàm health() trả về trạng thái hợp lệ và KHÔNG làm lộ đường dẫn ổ đĩa."""
    response = health()
    assert response["status"] in {"ok", "degraded"}
    assert isinstance(response["index_ready"], bool)
    # Bảo mật: Tuyệt đối không để lộ artifacts_directory ra ngoài client
    assert "artifacts_directory" not in response


def test_health_live_endpoint():
    """Kiểm tra liveness endpoint."""
    res = health_live()
    assert res["status"] == "alive"
    assert "timestamp" in res


def test_index_config_has_reproducibility_metadata():
    """Kiểm tra tệp config.json nếu index đã được xây dựng."""
    config_file = INDEX_DIR / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        assert config["schema_version"] in {1, 2}
        assert config["chunk_count"] >= 0
        assert config["embedding_model"]
        assert 0 <= config["overlap_words"] < config["chunk_words"]


def test_citation_validation_logic():
    """Kiểm tra CitationValidator bóc tách đúng thẻ [C1] và gắn trích đoạn minh chứng."""
    hits = [
        {"source": "policy_leave.txt", "page": None, "section": "Nghỉ phép", "text": "12 ngày phép năm"},
        {"source": "security_guide.txt", "page": 1, "section": "Bảo mật", "text": "Bắt buộc bật 2FA"},
    ]
    answer = "Nhân viên có 12 ngày phép [C1] và bắt buộc 2FA [C2]."

    citations, is_clean = validate_citations(answer, hits)
    assert is_clean is True
    assert len(citations) == 2
    assert citations[0]["id"] == "C1"
    assert citations[0]["document"] == "policy_leave.txt"
    assert citations[1]["id"] == "C2"
    assert citations[1]["document"] == "security_guide.txt"

    # Trường hợp câu trả lời sinh ra thẻ ảo [C9] không có trong hits
    bad_answer = "Thông tin bảo mật [C9]."
    bad_citations, is_clean_bad = validate_citations(bad_answer, hits)
    assert is_clean_bad is False
