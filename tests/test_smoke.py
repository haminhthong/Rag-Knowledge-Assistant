"""Smoke test kiểm tra import module và endpoint kiểm tra sức khỏe (/health)."""

from __future__ import annotations

import json

from src.api import INDEX_DIR, health


def test_health_reports_index_readiness():
    """Kiểm tra hàm health() trả về trạng thái hợp lệ."""
    response = health()
    assert response["status"] in {"ok", "degraded"}
    assert isinstance(response["index_ready"], bool)


def test_index_config_has_reproducibility_metadata():
    """Kiểm tra tệp config.json nếu index đã được xây dựng."""
    config_file = INDEX_DIR / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        assert config["schema_version"] == 1
        assert config["chunk_count"] >= 0
        assert config["embedding_model"]
        assert 0 <= config["overlap_words"] < config["chunk_words"]
