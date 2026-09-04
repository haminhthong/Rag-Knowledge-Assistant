"""Dịch vụ HTTP REST API cho Trợ lý Tra cứu Tri thức Nội bộ (FastAPI Service).

Cung cấp các endpoint kiểm tra sức khỏe hệ thống (/health) và endpoint gửi câu hỏi (/query).
Tích hợp kiểm tra tính sẵn sàng của chỉ mục (Index Readiness) và chuẩn hóa payload bằng Pydantic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generation import generate_answer
from .utils import setup_logging

if TYPE_CHECKING:
    from .retrieval import Retriever

# Thiết lập logging cho API
setup_logging()
LOGGER = logging.getLogger("rag_knowledge_assistant.api")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "models/rag_index"
MODEL_VERSION = "rag-faiss-v1"

app = FastAPI(
    title="Enterprise Knowledge Assistant (RAG)",
    description="REST API tra cứu tri thức nội bộ bằng tiếng Việt kèm trích dẫn nguồn và số trang.",
    version="1.0.0",
)

# Thể hiện đơn lẻ (Singleton) của Retriever được tải theo cơ chế Lazy Loading
_retriever: Retriever | None = None


class QueryIn(BaseModel):
    """Payload đầu vào cho yêu cầu hỏi đáp /query."""

    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Câu hỏi bằng tiếng Việt cần tra cứu tri thức",
        json_schema_extra={"example": "Nhân viên có bao nhiêu ngày phép năm?"},
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Số lượng đoạn văn bản trích dẫn tối đa",
    )
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Ngưỡng điểm tương đồng tối thiểu để nhận kết quả (tùy chọn)",
    )
    dense_weight: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Tỷ trọng giữa Dense Retrieval và Lexical Search (0.0: Lexical, 1.0: Dense)",
    )


class SourceItem(BaseModel):
    """Thông tin nguồn trích dẫn tài liệu."""

    source: str = Field(..., description="Tên tệp tài liệu gốc")
    page: int | None = Field(None, description="Số trang tương ứng (nếu có)")
    score: float = Field(..., description="Điểm tổng hợp Hybrid Score")
    dense_score: float = Field(..., description="Điểm Cosine Dense Similarity")
    lexical_score: float = Field(
        ..., description="Điểm trùng lặp từ khóa Lexical Score"
    )


class QueryOut(BaseModel):
    """Payload đầu ra cho câu trả lời và thông tin trích dẫn."""

    answer: str = Field(
        ..., description="Câu trả lời tổng hợp hoặc danh sách context trích dẫn"
    )
    sources: list[SourceItem] = Field(
        ..., description="Danh sách các nguồn tài liệu tham khảo"
    )
    model_version: str = Field(..., description="Phiên bản mô hình/index được sử dụng")


@app.get(
    "/health",
    summary="Kiểm tra sức khỏe dịch vụ và chỉ mục FAISS",
    response_model=dict[str, Any],
)
def health() -> dict[str, Any]:
    """Endpoint kiểm tra trạng thái hoạt động của hệ thống API và sự tồn tại của FAISS Index."""
    required_files = ("config.json", "index.faiss", "chunks.json")
    ready = all((INDEX_DIR / filename).exists() for filename in required_files)

    return {
        "status": "ok" if ready else "degraded",
        "index_ready": ready,
        "model_version": MODEL_VERSION if ready else "not_trained",
        "artifacts_directory": str(INDEX_DIR),
    }


@app.post(
    "/query",
    summary="Gửi câu hỏi tra cứu tri thức nội bộ",
    response_model=QueryOut,
)
def query(payload: QueryIn) -> QueryOut:
    """Endpoint xử lý câu hỏi tra cứu tri thức.

    Thực hiện truy xuất văn bản bằng Hybrid Search và sinh câu trả lời căn thực.
    """
    global _retriever
    try:
        if _retriever is None:
            LOGGER.info("Khởi tạo Retriever lần đầu tiên (Lazy Loading)...")
            from .retrieval import Retriever

            _retriever = Retriever(model_dir=INDEX_DIR)

        hits = _retriever.search(
            query=payload.question,
            k=payload.top_k,
            min_score=payload.min_score,
            dense_weight=payload.dense_weight,
        )
    except (OSError, ValueError, KeyError, FileNotFoundError) as exc:
        LOGGER.error("Lỗi khi tải hoặc tìm kiếm trên chỉ mục FAISS: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Hệ thống chỉ mục FAISS chưa sẵn sàng hoặc artifact bị lỗi. Vui lòng chạy 'python -m src.train' trước.",
        ) from exc

    if not hits:
        return QueryOut(
            answer="Không đủ thông tin trong tài liệu nội bộ để trả lời câu hỏi này.",
            sources=[],
            model_version=MODEL_VERSION,
        )

    # Sinh câu trả lời dựa trên context thu được
    answer_text = generate_answer(payload.question, hits)

    # Đóng gói danh sách nguồn trích dẫn
    formatted_sources = [
        SourceItem(
            source=hit.get("source", "unknown"),
            page=hit.get("page"),
            score=round(hit.get("score", 0.0), 4),
            dense_score=round(hit.get("dense_score", 0.0), 4),
            lexical_score=round(hit.get("lexical_score", 0.0), 4),
        )
        for hit in hits
    ]

    return QueryOut(
        answer=answer_text,
        sources=formatted_sources,
        model_version=MODEL_VERSION,
    )
