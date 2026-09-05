"""Dịch vụ HTTP REST API cho Vietnamese Evidence-Grounded Knowledge Assistant (FastAPI Service).

Cung cấp các endpoint:
- /health: Kiểm tra tổng quát trạng thái hệ thống (đã làm sạch, không để lộ filesystem path).
- /health/live: Liveness probe cho orchestrator / container.
- /health/ready: Readiness probe kiểm tra tính toàn vẹn và đồng bộ của FAISS & BM25 Artifacts.
- /query: Endpoint hỏi đáp tri thức nội bộ với bằng chứng phân tầng, trích dẫn [C1], [C2],
  và cổng Evidence Quality Gate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generation import ABSTAIN_PHRASE, generate_grounded_response
from .utils import load_json, setup_logging

if TYPE_CHECKING:
    from .retrieval import Retriever

# Thiết lập logging cho API
setup_logging()
LOGGER = logging.getLogger("rag_knowledge_assistant.api")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "models/rag_index"

app = FastAPI(
    title="Vietnamese Evidence-Grounded Knowledge Assistant (Hybrid RAG)",
    description=(
        "REST API tra cứu tri thức nội bộ tiếng Việt chính xác cao, kết hợp Dense FAISS, "
        "BM25Okapi, RRF, Cross-Encoder Reranking, Evidence Quality Gate và Claim-Level Citations."
    ),
    version="2.0.0",
)

# Thể hiện đơn lẻ (Singleton) của Retriever được nạp theo cơ chế Lazy Loading
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Khởi tạo hoặc trả về thể hiện Singleton của Retriever."""
    global _retriever
    if _retriever is None:
        LOGGER.info("Khởi tạo Retriever lần đầu tiên (Lazy Loading)...")
        from .retrieval import Retriever

        _retriever = Retriever(model_dir=INDEX_DIR)
    return _retriever


class QueryIn(BaseModel):
    """Payload đầu vào cho yêu cầu hỏi đáp /query."""

    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Câu hỏi bằng tiếng Việt cần tra cứu tri thức",
        json_schema_extra={"example": "Nhân viên chính thức có bao nhiêu ngày phép năm?"},
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Số lượng đoạn văn bản trích dẫn tối đa đưa vào ngữ cảnh",
    )
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Ngưỡng điểm tương đồng tối thiểu để nhận kết quả (tùy chọn)",
    )
    dense_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Trọng số ép nhánh tìm kiếm (None: Dual Search + RRF; 1.0: Dense-only; 0.0: BM25-only)",
    )
    use_reranker: bool = Field(
        default=True,
        description="Kích hoạt mô hình Cross-Encoder để xếp hạng lại",
    )


class CitationItem(BaseModel):
    """Chi tiết trích dẫn minh chứng cấp độ khẳng định (Claim-Level Citation)."""

    id: str = Field(..., description="Mã trích dẫn (ví dụ: C1, C2)")
    document: str = Field(..., description="Tên tệp tài liệu gốc")
    source_path: str = Field(..., description="Đường dẫn tương đối của tài liệu")
    page: int | None = Field(None, description="Số trang tương ứng (khi khả dụng)")
    section: str | None = Field(None, description="Tên phần/chương/mục chứa trích dẫn")
    chunk_id: str = Field(..., description="Mã định danh duy nhất của chunk")
    quote: str = Field(..., description="Trích đoạn ngắn minh chứng sự thật")


class SourceItem(BaseModel):
    """Thông tin chi tiết của từng đoạn tài liệu nguồn thu thập được."""

    chunk_id: str = Field(..., description="Mã định danh duy nhất của chunk")
    document_id: str = Field(..., description="Mã hash định danh tài liệu")
    source: str = Field(..., description="Tên tệp tài liệu gốc")
    source_path: str = Field(..., description="Đường dẫn tương đối của tệp")
    page: int | None = Field(None, description="Số trang tương ứng (khi có)")
    section: str | None = Field(None, description="Tiêu đề mục chứa chunk")
    retrieval_score: float = Field(
        ..., description="Điểm xếp hạng độ liên quan chuẩn hóa [0.0 - 1.0]"
    )
    score: float = Field(..., description="Điểm số tương thích ngược")
    dense_score: float = Field(..., description="Điểm tương đồng Cosine ngữ nghĩa")
    bm25_score: float = Field(..., description="Điểm từ khóa BM25")
    rerank_score: float = Field(..., description="Điểm sau Cross-Encoder Reranking")


class QueryOut(BaseModel):
    """Payload đầu ra chuẩn hóa cho câu trả lời và trích dẫn."""

    answer: str = Field(
        ..., description="Câu trả lời tổng hợp căn thực hoặc thông báo từ chối"
    )
    citations: list[CitationItem] = Field(
        default_factory=list,
        description="Danh sách các trích dẫn có cấu trúc [C1], [C2]",
    )
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="Danh sách toàn bộ các nguồn tài liệu ứng viên được duyệt",
    )
    model_version: str = Field(..., description="Phiên bản mô hình/index đang sử dụng")
    index_version: str = Field(..., description="Mã phiên bản chỉ mục thời gian thực")
    evidence_gate_passed: bool = Field(
        ..., description="Cờ xác nhận bằng chứng có vượt qua Evidence Quality Gate không"
    )


@app.get(
    "/health",
    summary="Kiểm tra tổng quát sức khỏe dịch vụ và chỉ mục",
    response_model=dict[str, Any],
)
def health() -> dict[str, Any]:
    """Endpoint kiểm tra trạng thái hoạt động của hệ thống, không làm lộ đường dẫn máy chủ."""
    required_files = ("config.json", "index.faiss", "chunks.json")
    ready = all((INDEX_DIR / filename).exists() for filename in required_files)

    model_version = "not_trained"
    index_version = "none"
    chunk_count = 0

    if ready:
        try:
            config_data = load_json(INDEX_DIR / "config.json")
            model_version = config_data.get("model_version", "rag-evidence-v2")
            index_version = config_data.get("index_version", "unknown")
            chunk_count = config_data.get("chunk_count", 0)
        except Exception:  # noqa: BLE001
            ready = False

    return {
        "status": "ok" if ready else "degraded",
        "index_ready": ready,
        "model_version": model_version,
        "index_version": index_version,
        "chunk_count": chunk_count,
    }


@app.get(
    "/health/live",
    summary="Liveness probe cho orchestrator",
    response_model=dict[str, Any],
)
def health_live() -> dict[str, Any]:
    """Kiểm tra tiến trình API đang hoạt động bình thường."""
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get(
    "/health/ready",
    summary="Readiness probe kiểm tra tính sẵn sàng của toàn bộ artifacts",
    response_model=dict[str, Any],
)
def health_ready() -> dict[str, Any]:
    """Kiểm tra chuyên sâu: Sự tồn tại và tính đồng bộ giữa FAISS index và chunks.json."""
    required_files = ("config.json", "index.faiss", "chunks.json", "bm25_index.json")
    missing = [f for f in required_files if not (INDEX_DIR / f).exists()]

    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Hệ thống chưa sẵn sàng. Thiếu các artifact: {', '.join(missing)}",
        )

    try:
        config_data = load_json(INDEX_DIR / "config.json")
        chunks_data = load_json(INDEX_DIR / "chunks.json")
        retriever_inst = get_retriever()

        if retriever_inst.index.ntotal != len(chunks_data):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Bất nhất artifact: FAISS ({retriever_inst.index.ntotal}) != "
                    f"chunks.json ({len(chunks_data)})"
                ),
            )

        return {
            "status": "ready",
            "model_version": config_data.get("model_version", "rag-evidence-v2"),
            "index_version": config_data.get("index_version", "unknown"),
            "vector_dimension": config_data.get("vector_dimension"),
            "total_chunks": len(chunks_data),
            "reranker_ready": retriever_inst.reranker.enabled,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Lỗi khi kiểm tra tính sẵn sàng: {exc}",
        ) from exc


@app.post(
    "/query",
    summary="Gửi câu hỏi tra cứu tri thức nội bộ",
    response_model=QueryOut,
)
def query(payload: QueryIn) -> QueryOut:
    """Endpoint xử lý câu hỏi tra cứu tri thức.

    Thực hiện truy xuất văn bản Canonical Hybrid Search (Dense + BM25 + RRF + Reranker),
    kiểm định Evidence Gate, và sinh câu trả lời căn thực kèm trích dẫn có cấu trúc.
    """
    try:
        retriever_inst = get_retriever()
        hits = retriever_inst.search(
            query=payload.question,
            k=payload.top_k,
            min_score=payload.min_score,
            dense_weight=payload.dense_weight,
            use_reranker=payload.use_reranker,
        )
    except (OSError, ValueError, KeyError, FileNotFoundError) as exc:
        LOGGER.error("Lỗi khi tải hoặc tìm kiếm trên chỉ mục: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Hệ thống chỉ mục chưa sẵn sàng hoặc artifact bị lỗi. Vui lòng chạy 'python -m src.train' trước.",
        ) from exc

    model_ver = retriever_inst.config.get("model_version", "rag-evidence-v2")
    index_ver = retriever_inst.config.get("index_version", "unknown")

    # Xử lý trường hợp không tìm thấy bất kỳ chunk nào hoặc bị lọc toàn bộ
    if not hits:
        return QueryOut(
            answer=ABSTAIN_PHRASE,
            citations=[],
            sources=[],
            model_version=model_ver,
            index_version=index_ver,
            evidence_gate_passed=False,
        )

    # Sinh câu trả lời căn thực và trích xuất danh sách trích dẫn
    answer_text, citations_data, gate_passed = generate_grounded_response(
        payload.question, hits, max_chunks=payload.top_k
    )

    formatted_citations = [
        CitationItem(
            id=c.get("id", "C1"),
            document=c.get("document", "unknown"),
            source_path=c.get("source_path", c.get("document", "unknown")),
            page=c.get("page"),
            section=c.get("section"),
            chunk_id=c.get("chunk_id", ""),
            quote=c.get("quote", ""),
        )
        for c in citations_data
    ]

    formatted_sources = [
        SourceItem(
            chunk_id=hit.get("chunk_id", ""),
            document_id=hit.get("document_id", ""),
            source=hit.get("source", "unknown"),
            source_path=hit.get("source_path", hit.get("source", "unknown")),
            page=hit.get("page"),
            section=hit.get("section"),
            retrieval_score=round(float(hit.get("retrieval_score", 0.0)), 4),
            score=round(float(hit.get("score", 0.0)), 4),
            dense_score=round(float(hit.get("dense_score", 0.0)), 4),
            bm25_score=round(float(hit.get("bm25_score", 0.0)), 4),
            rerank_score=round(float(hit.get("rerank_score", 0.0)), 4),
        )
        for hit in hits
    ]

    return QueryOut(
        answer=answer_text,
        citations=formatted_citations,
        sources=formatted_sources,
        model_version=model_ver,
        index_version=index_ver,
        evidence_gate_passed=gate_passed,
    )
