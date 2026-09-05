"""Cấu hình và tham số hệ thống cho RAG Knowledge Assistant.

Tệp này định nghĩa các lớp cấu hình nhẹ và trình xử lý tham số dòng lệnh (CLI).
Các lớp cấu hình được tối ưu để không tải các mô hình ML nặng khi chỉ cần
kiểm tra hoặc khởi tạo tham số.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexConfig:
    """Cấu hình lưu trữ thông số chia nhỏ tài liệu (chunking) và xây dựng FAISS Index & BM25 Index.

    Attributes:
        data_dir (str): Thư mục chứa dữ liệu tài liệu đầu vào (TXT, MD, PDF, DOCX).
        model_dir (str): Thư mục lưu trữ artifact (FAISS index, config.json, chunks.json, bm25_index.json).
        chunk_words (int): Số lượng từ mục tiêu trong một chunk tài liệu.
        overlap_words (int): Số lượng từ gối đầu (overlap) giữa 2 chunk liền kề.
        strategy (str): Chiến lược chunking ("structure_aware" hoặc "sliding_window").
        embedding_model (str): Tên mô hình SentenceTransformers để vectorize.
        reranker_model (str): Tên mô hình Cross-Encoder reranker.
        candidate_pool_k (int): Số lượng candidate pool trích xuất từ mỗi nhánh (Dense & BM25).
        rrf_k (int): Hằng số điều chỉnh Reciprocal Rank Fusion.
        rerank_top_k (int): Số lượng candidate giữ lại sau reranking.
        evidence_gate_threshold (float): Ngưỡng điểm chấp nhận bằng chứng trước khi chuyển LLM.
        use_reranker (bool): Có kích hoạt mô hình neural Cross-Encoder hay không.
    """

    data_dir: str = "data/raw"
    model_dir: str = "models/rag_index"
    chunk_words: int = 250
    overlap_words: int = 40
    strategy: str = "structure_aware"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_pool_k: int = 30
    rrf_k: int = 60
    rerank_top_k: int = 5
    evidence_gate_threshold: float = 0.25
    use_reranker: bool = True

    def validate(self) -> None:
        """Xác thực tính hợp lệ của thông số cấu hình.

        Raises:
            ValueError: Nếu các tham số vi phạm ràng buộc kỹ thuật.
        """
        if self.chunk_words <= 0:
            raise ValueError("Kích thước chunk (chunk_words) phải lớn hơn 0.")
        if not (0 <= self.overlap_words < self.chunk_words):
            raise ValueError(
                f"Độ chồng lấp (overlap_words={self.overlap_words}) phải thuộc khoảng [0, chunk_words={self.chunk_words})."
            )
        if self.strategy not in {"structure_aware", "sliding_window"}:
            raise ValueError(
                f"Chiến lược chunking không hợp lệ: '{self.strategy}'. Chọn 'structure_aware' hoặc 'sliding_window'."
            )
        if self.candidate_pool_k <= 0:
            raise ValueError("candidate_pool_k phải lớn hơn 0.")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k phải lớn hơn 0.")
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k phải lớn hơn 0.")
        if not (0.0 <= self.evidence_gate_threshold <= 1.0):
            raise ValueError("evidence_gate_threshold phải nằm trong đoạn [0.0, 1.0].")


def parse_args() -> IndexConfig:
    """Trích xuất tham số từ dòng lệnh (CLI) để khởi tạo IndexConfig.

    Returns:
        IndexConfig: Đối tượng cấu hình đã qua xác thực.
    """
    parser = argparse.ArgumentParser(
        description="Xây dựng FAISS Index và BM25 Index cho Vietnamese Evidence-Grounded RAG Assistant"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Đường dẫn thư mục chứa tài liệu gốc (mặc định: data/raw)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/rag_index",
        help="Đường dẫn thư mục lưu trữ index và artifact (mặc định: models/rag_index)",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=250,
        help="Kích thước mục tiêu của mỗi chunk theo số từ (mặc định: 250)",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=40,
        help="Số từ gối đầu giữa các chunk liên tiếp (mặc định: 40)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["structure_aware", "sliding_window"],
        default="structure_aware",
        help="Chiến lược phân tách đoạn văn (mặc định: structure_aware)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Tên hoặc đường dẫn mô hình embedding",
    )
    parser.add_argument(
        "--candidate-pool-k",
        type=int,
        default=30,
        help="Kích thước candidate pool trích xuất từ mỗi nhánh (mặc định: 30)",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Tắt mô hình neural Cross-Encoder để tối ưu latency",
    )

    args = parser.parse_args()
    config = IndexConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        strategy=args.strategy,
        embedding_model=args.embedding_model,
        candidate_pool_k=args.candidate_pool_k,
        use_reranker=not args.no_reranker,
    )
    config.validate()
    return config

