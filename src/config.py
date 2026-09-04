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
    """Cấu hình lưu trữ thông số chia nhỏ tài liệu (chunking) và xây dựng FAISS Index.

    Attributes:
        data_dir (str): Thư mục chứa dữ liệu tài liệu đầu vào (TXT, MD, PDF, DOCX).
        model_dir (str): Thư mục lưu trữ artifact (FAISS index, config.json, chunks.json).
        chunk_words (int): Số lượng từ tối đa trong một chunk tài liệu.
        overlap_words (int): Số lượng từ gối đầu (overlap) giữa 2 chunk liền kề.
    """

    data_dir: str = "data/raw"
    model_dir: str = "models/rag_index"
    chunk_words: int = 220
    overlap_words: int = 30

    def validate(self) -> None:
        """Xác thực tính hợp lệ của thông số cấu hình.

        Raises:
            ValueError: Nếu chunk_words <= 0 hoặc overlap_words nằm ngoài khoảng [0, chunk_words).
        """
        if self.chunk_words <= 0:
            raise ValueError("Kích thước chunk (chunk_words) phải lớn hơn 0.")
        if not (0 <= self.overlap_words < self.chunk_words):
            raise ValueError(
                f"Độ chồng lấp (overlap_words={self.overlap_words}) phải thuộc khoảng [0, chunk_words={self.chunk_words})."
            )


def parse_args() -> IndexConfig:
    """Trích xuất tham số từ dòng lệnh (CLI) để khởi tạo IndexConfig.

    Returns:
        IndexConfig: Đối tượng cấu hình đã qua xác thực.
    """
    parser = argparse.ArgumentParser(
        description="Xây dựng FAISS Index và BM25 Metadata cho Trợ lý Tri thức Nội bộ"
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
        default=220,
        help="Kích thước tối đa của mỗi chunk theo số từ (mặc định: 220)",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=30,
        help="Số từ gối đầu giữa các chunk liên tiếp (mặc định: 30)",
    )

    args = parser.parse_args()
    config = IndexConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    config.validate()
    return config
