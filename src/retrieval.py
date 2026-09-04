"""Module truy xuất tri thức lai (Hybrid Retriever Engine).

Tệp này chịu trách nhiệm nhận câu hỏi từ người dùng, thực hiện Dense Vector Search trên FAISS Index,
kết hợp Reranking với điểm số Từ khóa (Lexical Overlap & BM25), lọc bớt bằng chứng yếu (min_score),
và trả về danh sách các đoạn văn bản phù hợp nhất kèm trích dẫn nguồn.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .ranking import hybrid_score, lexical_overlap
from .utils import load_json

LOGGER = logging.getLogger("rag_knowledge_assistant.retrieval")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Retriever:
    """Bộ truy xuất tri thức dựa trên chỉ mục FAISS đã huấn luyện và thuật toán Hybrid Search.

    Attributes:
        encoder (SentenceTransformer): Mô hình hóa câu hỏi thành vector.
        index (faiss.Index): Chỉ mục FAISS lưu trữ vector tài liệu.
        chunks (List[Dict[str, Any]]): Danh sách metadata của các chunk.
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        """Khởi tạo lớp Retriever và tải các artifact cần thiết.

        Args:
            model_dir (Optional[Union[str, Path]]): Thư mục chứa chỉ mục và metadata.
                Nếu không chỉ định, mặc định sử dụng 'models/rag_index'.

        Raises:
            FileNotFoundError: Nếu tệp index hoặc metadata không tồn tại.
            ValueError: Nếu số lượng phần tử trong index và metadata không khớp.
        """
        index_dir = Path(model_dir) if model_dir else PROJECT_ROOT / "models/rag_index"

        config_path = index_dir / "config.json"
        faiss_path = index_dir / "index.faiss"
        chunks_path = index_dir / "chunks.json"

        if not (config_path.exists() and faiss_path.exists() and chunks_path.exists()):
            raise FileNotFoundError(
                f"Không tìm thấy đầy đủ tệp artifact chỉ mục tại '{index_dir}'. "
                "Vui lòng chạy 'python -m src.train' trước khi thực hiện truy xuất."
            )

        config_data = load_json(config_path)
        embedding_model_name = config_data.get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

        LOGGER.info("Đang tải mô hình Embedding '%s'...", embedding_model_name)
        self.encoder = SentenceTransformer(embedding_model_name)

        LOGGER.info("Đang tải FAISS Index từ '%s'...", faiss_path.name)
        self.index = faiss.read_index(str(faiss_path))

        self.chunks: list[dict[str, Any]] = load_json(chunks_path)

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Lỗi bất nhất artifact: FAISS index chứa {self.index.ntotal} vectors "
                f"nhưng chunks.json chứa {len(self.chunks)} phần tử."
            )

        LOGGER.info("Retriever sẵn sàng với %d chunks tài liệu.", len(self.chunks))

    def search(
        self,
        query: str,
        k: int = 4,
        min_score: float | None = None,
        dense_weight: float = 0.85,
    ) -> list[dict[str, Any]]:
        """Tìm kiếm top-k đoạn văn bản liên quan nhất cho câu hỏi đầu vào.

        Quy trình xử lý:
        1. Làm sạch truy vấn đầu vào.
        2. Tạo Candidate Pool mở rộng từ FAISS Vector Search.
        3. Tính toán Lexical Score và điểm Hybrid Fusion.
        4. Lọc bỏ các kết quả có điểm dưới min_score (nếu chỉ định).
        5. Sắp xếp lại và trả về k kết quả có điểm số cao nhất.

        Args:
            query (str): Câu hỏi của người dùng.
            k (int): Số lượng kết quả top cần trả về (mặc định: 4).
            min_score (Optional[float]): Ngưỡng điểm tối thiểu để chấp nhận kết quả [0.0 - 1.0].
            dense_weight (float): Trọng số của Dense Retrieval so với Lexical Search [0.0 - 1.0].

        Returns:
            List[Dict[str, Any]]: Danh sách các chunk liên quan nhất kèm thông tin score và nguồn.

        Raises:
            ValueError: Nếu dense_weight nằm ngoài khoảng [0, 1].
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        if not (0.0 <= dense_weight <= 1.0):
            raise ValueError(
                f"dense_weight={dense_weight} phải nằm trong khoảng [0, 1]"
            )

        total_chunks = len(self.chunks)
        effective_k = min(max(1, k), total_chunks)

        # Lexical-only phải chấm toàn bộ corpus. Nếu vẫn lấy candidate từ FAISS,
        # baseline lexical sẽ phụ thuộc dense retrieval và làm sai phép so sánh.
        candidate_k = (
            total_chunks
            if dense_weight == 0.0
            else min(max(effective_k * 4, 20), total_chunks)
        )

        # Biến đổi câu hỏi thành vector embedding
        query_vector = self.encoder.encode([clean_query], normalize_embeddings=True)
        query_arr = np.asarray(query_vector, dtype="float32")

        # Tìm kiếm trên chỉ mục FAISS
        raw_scores, raw_indices = self.index.search(query_arr, candidate_k)

        results: list[dict[str, Any]] = []
        for dense_score_val, idx in zip(raw_scores[0], raw_indices[0], strict=True):
            if idx < 0 or idx >= total_chunks:
                continue

            item = dict(self.chunks[idx])
            dense_score = float(dense_score_val)

            # Tính điểm trùng lặp từ khóa
            lexical_sc = lexical_overlap(clean_query, item["text"])

            # Tính điểm kết hợp Hybrid
            combined_sc = hybrid_score(
                dense_score, lexical_sc, dense_weight=dense_weight
            )

            # Lọc theo min_score nếu được thiết lập
            if min_score is not None and combined_sc < min_score:
                continue

            item["score"] = combined_sc
            item["dense_score"] = dense_score
            item["lexical_score"] = lexical_sc
            results.append(item)

        # Sắp xếp kết quả giảm dần theo điểm số tổng hợp
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return sorted_results[:effective_k]
