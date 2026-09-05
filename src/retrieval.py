"""Module truy xuất tri thức lai chính quy (Canonical Hybrid Retrieval & Reranking Engine).

Tệp này thực hiện đầy đủ quy trình Canonical Online RAG:
1. Chuẩn hóa truy vấn (Query Normalization).
2. Tìm kiếm ứng viên kép độc lập (Dual Retrieval):
   - Dense Vector Search (FAISS IndexFlatIP Top-N).
   - BM25 Lexical Search (BM25Okapi Top-N).
3. Hợp nhất tập ứng viên (Candidate Union).
4. Dung hợp thứ hạng Reciprocal Rank Fusion (RRF).
5. Xếp hạng lại bằng Cross-Encoder Reranker.
6. Kiểm định chất lượng bằng chứng (Evidence Quality Gate) trước khi chuyển LLM.
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .ranking import BM25Index, CrossEncoderReranker, reciprocal_rank_fusion
from .utils import load_json

LOGGER = logging.getLogger("rag_knowledge_assistant.retrieval")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_query(query: str) -> str:
    """Chuẩn hóa câu hỏi truy vấn của người dùng.

    Thực hiện:
    - Loại bỏ khoảng trắng thừa đầu cuối và giữa các từ.
    - Chuẩn hóa Unicode dạng NFKC (tương thích các biến thể gõ tiếng Việt).

    Args:
        query (str): Câu hỏi thô từ người dùng.

    Returns:
        str: Chuỗi câu hỏi đã được làm sạch và chuẩn hóa.
    """
    normalized = unicodedata.normalize("NFKC", query.strip())
    # Thu gọn khoảng trắng liên tiếp
    return " ".join(normalized.split())


class Retriever:
    """Bộ truy xuất tri thức lai kết hợp FAISS Dense Search, BM25 Index, RRF và Cross-Encoder.

    Attributes:
        encoder (SentenceTransformer): Mô hình sinh vector biểu diễn câu hỏi.
        index (faiss.Index): Chỉ mục FAISS lưu trữ vector tài liệu.
        chunks (List[Dict[str, Any]]): Danh sách metadata của các chunk.
        bm25_index (BM25Index): Chỉ mục từ khóa BM25Okapi.
        reranker (CrossEncoderReranker): Mô hình chấm điểm chéo rerank.
        config (Dict[str, Any]): Cấu hình chỉ mục được tải từ config.json.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        use_reranker: bool = True,
    ) -> None:
        """Khởi tạo lớp Retriever và nạp toàn bộ artifact cần thiết.

        Args:
            model_dir (Optional[Union[str, Path]]): Thư mục chứa chỉ mục và metadata.
                Nếu không chỉ định, mặc định sử dụng 'models/rag_index'.
            use_reranker (bool): Có sử dụng Cross-Encoder reranker hay không (mặc định: True).

        Raises:
            FileNotFoundError: Nếu tệp index hoặc metadata không tồn tại.
            ValueError: Nếu số lượng vector trong FAISS và chunks.json không khớp.
        """
        index_dir = Path(model_dir) if model_dir else PROJECT_ROOT / "models/rag_index"

        config_path = index_dir / "config.json"
        faiss_path = index_dir / "index.faiss"
        chunks_path = index_dir / "chunks.json"
        bm25_path = index_dir / "bm25_index.json"

        if not (config_path.exists() and faiss_path.exists() and chunks_path.exists()):
            raise FileNotFoundError(
                f"Không tìm thấy đầy đủ tệp artifact chỉ mục tại '{index_dir}'. "
                "Vui lòng chạy 'python -m src.train' trước khi thực hiện truy xuất."
            )

        self.config: dict[str, Any] = load_json(config_path)
        embedding_model_name = self.config.get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        reranker_model_name = self.config.get(
            "reranker_model",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        self.evidence_gate_threshold: float = float(
            self.config.get("evidence_gate_threshold", 0.25)
        )
        self.default_candidate_k: int = int(self.config.get("candidate_pool_k", 30))
        self.rrf_k: int = int(self.config.get("rrf_k", 60))

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

        # Khởi tạo hoặc nạp BM25 Index
        if bm25_path.exists():
            LOGGER.info("Đang tải BM25 Index từ '%s'...", bm25_path.name)
            bm25_data = load_json(bm25_path)
            self.bm25_index = BM25Index.from_dict(bm25_data)
        else:
            LOGGER.info("Tự động xây dựng BM25 Index trong bộ nhớ...")
            self.bm25_index = BM25Index.from_texts([c["text"] for c in self.chunks])

        # Khởi tạo Cross-Encoder Reranker
        self.reranker = CrossEncoderReranker(
            model_name=reranker_model_name,
            enabled=use_reranker,
        )

        LOGGER.info(
            "Canonical Retriever sẵn sàng với %d chunks (Dense + BM25 + RRF + Reranker).",
            len(self.chunks),
        )

    def search(
        self,
        query: str,
        k: int = 4,
        candidate_k: int | None = None,
        min_score: float | None = None,
        dense_weight: float | None = None,
        use_reranker: bool = True,
    ) -> list[dict[str, Any]]:
        """Tìm kiếm các đoạn văn bản liên quan nhất theo chuẩn Canonical Hybrid RAG Pipeline.

        Quy trình:
        1. Chuẩn hóa Query.
        2. Nhánh 1: Dense Search trên FAISS -> Top candidate_k.
        3. Nhánh 2: BM25 Lexical Search trên BM25Index -> Top candidate_k.
        4. Candidate Union: Hợp nhất toàn bộ ứng viên độc nhất từ hai nhánh.
        5. Reciprocal Rank Fusion (RRF, k=60) -> Chọn Top 20 ứng viên tốt nhất.
        6. Cross-Encoder Reranking: Chấm điểm tương tác chéo (query, chunk) -> Top k.
        7. Evidence Quality Gate: Đánh giá xem bằng chứng có vượt ngưỡng tin cậy không.

        Args:
            query (str): Câu hỏi của người dùng.
            k (int): Số lượng kết quả top cần trả về (mặc định: 4).
            candidate_k (Optional[int]): Kích thước pool ứng viên mỗi nhánh (mặc định: 30).
            min_score (Optional[float]): Ngưỡng điểm tối thiểu bắt buộc của kết quả.
            dense_weight (Optional[float]): Tham số hỗ trợ tương thích ngược (nếu =1.0 chỉ dùng Dense, nếu =0.0 chỉ dùng BM25).
            use_reranker (bool): Kích hoạt Cross-Encoder reranker.

        Returns:
            List[Dict[str, Any]]: Danh sách các chunk liên quan nhất kèm thông tin score và gate_passed.
        """
        clean_query = normalize_query(query)
        if not clean_query:
            return []

        total_chunks = len(self.chunks)
        effective_k = min(max(1, k), total_chunks)
        pool_size = min(candidate_k or self.default_candidate_k, total_chunks)

        dense_ranks: dict[int, int] = {}
        dense_scores_map: dict[int, float] = {}

        bm25_ranks: dict[int, int] = {}
        bm25_scores_map: dict[int, float] = {}

        # 1. Nhánh Dense Vector Search (trừ khi người dùng ép chỉ dùng lexical: dense_weight == 0.0)
        if dense_weight is None or dense_weight > 0.0:
            query_vector = self.encoder.encode([clean_query], normalize_embeddings=True)
            query_arr = np.asarray(query_vector, dtype="float32")
            raw_scores, raw_indices = self.index.search(query_arr, pool_size)

            for rank_idx, (sc, idx) in enumerate(
                zip(raw_scores[0], raw_indices[0], strict=True), start=1
            ):
                if 0 <= idx < total_chunks:
                    dense_ranks[int(idx)] = rank_idx
                    dense_scores_map[int(idx)] = float(sc)

        # 2. Nhánh BM25 Lexical Search (trừ khi người dùng ép chỉ dùng dense: dense_weight == 1.0)
        if dense_weight is None or dense_weight < 1.0:
            bm25_results = self.bm25_index.search(clean_query, top_k=pool_size)
            for rank_idx, (idx, bm_score) in enumerate(bm25_results, start=1):
                bm25_ranks[idx] = rank_idx
                bm25_scores_map[idx] = float(bm_score)

        # 3. Candidate Union: Hợp nhất toàn bộ ứng viên tìm được
        union_indices = list(set(dense_ranks.keys()) | set(bm25_ranks.keys()))
        if not union_indices:
            return []

        # 4. Reciprocal Rank Fusion (RRF)
        rrf_scores = reciprocal_rank_fusion(dense_ranks, bm25_ranks, k=self.rrf_k)

        candidate_items: list[dict[str, Any]] = []
        for idx in union_indices:
            item = dict(self.chunks[idx])
            d_sc = dense_scores_map.get(idx, 0.0)
            b_sc = bm25_scores_map.get(idx, 0.0)
            r_sc = rrf_scores.get(idx, 0.0)

            item["dense_score"] = round(d_sc, 4)
            item["bm25_score"] = round(b_sc, 4)
            item["rrf_score"] = round(r_sc, 6)
            item["dense_rank"] = dense_ranks.get(idx)
            item["bm25_rank"] = bm25_ranks.get(idx)
            item["chunk_index_in_corpus"] = idx
            candidate_items.append(item)

        # Sắp xếp danh sách ứng viên theo RRF score giảm dần
        candidate_items.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Chọn Top 20 ứng viên tốt nhất đi vào Cross-Encoder Reranker
        top_rrf_pool = candidate_items[: min(20, len(candidate_items))]

        # 5. Cross-Encoder Reranker
        if use_reranker:
            reranked = self.reranker.rerank(clean_query, top_rrf_pool, top_k=effective_k)
        else:
            # Nếu không dùng reranker, điểm retrieval_score chính là chuẩn hóa của RRF
            for item in top_rrf_pool:
                item["retrieval_score"] = round(item["rrf_score"] * 30.0, 4)
                item["rerank_score"] = item["retrieval_score"]
            reranked = top_rrf_pool[:effective_k]

        # 6. Evidence Quality Gate
        gate_threshold = min_score if min_score is not None else self.evidence_gate_threshold
        final_results: list[dict[str, Any]] = []

        for item in reranked:
            score = item.get("retrieval_score", 0.0)
            item["score"] = score  # Giữ trường score cho tương thích ngược
            item["lexical_score"] = (
                item["bm25_score"] / 20.0 if item["bm25_score"] > 0 else 0.0
            )

            # Đánh giá xem chunk có vượt qua Evidence Gate không
            is_valid_evidence = score >= gate_threshold
            item["gate_passed"] = is_valid_evidence

            if min_score is not None and not is_valid_evidence:
                continue

            final_results.append(item)

        return final_results
