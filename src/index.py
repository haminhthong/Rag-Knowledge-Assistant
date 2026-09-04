"""Xây dựng và Lưu trữ chỉ mục FAISS Index (Indexing Module).

Tệp này quản lý quá trình vectorize văn bản bằng SentenceTransformer,
tạo chỉ mục tìm kiếm tích trong (Inner Product - tương đương Cosine Similarity khi vector được chuẩn hóa),
và lưu trữ các artifact phục vụ quy trình tra cứu (Retrieval).
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import IndexConfig
from .ingestion import ingest_folder
from .utils import save_json, setup_logging

LOGGER = logging.getLogger("rag_knowledge_assistant.index")

# Mô hình Embedding đa ngôn ngữ mặc định hỗ trợ tiếng Việt xuất sắc
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_index(config: IndexConfig | None = None) -> None:
    """Đọc tài liệu, trích xuất embedding và lưu trữ FAISS Index cùng metadata.

    Args:
        config (Optional[IndexConfig]): Cấu hình xây dựng index. Nếu None, dùng mặc định.

    Raises:
        RuntimeError: Nếu không tìm thấy tài liệu nào để xây dựng chỉ mục.
    """
    config = config or IndexConfig()
    config.validate()
    setup_logging()

    LOGGER.info("Bắt đầu quy trình Ingest tài liệu từ thư mục: %s", config.data_dir)
    chunks = ingest_folder(
        config.data_dir,
        chunk_words=config.chunk_words,
        overlap_words=config.overlap_words,
    )

    if not chunks:
        raise RuntimeError(
            f"Không tìm thấy tài liệu hợp lệ trong '{config.data_dir}'. "
            "Hãy chạy 'python scripts/download_data.py' để tạo dữ liệu mẫu."
        )

    LOGGER.info("Khởi tạo mô hình Embedding: %s", DEFAULT_EMBEDDING_MODEL)
    encoder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

    texts = [c.text for c in chunks]
    LOGGER.info("Bắt đầu vectorize %d chunks văn bản...", len(texts))

    # Mã hóa văn bản và chuẩn hóa L2 vector để tính Inner Product tương đương Cosine Similarity
    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(embeddings, dtype="float32")

    # Khởi tạo chỉ mục FAISS theo phương pháp tích trong (IndexFlatIP)
    vector_dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(vector_dimension)
    index.add(vectors)

    # Đảm bảo thư mục đầu ra tồn tại
    output_dir = Path(config.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lưu tệp index FAISS
    faiss_path = output_dir / "index.faiss"
    faiss.write_index(index, str(faiss_path))
    LOGGER.info(
        "Đã ghi FAISS index (%d vectors, dim=%d) tại: %s",
        index.ntotal,
        vector_dimension,
        faiss_path,
    )

    # Lưu thông tin chi tiết của từng Chunk dưới dạng JSON
    chunks_dict_list = [c.__dict__ for c in chunks]
    save_json(output_dir / "chunks.json", chunks_dict_list)

    # Lưu cấu hình và thông tin phiên bản mô hình (Artifact Contract)
    config_metadata = {
        "schema_version": 1,
        "model_version": "rag-faiss-v1",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "vector_dimension": vector_dimension,
        "chunk_count": len(chunks),
        "chunk_words": config.chunk_words,
        "overlap_words": config.overlap_words,
        "data_dir": str(config.data_dir),
    }
    save_json(output_dir / "config.json", config_metadata)

    LOGGER.info("Xây dựng chỉ mục hoàn tất thành công! Tổng số chunks: %d", len(chunks))


if __name__ == "__main__":
    build_index()
