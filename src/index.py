"""Xây dựng và Lưu trữ chỉ mục FAISS Dense Index & BM25 Lexical Index (Indexing Module).

Tệp này quản lý:
1. Nạp và phân rã văn bản theo cấu trúc (Structure-Aware Ingestion).
2. Tạo mã định danh ổn định và lập Document Manifest theo dõi xuất xứ (Provenance).
3. Vectorize bằng SentenceTransformers và tạo FAISS IndexFlatIP.
4. Xây dựng BM25Okapi Index phục vụ tìm kiếm từ khóa độc lập.
5. Lưu trữ các artifact chuẩn hóa (config.json schema v2, index.faiss, chunks.json, bm25_index.json, document_manifest.json).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import IndexConfig
from .ingestion import create_document_manifest, ingest_folder
from .ranking import BM25Index
from .utils import save_json, setup_logging

LOGGER = logging.getLogger("rag_knowledge_assistant.index")


def build_index(config: IndexConfig | None = None) -> None:
    """Đọc tài liệu, trích xuất embedding và lưu trữ FAISS Index, BM25 Index cùng metadata.

    Args:
        config (Optional[IndexConfig]): Cấu hình xây dựng index. Nếu None, dùng mặc định.

    Raises:
        RuntimeError: Nếu không tìm thấy tài liệu nào để xây dựng chỉ mục.
    """
    config = config or IndexConfig()
    config.validate()
    setup_logging()

    LOGGER.info(
        "Bắt đầu quy trình Ingest tài liệu từ '%s' (strategy='%s', chunk_words=%d)...",
        config.data_dir,
        config.strategy,
        config.chunk_words,
    )
    chunks = ingest_folder(
        config.data_dir,
        chunk_words=config.chunk_words,
        overlap_words=config.overlap_words,
        strategy=config.strategy,
    )

    if not chunks:
        raise RuntimeError(
            f"Không tìm thấy tài liệu hợp lệ trong '{config.data_dir}'. "
            "Hãy chạy 'python scripts/download_data.py' để tạo dữ liệu mẫu."
        )

    # 1. Tạo Document Manifest theo dõi Provenance và tính toàn vẹn
    manifest = create_document_manifest(chunks, config.data_dir)
    output_dir = Path(config.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "document_manifest.json", manifest)
    LOGGER.info(
        "Đã lưu Document Manifest (%d tài liệu, %d chunks) tại document_manifest.json",
        manifest["total_documents"],
        len(chunks),
    )

    # 2. Xây dựng Dense Embeddings & FAISS Index
    LOGGER.info("Khởi tạo mô hình Embedding: %s", config.embedding_model)
    encoder = SentenceTransformer(config.embedding_model)

    texts = [c.text for c in chunks]
    LOGGER.info("Bắt đầu vectorize %d chunks văn bản...", len(texts))

    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(embeddings, dtype="float32")

    vector_dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(vector_dimension)
    index.add(vectors)

    faiss_path = output_dir / "index.faiss"
    faiss.write_index(index, str(faiss_path))
    LOGGER.info(
        "Đã ghi FAISS index (%d vectors, dim=%d) tại: %s",
        index.ntotal,
        vector_dimension,
        faiss_path.name,
    )

    # 3. Xây dựng và lưu trữ BM25 Index
    LOGGER.info("Xây dựng BM25Okapi Lexical Index cho %d chunks...", len(texts))
    bm25_index = BM25Index.from_texts(texts)
    save_json(output_dir / "bm25_index.json", bm25_index.to_dict())
    LOGGER.info("Đã ghi BM25 Index tại bm25_index.json")

    # 4. Lưu thông tin chi tiết của từng Chunk dưới dạng JSON
    chunks_dict_list = [c.__dict__ for c in chunks]
    save_json(output_dir / "chunks.json", chunks_dict_list)

    # 5. Tính toán mã băm toàn bộ ngữ liệu (Corpus Hash) và Versioning
    corpus_hasher = hashlib.sha256()
    for text in texts:
        corpus_hasher.update(text.encode("utf-8"))
    corpus_hash = corpus_hasher.hexdigest()[:16]

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    index_version = f"{timestamp_str}-{corpus_hash[:6]}"

    # Lưu cấu hình và thông tin phiên bản mô hình (Artifact Contract schema v2)
    config_metadata = {
        "schema_version": 2,
        "model_version": "rag-evidence-v2",
        "index_version": index_version,
        "embedding_model": config.embedding_model,
        "reranker_model": config.reranker_model,
        "vector_dimension": vector_dimension,
        "chunk_count": len(chunks),
        "document_count": manifest["total_documents"],
        "chunk_words": config.chunk_words,
        "overlap_words": config.overlap_words,
        "strategy": config.strategy,
        "corpus_hash": corpus_hash,
        "candidate_pool_k": config.candidate_pool_k,
        "rrf_k": config.rrf_k,
        "rerank_top_k": config.rerank_top_k,
        "evidence_gate_threshold": config.evidence_gate_threshold,
        "data_dir": str(config.data_dir),
    }
    save_json(output_dir / "config.json", config_metadata)

    LOGGER.info(
        "Xây dựng chỉ mục hoàn tất thành công! Version: %s, Tổng số chunks: %d",
        index_version,
        len(chunks),
    )


if __name__ == "__main__":
    build_index()
