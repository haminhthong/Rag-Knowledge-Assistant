"""Module xếp hạng lai (Hybrid Reranking) kết hợp Tìm kiếm Ngữ nghĩa (Dense Cosine) và Từ khóa (Lexical BM25).

Tệp này chứa các thuật toán tách token (Tokenization), tính điểm trùng lặp từ khóa (Lexical Overlap & BM25),
và kết hợp điểm số (Score Fusion) độc lập với mô hình embedding và index FAISS.
"""

from __future__ import annotations

import math
import re

# Regex nhận diện các từ unicode (hỗ trợ tiếng Việt)
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tách văn bản thành danh sách từ (token) dạng chữ thường (case-folded).

    Args:
        text (str): Chuỗi văn bản đầu vào.

    Returns:
        List[str]: Danh sách các token thu được.
    """
    return TOKEN_PATTERN.findall(text.casefold())


def get_token_set(text: str) -> set[str]:
    """Tách văn bản và chuyển thành tập hợp các từ độc nhất.

    Args:
        text (str): Chuỗi văn bản đầu vào.

    Returns:
        Set[str]: Tập hợp các token.
    """
    return set(tokenize(text))


def lexical_overlap(query: str, document: str) -> float:
    """Tính tỷ lệ trùng lặp từ khóa (Taccard-like Overlap) giữa câu hỏi và văn bản.

    Args:
        query (str): Câu hỏi/truy vấn của người dùng.
        document (str): Đoạn văn bản tài liệu.

    Returns:
        float: Giá trị trong khoảng [0.0, 1.0] thể hiện tỷ lệ token của query có trong document.
    """
    query_tokens = get_token_set(query)
    if not query_tokens:
        return 0.0
    doc_tokens = get_token_set(document)
    intersection = query_tokens & doc_tokens
    return len(intersection) / len(query_tokens)


def bm25_score_single(
    query_tokens: list[str],
    doc_tokens: list[str],
    total_docs: int = 100,
    doc_freqs: dict[str, int] | None = None,
    avg_doc_len: float = 200.0,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Tính điểm BM25 cho một văn bản đối với danh sách token truy vấn.

    Args:
        query_tokens (List[str]): Danh sách các từ trong câu hỏi.
        doc_tokens (List[str]): Danh sách các từ trong văn bản.
        total_docs (int): Tổng số văn bản trong corpus.
        doc_freqs (Dict[str, int], optional): Số văn bản chứa từng từ.
        avg_doc_len (float): Độ dài trung bình của văn bản trong corpus.
        k1 (float): Hằng số điều chỉnh tần suất từ (Term Frequency saturation parameter).
        b (float): Hằng số điều chỉnh phạt độ dài văn bản (Length Normalization parameter).

    Returns:
        float: Điểm số BM25 (chưa chuẩn hóa).
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    doc_len = len(doc_tokens)
    score = 0.0
    doc_freqs = doc_freqs or {}

    # Đếm tần suất các từ trong văn bản
    tf_map: dict[str, int] = {}
    for term in doc_tokens:
        tf_map[term] = tf_map.get(term, 0) + 1

    for term in set(query_tokens):
        if term not in tf_map:
            continue

        freq = tf_map[term]
        df = doc_freqs.get(term, 1)
        # Điểm Inverse Document Frequency (IDF) của Robertson BM25
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

        # Phần thưởng Term Frequency đã qua chuẩn hóa độ dài
        numerator = freq * (k1 + 1.0)
        denominator = freq + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_doc_len)))
        score += idf * (numerator / denominator)

    return max(0.0, score)


def hybrid_score(
    dense_score: float,
    lexical_score: float,
    dense_weight: float = 0.85,
) -> float:
    """Kết hợp điểm Dense Cosine Similarity và điểm Lexical Relevance thành điểm tổng hợp.

    Công thức: Combine = dense_weight * Normal(dense_score) + (1 - dense_weight) * lexical_score

    Args:
        dense_score (float): Điểm tương đồng Cosine từ FAISS index [-1.0, 1.0].
        lexical_score (float): Điểm trùng lặp từ khóa [0.0, 1.0].
        dense_weight (float): Tỷ trọng dành cho Dense Retrieval (từ 0.0 đến 1.0, mặc định: 0.85).

    Returns:
        float: Điểm số tổng hợp chuẩn hóa trong khoảng [0.0, 1.0].

    Raises:
        ValueError: Nếu dense_weight không nằm trong khoảng [0.0, 1.0].
    """
    if not (0.0 <= dense_weight <= 1.0):
        raise ValueError(
            f"dense_weight={dense_weight} phải nằm trong khoảng [0.0, 1.0]"
        )

    # Chuẩn hóa Dense Cosine Score từ [-1, 1] về khoảng [0, 1]
    normalized_dense = min(max((dense_score + 1.0) / 2.0, 0.0), 1.0)

    # Kết hợp tuyến tính trọng số
    final_score = dense_weight * normalized_dense + (1.0 - dense_weight) * lexical_score
    return round(final_score, 4)
