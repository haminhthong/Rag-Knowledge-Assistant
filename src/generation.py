"""Module sinh câu trả lời bằng LLM (Grounded Generation & Citation Engine).

Tệp này chịu trách nhiệm đóng gói ngữ cảnh (Context) được truy xuất từ các chunk tài liệu,
xây dựng prompt chống suy diễn sai (Anti-Hallucination Prompt), và gọi mô hình LLM local (qua Ollama)
hoặc trả về chế độ Context Fallback khi không có LLM.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

LOGGER = logging.getLogger("rag_knowledge_assistant.generation")


def format_context(hits: list[dict[str, Any]], max_chunks: int = 4) -> str:
    """Định dạng danh sách các chunk thành chuỗi ngữ cảnh có cấu trúc kèm nguồn trích dẫn.

    Args:
        hits (List[Dict[str, Any]]): Danh sách các đoạn tài liệu truy xuất được.
        max_chunks (int): Số đoạn văn bản tối đa đưa vào ngữ cảnh (mặc định: 4).

    Returns:
        str: Chuỗi văn bản đã định dạng kèm thông tin [Nguồn: file | Trang: x].
    """
    context_blocks = []
    for h in hits[:max_chunks]:
        source = h.get("source", "Không rõ nguồn")
        page = h.get("page")
        page_str = f" | trang: {page}" if page is not None else ""
        text = h.get("text", "")
        block = f"[Nguồn: {source}{page_str}]\n{text}"
        context_blocks.append(block)
    return "\n\n".join(context_blocks)


def generate_answer(question: str, hits: list[dict[str, Any]]) -> str:
    """Sinh câu trả lời được căn thực (Grounded Answer) cho câu hỏi dựa trên các tài liệu trích xuất.

    Nếu biến môi trường OLLAMA_URL được thiết lập, gửi yêu cầu tới Ollama LLM local.
    Nếu không có OLLAMA_URL, hoạt động ở chế độ Semantic Search: Trả về danh sách trích đoạn liên quan.

    Args:
        question (str): Câu hỏi của người dùng.
        hits (List[Dict[str, Any]]): Danh sách các chunk liên quan thu được từ Retriever.

    Returns:
        str: Câu trả lời tổng hợp bằng tiếng Việt kèm nguồn trích dẫn hoặc nội dung Context.
    """
    if not hits:
        return "Không tìm thấy thông tin phù hợp trong tài liệu nội bộ."

    context_str = format_context(hits, max_chunks=4)
    ollama_url = os.getenv("OLLAMA_URL")

    # Chế độ Fallback: Trả về trực tiếp Context có trích dẫn khi không sử dụng Ollama LLM
    if not ollama_url:
        LOGGER.debug(
            "OLLAMA_URL không được cấu hình. Trả về câu trả lời dạng Context-only."
        )
        return f"Dữ liệu liên quan tìm thấy trong tài liệu nội bộ:\n\n{context_str}"

    # Cấu hình Prompt chống Hallucination nghiêm ngặt
    prompt = (
        "Bạn là Trợ lý Tri thức Nội bộ của Doanh nghiệp. Hãy trả lời câu hỏi dựa HOÀN TOÀN vào CONTEXT bên dưới.\n"
        "Quy tắc:\n"
        "1. KHÔNG tự nghĩ ra thông tin ngoài CONTEXT được cung cấp.\n"
        "2. Nếu CONTEXT không chứa đủ bằng chứng để trả lời, hãy trả lời chính xác: 'Không đủ thông tin trong tài liệu nội bộ.'\n"
        "3. Trả lời ngắn gọn, súc tích bằng tiếng Việt và trích dẫn rõ nguồn tài liệu (tên tệp/trang).\n\n"
        f"CONTEXT TÀI LIỆU:\n{context_str}\n\n"
        f"CÂU HỎI: {question}\n\n"
        "CÂU TRẢ LỜI:"
    )

    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    endpoint = f"{ollama_url.rstrip('/')}/api/generate"

    try:
        LOGGER.info("Gửi request tới Ollama (%s, model=%s)...", endpoint, model_name)
        response = requests.post(
            endpoint,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        res_data = response.json()
        answer = res_data.get("response", "").strip()
        return answer if answer else "Không đủ thông tin trong tài liệu nội bộ."
    except requests.exceptions.RequestException as exc:
        LOGGER.error("Lỗi khi kết nối tới dịch vụ Ollama (%s): %s", endpoint, exc)
        return (
            f"Lỗi kết nối tới mô hình Ollama LLM ({exc}). "
            f"Dưới đây là thông tin trích xuất trực tiếp từ tài liệu:\n\n{context_str}"
        )
