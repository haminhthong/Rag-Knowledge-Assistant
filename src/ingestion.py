"""Xử lý nạp và trích xuất tài liệu (Data Ingestion & Chunking Pipeline).

Tệp này hỗ trợ đọc các định dạng tài liệu nội bộ phổ biến (TXT, Markdown, PDF, DOCX),
giữ lại thông tin nguồn (Source Metadata) và số trang (Page Number), đồng thời thực hiện
chia nhỏ đoạn văn (Chunking) theo cơ chế Cửa sổ trượt (Sliding Window) có độ chồng lấp (Overlap).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .utils import compute_file_checksum

LOGGER = logging.getLogger("rag_knowledge_assistant.ingestion")


@dataclass
class Chunk:
    """Cấu trúc dữ liệu đại diện cho một đoạn văn bản (Chunk) sau khi phân tách.

    Attributes:
        chunk_id (str): Mã định danh duy nhất của chunk.
        text (str): Nội dung văn bản của chunk.
        source (str): Tên tệp tài liệu gốc.
        page (Optional[int]): Số trang tương ứng (nếu có, ví dụ tệp PDF).
        checksum (Optional[str]): Mã băm MD5 của tệp nguồn phục vụ kiểm tra tính toàn vẹn.
        word_count (int): Số lượng từ trong chunk.
    """

    chunk_id: str
    text: str
    source: str
    page: int | None = None
    checksum: str | None = None
    word_count: int = 0

    def __post_init__(self) -> None:
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())


def read_text(path: Path) -> list[tuple[str, int | None]]:
    """Đọc nội dung văn bản từ tệp tài liệu dựa trên phần mở rộng file.

    Hỗ trợ các định dạng: .txt, .md, .pdf, .docx.

    Args:
        path (Path): Đường dẫn tới tệp tài liệu cần đọc.

    Returns:
        List[Tuple[str, Optional[int]]]: Danh sách các cặp (Nội dung trang, Số trang).

    Raises:
        ValueError: Nếu định dạng tệp không nằm trong danh sách hỗ trợ.
    """
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [(text, None)]

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages_content: list[tuple[str, int | None]] = []
            for idx, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                pages_content.append((extracted, idx + 1))
            return pages_content
        except ImportError:
            LOGGER.warning(
                "Thư viện 'pypdf' chưa được cài đặt. Không thể xử lý file PDF: %s",
                path.name,
            )
            return []
        except Exception as exc:  # noqa: BLE001 - cô lập lỗi parser của tài liệu bên ngoài
            LOGGER.error("Lỗi khi đọc file PDF %s: %s", path.name, exc)
            return []

    if suffix == ".docx":
        try:
            from docx import Document

            doc = Document(str(path))
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return [(full_text, None)]
        except ImportError:
            LOGGER.warning(
                "Thư viện 'python-docx' chưa được cài đặt. Không thể xử lý file DOCX: %s",
                path.name,
            )
            return []
        except Exception as exc:  # noqa: BLE001 - cô lập lỗi parser của tài liệu bên ngoài
            LOGGER.error("Lỗi khi đọc file DOCX %s: %s", path.name, exc)
            return []

    raise ValueError(
        f"Định dạng tệp không được hỗ trợ: '{suffix}' (Chỉ hỗ trợ .txt, .md, .pdf, .docx)"
    )


def chunk_text(
    text: str,
    source: str,
    page: int | None = None,
    chunk_words: int = 220,
    overlap_words: int = 30,
    checksum: str | None = None,
) -> list[Chunk]:
    """Chia nhỏ một đoạn văn bản thành các Chunk có độ dài tối đa chunk_words và overlap_words.

    Sử dụng kỹ thuật Cửa sổ trượt (Sliding Window) dựa trên số từ.

    Args:
        text (str): Nội dung văn bản cần phân tách.
        source (str): Tên file nguồn.
        page (Optional[int]): Số trang (nếu có).
        chunk_words (int): Kích thước từ tối đa mỗi chunk (mặc định: 220).
        overlap_words (int): Số từ chồng lấp giữa các chunk liên tiếp (mặc định: 30).
        checksum (Optional[str]): Mã hash của tệp nguồn.

    Returns:
        List[Chunk]: Danh sách các đối tượng Chunk đã tạo.

    Raises:
        ValueError: Nếu chunk_words <= 0 hoặc overlap_words không hợp lệ.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words phải lớn hơn 0")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError(
            f"overlap_words={overlap_words} phải nằm trong khoảng [0, chunk_words={chunk_words})"
        )

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = max(1, chunk_words - overlap_words)
    stem_name = Path(source).stem

    for i, start_idx in enumerate(range(0, len(words), step)):
        part_words = words[start_idx : start_idx + chunk_words]
        # Bỏ qua các chunk quá nhỏ (dưới 15 từ) để lọc nhiễu văn bản
        if len(part_words) < 15 and len(words) > 15:
            continue

        chunk_text_str = " ".join(part_words)
        page_str = str(page) if page is not None else "0"
        chunk_id = f"{stem_name}-p{page_str}-c{i}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text_str,
                source=source,
                page=page,
                checksum=checksum,
                word_count=len(part_words),
            )
        )

    return chunks


def ingest_folder(
    folder: str | Path,
    chunk_words: int = 220,
    overlap_words: int = 30,
) -> list[Chunk]:
    """Quét toàn bộ thư mục dữ liệu và chuyển đổi tất cả tài liệu thành danh sách Chunks.

    Args:
        folder (Union[str, Path]): Thư mục chứa các tệp tài liệu nội bộ.
        chunk_words (int): Kích thước chunk tính theo số từ.
        overlap_words (int): Độ chồng lấp tính theo số từ.

    Returns:
        List[Chunk]: Danh sách toàn bộ các chunk thu thập được từ tất cả tài liệu.

    Raises:
        FileNotFoundError: Nếu thư mục không tồn tại.
    """
    data_path = Path(folder)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: '{data_path.resolve()}'"
        )

    all_chunks: list[Chunk] = []
    # Quét đệ quy tất cả các file trong thư mục
    file_paths = sorted([p for p in data_path.rglob("*") if p.is_file()])

    for path in file_paths:
        try:
            file_hash = compute_file_checksum(path)
            doc_pages = read_text(path)

            for content, page_num in doc_pages:
                if not content.strip():
                    continue
                page_chunks = chunk_text(
                    text=content,
                    source=path.name,
                    page=page_num,
                    chunk_words=chunk_words,
                    overlap_words=overlap_words,
                    checksum=file_hash,
                )
                all_chunks.extend(page_chunks)
        except ValueError as ve:
            # Bỏ qua các định dạng file không được hỗ trợ mà không dừng chương trình
            LOGGER.debug("Bỏ qua tệp %s: %s", path.name, ve)
            continue
        except Exception as exc:  # noqa: BLE001 - một tệp lỗi không được dừng toàn bộ batch
            LOGGER.error("Lỗi không xác định khi xử lý tệp %s: %s", path.name, exc)
            continue

    LOGGER.info(
        "Đã ingest thành công %d chunks từ %d tệp tài liệu.",
        len(all_chunks),
        len(file_paths),
    )
    return all_chunks
