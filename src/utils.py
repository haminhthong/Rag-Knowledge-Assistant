"""Các tiện ích bổ trợ cho dự án RAG Knowledge Assistant.

Tệp này cung cấp các hàm hỗ trợ về ghi log (logging), khởi tạo seed ngẫu nhiên
để đảm bảo tính tái lập (reproducibility), đọc/lưu dữ liệu JSON và tính mã checksum (hash) tệp.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

# Khởi tạo Logger chung cho dự án
LOGGER = logging.getLogger("rag_knowledge_assistant")


def setup_logging(level_name: str | None = None) -> None:
    """Cấu hình định dạng và cấp độ hiển thị log cho hệ thống, tự động bật UTF-8 trên Windows.

    Args:
        level_name (Optional[str]): Cấp độ log (DEBUG, INFO, WARNING, ERROR).
            Nếu Không truyền, lấy từ biến môi trường LOG_LEVEL (mặc định INFO).
    """
    # Tự động cấu hình sys.stdout và sys.stderr về UTF-8 trên Windows console
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (OSError, ValueError) as exc:
            LOGGER.debug("Không thể đổi encoding console sang UTF-8: %s", exc)

    if level_name is None:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=level_name,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int = 42) -> None:
    """Khởi tạo seed ngẫu nhiên cho tất cả các thư viện liên quan để tái lập kết quả.

    Args:
        seed (int): Giá trị seed (mặc định: 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        LOGGER.debug("NumPy chưa được cài đặt, bỏ qua np.random.seed")

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        LOGGER.debug("PyTorch chưa được cài đặt, bỏ qua torch.manual_seed")


def save_json(path: str | Path, payload: Any) -> None:
    """Lưu dữ liệu dưới dạng tệp JSON với định dạng Unicode UTF-8 chuẩn.

    Args:
        path (Union[str, Path]): Đường dẫn tệp cần lưu.
        payload (Any): Dữ liệu Python (dict, list) cần ghi.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> Any:
    """Đọc và giải mã dữ liệu từ tệp JSON.

    Args:
        path (Union[str, Path]): Đường dẫn tệp JSON.

    Returns:
        Any: Dữ liệu cấu trúc thu được từ tệp JSON.

    Raises:
        FileNotFoundError: Nếu tệp không tồn tại.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp JSON tại: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


def compute_file_checksum(file_path: str | Path) -> str:
    """Tính mã băm MD5 để nhận diện thay đổi, không dùng cho mục đích bảo mật.

    Args:
        file_path (Union[str, Path]): Đường dẫn tới tệp tài liệu.

    Returns:
        str: Chuỗi mã băm Hexadecimal MD5 của tệp.
    """
    hasher = hashlib.md5(usedforsecurity=False)
    path = Path(file_path)
    with path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
