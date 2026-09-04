"""Script khởi tạo bộ dữ liệu tài liệu nội bộ mẫu phục vụ thử nghiệm RAG.

Tạo các tệp quy trình nghỉ phép, hướng dẫn bảo mật và chính sách hoàn ứng trong data/raw/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Đảm bảo Windows console in đúng UTF-8 không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (OSError, ValueError) as exc:
        logging.getLogger("download_data").debug(
            "Không thể đổi encoding console sang UTF-8: %s", exc
        )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger("download_data")

SAMPLES = {
    "policy_leave.txt": (
        "CHÍNH SÁCH NGHỈ PHÉP NỘI BỘ\n\n"
        "1. Quyền lợi nghỉ phép:\n"
        "Nhân viên toàn thời gian chính thức có 12 ngày phép năm hưởng nguyên lương.\n"
        "Nhân viên có thâm niên từ 5 năm trở lên được cộng thêm 1 ngày phép cho mỗi năm tiếp theo.\n\n"
        "2. Quy định đăng ký:\n"
        "Đơn xin nghỉ phép từ 3 ngày liên tiếp trở lên cần gửi đăng ký trước tối thiểu 5 ngày làm việc.\n"
        "Trường hợp khẩn cấp (ốm đau, việc gia đình đột xuất), nhân viên có thể báo trực tiếp cho quản lý "
        "và bổ sung đơn nghỉ trên hệ thống trong vòng 24 giờ sau khi quay lại làm việc."
    ),
    "security_guide.txt": (
        "HƯỚNG DẪN BẢO MẬT AN THÔNG TIN\n\n"
        "1. Quản lý tài khoản và mật khẩu:\n"
        "Tuyệt đối không chia sẻ mật khẩu cá nhân hoặc tài khoản nội bộ cho bất kỳ ai.\n"
        "Bắt buộc bật xác thực hai yếu tố (2FA) cho toàn bộ tài khoản công ty.\n\n"
        "2. Quy trình xử lý sự cố an ninh mạng:\n"
        "Khi phát hiện nghi ngờ lộ thông tin hoặc tấn công mạng, nhân viên phải báo lập tức "
        "cho đội Bảo mật (Security Team) trong vòng 30 phút kể từ khi phát hiện qua email security@company.com "
        "hoặc kênh Slack #incident-report."
    ),
    "expense_policy.txt": (
        "CHÍNH SÁCH VÀ QUY TRÌNH HOÀN ỨNG CHI PHÍ CÔNG TÁC\n\n"
        "1. Điều kiện hoàn ứng:\n"
        "Mọi chi phí công tác phát sinh chỉ được hoàn ứng khi có hóa đơn tài chính hợp lệ (hóa đơn đỏ/VAT).\n"
        "Đối với các khoản chi phí phát sinh trên 5.000.000 VND (năm triệu đồng), nhân viên cần phải "
        "có phê duyệt trước (Pre-approval) bằng văn bản hoặc email từ Quản lý trực tiếp.\n\n"
        "2. Thời hạn nộp hồ sơ:\n"
        "Hồ sơ thanh toán hoàn ứng phải được nộp cho phòng Kế toán trong vòng 10 ngày làm việc "
        "kể từ ngày kết thúc chuyến công tác."
    ),
}


def create_sample_data(data_dir: str = "data/raw") -> None:
    """Tạo các tệp tài liệu mẫu tại thư mục quy định.

    Args:
        data_dir (str): Thư mục đầu ra chứa các tệp tài liệu.
    """
    output_path = Path(data_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for filename, text_content in SAMPLES.items():
        file_path = output_path / filename
        file_path.write_text(text_content, encoding="utf-8")
        LOGGER.info("Đã tạo tệp tài liệu mẫu: %s", file_path.resolve())

    print(
        f"\n[OK] Đã tạo thành công {len(SAMPLES)} tệp tài liệu mẫu tại '{output_path.resolve()}'"
    )


if __name__ == "__main__":
    create_sample_data()
