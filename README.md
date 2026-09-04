# 🚀 Enterprise Knowledge Assistant (Vietnamese RAG System)

> **Trợ lý Tra cứu Tri thức Nội bộ Enterprise bằng Tiếng Việt dựa trên Kiến trúc Retrieval-Augmented Generation (RAG) Chuẩn Production Baseline.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Tổng Quan Dự Án (Project Overview)

**Enterprise Knowledge Assistant** là giải pháp tra cứu và hỏi đáp tự động trên tập tài liệu nội bộ doanh nghiệp (Chính sách nhân sự, Quy định bảo mật, Quy trình tài chính...). 

Hệ thống giúp giải quyết triệt để 2 vấn đề lớn nhất của các mô hình LLM truyền thống:
1. **Ảo giác thông tin (Hallucination)**: Đảm bảo câu trả lời chỉ được sinh ra từ dữ liệu thực tế được trích xuất (Grounded Generation).
2. **Tính kiểm chứng (Citation & Provenance)**: Mỗi câu trả lời đều đính kèm chính xác **tên tệp nguồn**, **số trang** và **điểm độ tin cậy**.

Dự án được thiết kế theo tiêu chuẩn **Production Baseline** dành cho **AI Engineer**, đáp ứng đầy đủ các tiêu chí: Clean Architecture, Type Hints, Chú thích Tiếng Việt, Test Suite coverage và Offline Benchmarking.

---

## 🏗️ Kiến Trúc Hệ Thống (System Architecture)

Hệ thống hoạt động theo quy trình 2 giai đoạn: **Offline Indexing** và **Online Hybrid Retrieval & Generation**.

```mermaid
flowchart TD
    subgraph Data_Ingestion ["1. Offline Data Ingestion & Indexing"]
        A[Tài liệu: TXT, MD, PDF, DOCX] --> B[Checksum & Reading]
        B --> C[Sliding Window Chunking]
        C --> D[SentenceTransformers Model]
        D --> E[FAISS Vector Index]
        C --> F[Chunks Metadata & Config]
    end

    subgraph Online_Query ["2. Online Hybrid Retrieval & Generation"]
        G[Người dùng gửi Query] --> H[FastAPI /query Endpoint]
        H --> I[Dense Search FAISS]
        H --> J[Lexical Search BM25]
        I --> K[Hybrid Fusion & Reranking]
        J --> K
        K --> L{Có cấu hình Ollama?}
        L -- Có -- M[Grounded LLM Prompting]
        L -- Không -- N[Semantic Context Fallback]
        M --> O[Trả về Answer + Sources]
        N --> O
    end
```

---

## 🔥 Các Tính Năng Nổi Bật (Key Technical Features)

| Tính năng | Mô tả chi tiết | Lợi ích cho Doanh nghiệp |
|---|---|---|
| **Hybrid Retrieval (Dense + Lexical)** | Kết hợp giữa **SentenceTransformers** (hiểu ngữ nghĩa) và **BM25 / Lexical Overlap** (bắt từ khóa exact match). | Tìm kiếm tốt cả ngữ nghĩa chung và các mã biểu mẫu, số tiền, tên riêng. |
| **Incremental Ingestion & File Hashing** | Tính mã băm MD5 cho từng tệp tài liệu để nhận diện và bỏ qua tài liệu không thay đổi. | Đẩy nhanh tốc độ re-index, tiết kiệm chi phí tính toán GPU/CPU. |
| **Grounded LLM Generation** | Prompt Engineering chống ảo giác nghiêm ngặt. Tự động chuyển về Semantic Context Fallback khi không có LLM. | Đảm bảo tính sẵn sàng cao (High Availability), không bị gián đoạn nếu LLM offline. |
| **Citation & Page Metadata** | Trích xuất chính xác nguồn văn bản (`source`) và số trang PDF (`page`). | Giúp người dùng kiểm chứng nguồn thông tin chỉ trong vài giây. |
| **Automated Evaluation Suite** | Tích hợp công cụ đo lường tự động: **Recall@k**, **Hit Rate@1**, **MRR**, và **Latency**. | Đánh giá chính xác hiệu năng tra cứu sau mỗi lần điều chỉnh tham số. |
| **FastAPI Enterprise REST API** | Tích hợp OpenAPI, Swagger UI, Pydantic data validation và `/health` readiness check. | Dễ dàng tích hợp với Frontend (React, Vue) hoặc hệ thống Chatbot doanh nghiệp. |

---

## 📂 Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
01_rag_knowledge_assistant/
├── configs/                # Tệp cấu hình bổ trợ
├── data/
│   └── raw/                # Thư mục chứa tài liệu gốc (TXT, MD, PDF, DOCX)
├── models/
│   └── rag_index/          # Artifact chứa index (index.faiss, chunks.json, config.json)
├── reports/
│   └── test_metrics.json   # Kết quả đánh giá tự động (Recall, MRR, Latency)
├── scripts/
│   └── download_data.py    # Script tạo bộ dữ liệu mẫu tiếng Việt
├── src/
│   ├── __init__.py
│   ├── api.py              # Dịch vụ FastAPI REST API (/health, /query)
│   ├── config.py           # Dataclass quản lý cấu hình và CLI parser
│   ├── evaluate.py         # Script đánh giá chỉ số benchmark offline
│   ├── generation.py       # Module căn thực ngữ cảnh và kết nối Ollama LLM
│   ├── index.py            # Module vectorize và xây dựng FAISS Index
│   ├── ingestion.py        # Module đọc file đa định dạng và chia chunk
│   ├── ranking.py          # Module tính BM25 và dung hợp điểm số Hybrid Fusion
│   ├── retrieval.py        # Module Engine tìm kiếm kết hợp Dense + Lexical
│   ├── train.py            # CLI Entrypoint khởi chạy quá trình Indexing
│   └── utils.py            # Tiện ích logging, random seed, checksum và JSON IO
├── tests/
│   ├── test_chunking.py    # Unit test cho logic chia chunk và config
│   ├── test_retrieval.py   # Unit test cho thuật toán BM25 và Hybrid scoring
│   └── test_smoke.py       # Test kiểm tra tính sẵn sàng của hệ thống API
├── Makefile                # Shortcut lệnh vận hành nhanh
├── requirements.txt        # Danh sách thư viện phụ thuộc
├── RESEARCH_REPORT.md      # Báo cáo nghiên cứu khoa học chuyên sâu
└── README.md               # Tài liệu dự án chi tiết
```

---

## ⚙️ Hướng Dẫn Cài Đặt & Chạy Dự Án (Quick Start)

### 1. Yêu cầu môi trường
- Python >= 3.10
- Windows / macOS / Linux

### 2. Khởi tạo môi trường ảo và Cài đặt phụ thuộc

```bash
# Tạo môi trường ảo venv
python -m venv .venv

# Kích hoạt venv (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Hoặc trên Linux/macOS:
# source .venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 3. Khởi tạo dữ liệu mẫu và Xây dựng FAISS Index

```bash
# 1. Tạo 3 tệp quy trình/chính sách mẫu tiếng Việt trong data/raw/
python scripts/download_data.py

# 2. Thực hiện chia chunk và xây dựng chỉ mục vector FAISS
python -m src.train --data-dir data/raw --model-dir models/rag_index
```

### 4. Đánh giá chỉ số Benchmark (Offline Evaluation)

```bash
python -m src.evaluate
```
*Kết quả sẽ được xuất ra màn hình console và lưu tại `reports/test_metrics.json`.*

### 5. Chạy Dịch Vụ REST API

```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```
- **Swagger UI Documentation**: Truy cập ngay tại `http://127.0.0.1:8000/docs`
- **Healthcheck Endpoint**: `http://127.0.0.1:8000/health`

---

## 🤖 Kết Nối Ollama LLM Local (Tùy chọn)

Nếu bạn muốn hệ thống sinh câu trả lời hoàn chỉnh bằng ngôn ngữ tự nhiên thay vì trả về danh sách trích dẫn Context:

1. Cài đặt [Ollama](https://ollama.ai/) và tải model tiếng Việt (ví dụ `qwen2.5:3b` hoặc `vinallama`):
   ```bash
   ollama run qwen2.5:3b
   ```
2. Thiết lập biến môi trường trước khi chạy API:
   ```powershell
   # Windows PowerShell
   $env:OLLAMA_URL="http://localhost:11434"
   $env:OLLAMA_MODEL="qwen2.5:3b"
   ```

---

## 📡 Tài Liệu REST API (API Reference)

### Endpoint `/query` (POST)

#### Request Payload:
```json
{
  "question": "Nhân viên có bao nhiêu ngày phép năm?",
  "top_k": 3,
  "dense_weight": 0.85,
  "min_score": 0.3
}
```

#### Sample Response:
```json
{
  "answer": "Theo Chính sách Nghỉ phép Nội bộ, nhân viên toàn thời gian chính thức có 12 ngày phép năm hưởng nguyên lương. [Nguồn: policy_leave.txt]",
  "sources": [
    {
      "source": "policy_leave.txt",
      "page": null,
      "score": 0.9245,
      "dense_score": 0.8821,
      "lexical_score": 1.0
    }
  ],
  "model_version": "rag-faiss-v1"
}
```

---

## 📊 Kết Quả Đánh Giá Hiệu Năng (Benchmark Results)

Benchmark được lưu tại `data/evaluation/questions.json` và tách hai vai trò:

- `dev`: chỉ dùng chọn `dense_weight` cho hybrid retrieval.
- `test`: chỉ dùng một lần để so sánh lexical-only, dense-only và hybrid.
- Báo cáo `reports/test_metrics.json` ghi Recall@K, Hit@1, MRR, latency trung bình và P95 cho từng cấu hình.

Benchmark đi kèm vẫn là tập minh họa nhỏ trên ba tài liệu mẫu. Không được suy rộng kết quả sang kho tri thức doanh nghiệp hoặc dùng nó để khẳng định chất lượng generation của LLM.

> Báo cáo JSON có sẵn trong bản bàn giao có thể còn ở định dạng benchmark 3 câu cũ. Chạy lại `python -m src.evaluate` sau khi cài đủ FAISS và SentenceTransformers để tạo báo cáo schema v2 với baseline comparison.

---

## 💼 Hướng Dẫn Trình Bày Trong CV AI Engineer (Resume Highlights)

Khi đưa dự án này vào **CV / Portfolio**, bạn nên trình bày theo cấu trúc STAR (Situation - Task - Action - Result):

### 1. Bullet Points Mẫu Cho CV:
- **RAG Retrieval Evaluation**: Xây dựng benchmark dev/test và so sánh lexical-only, dense-only, hybrid bằng Recall@K, MRR và P95 latency.
- **Data Pipeline**: Triển khai Sliding Window Chunking bảo toàn metadata nguồn cùng checksum phục vụ phát hiện tài liệu thay đổi.
- **Grounded Generation**: Thiết kế prompt dựa trên context, citation theo tên file/số trang và fallback khi LLM không khả dụng; chất lượng faithfulness cần được đánh giá riêng.
- **REST API**: Xây dựng FastAPI với OpenAPI, Pydantic validation, health check và lazy loading index.

### 2. Các Câu Hỏi Phỏng Vấn Thực Tế Nào Có Thể Được Hỏi?

<details>
<summary><b>Q1: Tại sao bạn lại chọn Hybrid Search (BM25 + Dense) thay vì chỉ dùng Vector Search đơn thuần?</b></summary>

> **Trả lời:** Dense Vector Search (SentenceTransformers) rất giỏi hiểu ngữ nghĩa chung nhưng lại gặp điểm yếu khi người dùng tìm kiếm các từ khóa chính xác như: mã biểu mẫu (BM-01), số tiền cụ thể (5.000.000 VND), tên riêng hoặc mã quy trình. Việc kết hợp BM25 (Lexical) qua công thức dung hợp trọng số `dense_weight` giúp khắc phục triệt để điểm yếu này, bảo đảm cân bằng giữa ngữ nghĩa và từ khóa chính xác.
</details>

<details>
<summary><b>Q2: Bạn xử lý bài toán ảo giác (Hallucination) của LLM trong dự án này như thế nào?</b></summary>

> **Trả lời:** Em áp dụng 3 tầng phòng ngự: (1) Ép khung Prompt cứng yêu cầu LLM chỉ được dùng dữ liệu từ CONTEXT trích xuất, nếu thiếu thông tin bắt buộc trả về 'Không đủ thông tin'. (2) Tích hợp chỉ số `min_score` để từ chối các đoạn context quá yếu trước khi đưa vào LLM. (3) Cung cấp cơ chế Semantic Fallback – khi Ollama LLM chưa sẵn sàng, hệ thống đóng vai trò như Search Engine trả về trực tiếp đoạn văn bản gốc kèm số trang.
</details>

---

## 🛣️ Định Hướng Phát Triển Tiếp Theo (Production Roadmap)

1. **Bổ sung Cross-Encoder Reranker**: Thử nghiệm các mô hình reranker tiên tiến (`bge-reranker-large`) để tối ưu hóa thứ tự top-k.
2. **Document Level Access Control (ACL)**: Phân quyền truy cập tài liệu theo Phòng ban/Chức vụ người dùng trước khi tiến hành retrieval.
3. **OCR cho PDF Scanned**: Tích hợp Tesseract / PaddleOCR để trích xuất văn bản từ tài liệu PDF dạng ảnh quét.

---

## 📜 Giấy Phép & Tác Giả (License)

Phát triển bởi **AI Engineer Portfolio**. Dự án phát hành theo giấy phép [MIT License](LICENSE).
