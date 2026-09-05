# Báo Cáo Nghiên Cứu Kỹ Thuật: Canonical Evidence-Grounded Hybrid RAG

## 1. Cơ Sở Lý Thuyết Nền Tảng (Theoretical Foundations)

1. **Retrieval-Augmented Generation (RAG)**:
   - Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (arXiv:2005.11401).
   - Đặt nền móng cho việc kết hợp bộ nhớ tham số (Parametric Memory - LLM) và bộ nhớ phi tham số có thể cập nhật động (Non-parametric Memory - Dense & Sparse Indexes).
2. **Dense Semantic Representation**:
   - Reimers & Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* (arXiv:1908.10084).
   - Ánh xạ văn bản vào không gian vector ngữ nghĩa 384 chiều, tối ưu hóa tìm kiếm tương đồng Cosine qua Inner Product trên vector chuẩn hóa L2.
3. **Probabilistic Relevance Framework (BM25)**:
   - Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond*.
   - Khắc phục giới hạn của Dense Retrieval đối với các từ khóa chính xác, mã biểu mẫu, và số tiền cụ thể thông qua Term Frequency saturation và Length Normalization.
4. **Reciprocal Rank Fusion (RRF)**:
   - Cormack, Clarke & Buettcher (2009), *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*.
   - Dung hợp thứ hạng không phụ thuộc vào phân phối điểm số thô: $RRF(d) = \sum_{r} \frac{1}{k + rank_r(d)}$.
5. **Cross-Encoder Neural Reranking**:
   - Nogueira & Cho (2019), *Passage Re-ranking with BERT* (arXiv:1901.04085).
   - Cho phép Full Cross-Attention giữa mọi token của câu hỏi và tài liệu, tạo ra biểu diễn tương tác sâu mà Bi-Encoder độc lập không thể đạt được.

---

## 2. So Sánh Kiến Trúc Trước & Sau Cải Tiến

| Tiêu chí kỹ thuật | Kiến trúc Ban đầu (Baseline Prototype) | Kiến trúc Chuẩn hóa Mới (Canonical Flagship RAG) |
|---|---|---|
| **Cơ chế Hybrid Retrieval** | Giả hybrid: Chỉ lấy Top-K từ FAISS, sau đó tính lexical overlap và kết hợp tuyến tính. Bỏ sót từ khóa nếu Dense không tìm thấy. | **True Dual Retrieval + Candidate Union**: Lấy độc lập Top 30 từ FAISS và Top 30 từ BM25Okapi, hợp nhất tập ứng viên trước khi xếp hạng. |
| **Thuật toán Dung hợp** | Tuyến tính điểm số thô: `w * dense + (1-w) * overlap`. Dễ bị méo mó khi thang đo không đồng nhất. | **Reciprocal Rank Fusion (RRF, $k=60$)**: Dung hợp dựa trên thứ tự xếp hạng, hoàn toàn độc lập với phân phối điểm số. |
| **Reranking** | Không có neural reranker (chỉ là score fusion). | **Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`)** với cơ chế Fallback thích ứng khi môi trường không có GPU. |
| **Chiến lược Chunking** | Sliding Window cố định 220 từ; dễ cắt ngang câu hoặc tách rời tiêu đề mục khỏi nội dung. | **Structure-Aware Chunking**: Nhận diện Heading/Section, phân đoạn theo ranh giới câu, đóng gói ~250 từ kèm context tiêu đề. |
| **Định danh Chunk & Document** | Dễ xung đột (`{file_stem}-p{page}-c{i}`); không lưu đường dẫn tương đối. | **Deterministic Stable IDs**: `document_id = SHA256(relative_path)[:12]` và `chunk_id = {doc_id}:p{page}:c{idx:03d}`. |
| **Kiểm soát Chất lượng Bằng chứng** | Không có (hoặc phụ thuộc LLM prompt); câu hỏi ngoài phạm vi vẫn gửi ngữ cảnh rác vào LLM. | **Evidence Quality Gate**: Kiểm tra điểm rerank tối thiểu; kích hoạt **Early Abstain** 100% trước khi gọi LLM đối với câu hỏi ngoài phạm vi. |
| **Xác thực Trích dẫn** | LLM tự do sinh tên file/trang; không có cơ chế hậu kiểm. | **CitationValidator**: Bóc tách thẻ `[C1]`, `[C2]` và kiểm định nghiêm ngặt tập trích dẫn thuộc danh sách bằng chứng được cung cấp. |
| **Phòng vệ Prompt Injection** | Ghép thẳng context vào prompt mà không có thẻ bao bọc. | **Untrusted XML Delimiters (`<evidence id="C1">`)** kèm chỉ thị hệ thống xem tài liệu là untrusted data. |
| **Quy mô & Chất lượng Benchmark** | 10 câu hỏi minh họa, chỉ kiểm tra ở mức document source name. | **50 câu hỏi phân tầng (Factual, Paraphrase, Keyword, Numeric, No-answer, Ambiguous)** kèm ground truth ở mức Section và Reference Answer. |

---

## 3. Kết Quả Thực Nghiệm & Bóc Tách (Empirical Findings)

### 3.1. RAG Component Ablation
Chạy trên 31 câu hỏi test có đáp án (nguồn: `reports/rag_ablation.json`):
- **BM25 Only**: Đạt MRR = 0.9839 với độ trễ siêu thấp (1.14 ms) nhờ vào các câu hỏi chứa chính xác từ khóa, số tiền, và thời hạn trong tập văn bản nội bộ.
- **Dense Only**: Đạt MRR = 0.9597, trễ 52.73 ms.
- **RRF Union**: Đạt Recall@4 = 1.0000, MRR = 0.9785, nDCG@4 = 1.4777 mà không cần tinh chỉnh trọng số thủ công `dense_weight`.
- **RRF + Cross-Encoder Reranker**: Mang lại khả năng chấm điểm tương quan sâu cấp độ câu, giúp lọc bỏ các chunk chỉ chứa từ khóa ngẫu nhiên nhưng sai ngữ cảnh.
- **Evidence Quality Gate**: Đạt **Tỷ lệ từ chối đúng 100% (7/7 câu hỏi)** trên tập câu hỏi ngoài phạm vi (No-Answer).

### 3.2. Chunking Strategy Ablation
Chạy trên 31 câu hỏi test (nguồn: `reports/chunk_ablation.json`):
- **Sliding Window (120w, 220w, 350w)**: Tạo ra 3 chunks lớn gom nhiều mục quy định khác nhau vào một chunk.
- **Structure-Aware Chunking**: Tạo ra 6 chunks phân lập rõ ràng theo từng Điều/Mục quy chế (Điều kiện hoàn ứng, Thời hạn nộp hồ sơ, Quyền lợi nghỉ phép, Quy trình đăng ký, Quản lý tài khoản, Xử lý sự cố). Nhờ đó, nDCG@4 tăng vọt từ 0.9881 lên **1.4777**, phản ánh độ hội tụ chính xác của trích dẫn ở cấp độ từng Section riêng biệt.
