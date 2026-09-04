# Báo cáo cải tiến dựa trên nghiên cứu

## Nghiên cứu nền tảng

- Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*: https://arxiv.org/abs/2005.11401
- Reimers & Gurevych (2019), *Sentence-BERT*: https://arxiv.org/abs/1908.10084

## Pipeline trước và sau

Trước: đọc file ở cấp thư mục gốc → cắt cửa sổ từ cố định → embedding → FAISS top-k → trả context/LLM.

Sau: đọc đệ quy có kiểm tra thư mục → kiểm tra cấu hình chunk/overlap → dense embedding chuẩn hóa → top-k được chặn theo kích thước index → có `min_score` để từ chối bằng chứng yếu → câu trả lời kèm nguồn.

Các kiểm tra mới ngăn overlap không hợp lệ, query rỗng, `k` vượt số chunk và thư mục dữ liệu không tồn tại. Thay đổi giữ đúng ý tưởng RAG: bộ nhớ phi tham số có thể cập nhật và cung cấp provenance; Sentence-BERT hỗ trợ truy hồi semantic hiệu quả.

## Đánh giá

- Retrieval: Recall@1/3/5, MRR, nDCG@k trên bộ câu hỏi có nguồn chuẩn.
- Generation: faithfulness/citation precision và tỷ lệ từ chối khi thiếu bằng chứng.
- Vận hành: p50/p95 latency, kích thước index, tỷ lệ query rỗng hoặc dưới `min_score`.

Bước tiếp theo: mở rộng validation set, chọn `min_score` trên validation, thử hybrid BM25+dense và reranker; chỉ giữ nếu cải thiện Recall/MRR mà latency chấp nhận được.

## Kết quả chạy thực tế

Ngày 29/08/2026, pipeline đã build FAISS index từ 3 tài liệu mẫu và đạt **Recall@3 = 1,0 trên 3/3 câu hỏi kiểm tra**. Đây là smoke benchmark nhỏ, không phải bằng chứng tổng quát; metric máy đọc được nằm tại `reports/test_metrics.json`.
