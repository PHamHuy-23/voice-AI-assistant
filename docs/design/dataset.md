# Quy định Dataset & Kiểm định Dữ liệu (Data Audit)

Tài liệu này xác định vai trò của từng bộ dữ liệu trong nghiên cứu và nguyên tắc kiểm chứng thực tế.

---

## 1. Google Speech Commands v2 (Dataset chính)
- **Vai trò:** Huấn luyện chính (Episodic training), validation, và đánh giá năng lực Few-shot trên tập test classes.
- **Tổng số lớp:** 35 từ khóa.
- **Phân chia theo lớp (Class Disjoint Split):**
  - **15 Train Classes:** `happy`, `house`, `bird`, `bed`, `backward`, `sheila`, `marvin`, `wow`, `tree`, `follow`, `dog`, `visual`, `forward`, `learn`, `cat`.
  - **10 Validation Classes:** `zero`, `one`, `two`, `three`, `four`, `five`, `six`, `seven`, `eight`, `nine`.
  - **10 Test Classes:** `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`.
- **Đặc điểm kiểm định:**
  - Tập Test hoàn toàn không xuất hiện trong quá trình huấn luyện của TD-ResNet.
  - Mọi thông số thống kê (duration min/max/mean, sample rate distribution, corruption check) được trích xuất bằng mã lệnh `scripts/audit_dataset.py`.

---

## 2. AudioMNIST (Dataset bổ sung 1)
- **Vai trò:** Dữ liệu single-word spoken digits (chữ số từ 0 đến 9) phát âm bởi 60 người nói khác nhau.
- **Mục đích:** Kiểm thử tính ổn định của pipeline tiền xử lý (resampling từ 48kHz về 16kHz, amplitude normalization).
- **Lưu ý:** Không coi đây là benchmark chính cho Few-shot KWS trừ khi có thí nghiệm mở rộng riêng.

---

## 3. Fluent Speech Commands (Dataset bổ sung 2)
- **Vai trò:** Dữ liệu spoken command đa từ (ví dụ: *"turn on the lights in the kitchen"*).
- **Mục đích:** Thử nghiệm tiền xử lý trên tín hiệu âm thanh có thời lượng dài hơn và mức độ phức tạp cao hơn.
- **Lưu ý:** Tuyệt đối không gọi FSC là multi-keyword spotting dataset vì bản chất của FSC là bài toán spoken intent classification.

---

## 4. Báo cáo kiểm định dữ liệu tự động
Các kết quả kiểm định thật được lưu trữ tại:
- `reports/audits/generated/gsc_summary.csv`
- `reports/audits/generated/gsc_class_distribution.csv`
- `reports/audits/generated/gsc_file_audit.csv`
