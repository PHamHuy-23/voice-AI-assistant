# Hướng dẫn Thực nghiệm & Quy trình Báo cáo Học thuật

Tài liệu này cung cấp hướng dẫn từng bước để chạy thực nghiệm trên Kaggle hoặc máy trạm có GPU và trích xuất số liệu cho báo cáo môn học.

---

## 1. Chuẩn bị môi trường trên Kaggle

Trong Notebook Kaggle:
```bash
# Clone repository
!git clone https://github.com/PHamHuy-23/voice-AI-assistant.git
%cd voice-AI-assistant

# Cài đặt thư viện
!pip install -q pyyaml soundfile librosa pandas numpy
```

---

## 2. Các bước sinh số liệu cho Báo cáo môn học

### REPORT 2.2 — DATASET AUDIT
Chạy kiểm định và trích xuất bảng số liệu thật:
```bash
python scripts/audit_dataset.py \
    --config configs/data/gsc.yaml \
    --data_dir /kaggle/input/google-speech-commands-v2/speech_commands_v2 \
    --output_dir reports/data_audit
```
Số liệu sẽ được xuất ra `reports/data_audit/gsc_summary.csv` để đưa thẳng vào Chương 2.2 của Báo cáo.

---

### REPORT 2.4 — FEATURE EXTRACTION
Chạy Notebook `notebooks/03_feature_analysis.ipynb` để sinh lưới đồ thị đặc trưng 6 khung hình:
- Waveform
- STFT Linear Spectrogram
- Log Mel-Spectrogram
- MFCC
- RMS Energy
- Zero Crossing Rate (ZCR)
Hình ảnh xuất ra: `reports/figures/feature_gallery.png`.

---

### REPORT 2.5 — THỰC NGHIỆM RAW VS NORMALIZED
So sánh sự khác biệt khi bật/tắt các bộ lọc tiền xử lý:
```bash
python scripts/compare_raw_normalized.py \
    --audio data/raw/speech_commands_v2/yes/0a7c4067_nohash_0.wav \
    --output reports/figures/raw_vs_normalized.png
```
Hình ảnh xuất ra so sánh trực tiếp dạng sóng, phổ Mel và năng lượng RMS giữa 2 chế độ.

---

### REPORT 3.x — HUẤN LUYỆN FEW-SHOT EPISODIC
Huấn luyện TD-ResNet và ProtoNet trên Kaggle GPU (T4 hoặc P100):
```bash
python scripts/train.py \
    --config configs/experiments/exp_001_5way_5shot.yaml \
    --data_dir /kaggle/input/google-speech-commands-v2/speech_commands_v2
```
Kết quả lưu tại `checkpoints/exp_001/`:
- `best.pt`: Trọng số tối ưu nhất trên tập validation classes.
- `metrics.json`: Lịch sử loss và accuracy qua từng epoch.
- `training_curves.png`: Đồ thị học tập phục vụ báo cáo.

---

### REPORT 3.x — ĐÁNH GIÁ MA TRẬN N-WAY K-SHOT
Đánh giá độ chính xác tổng quát hóa trên 10 từ khóa mới:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/exp_001/best.pt \
    --data_dir /kaggle/input/google-speech-commands-v2/speech_commands_v2 \
    --output_csv reports/tables/evaluation_results.csv
```
Xuất bảng độ chính xác (Accuracy ± 95% Confidence Interval) cho các thiết lập:
- 5-way 1-shot
- 5-way 5-shot
- 10-way 1-shot
- 10-way 5-shot
- Độ trễ suy luận trung bình (Inference Latency in ms).
