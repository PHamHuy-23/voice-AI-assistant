# Few-Shot Keyword Spotting using TD-ResNet and Prototypical Networks

> **Đồ án môn học:** Xử lý tiếng nói (Speech Processing)  
> **Hướng nghiên cứu:** Nhận diện từ khóa tiếng nói trong điều kiện ít mẫu tham chiếu (Few-Shot Keyword Spotting)  
> **Định hướng sản phẩm ứng dụng:** Hệ thống lõi nhận diện âm thanh cho Desktop Voice AI Assistant (kết hợp Wake-up Word & Command Dispatcher điều khiển Antigravity).

---

## 1. Project Overview (Tổng quan dự án)
Dự án này tập trung giải quyết bài toán nhận diện các từ khóa tiếng nói (Keyword Spotting - KWS) mới khi mỗi từ khóa chỉ có một số lượng rất nhỏ mẫu âm thanh tham chiếu (từ 1 đến 5 mẫu). Thay vì huấn luyện lại toàn bộ mạng nơ-ron hoặc bổ sung neuron cho lớp Softmax cố định, hệ thống kết hợp mạng trích xuất đặc trưng **TD-ResNet (Temporal Dilated Residual Network)** và bộ phân loại không gian metric **Prototypical Network (ProtoNet)** theo phương pháp học theo tập (Episodic Learning).

Đồng thời, module này được xây dựng với vai trò là **Audio Intelligence Core** cho ứng dụng **Desktop Voice AI Assistant**, cho phép hệ thống chạy ngầm, kích hoạt bằng wake-word và nhận diện khẩu lệnh tùy biến của người dùng để điều khiển máy tính thông qua **Antigravity**.

---

## 2. Problem Statement (Định nghĩa bài toán)
- **Đầu vào tổng quát:**
  - Tập các đoạn âm thanh ngắn chứa từ hoặc khẩu lệnh thoại.
  - Một số ít mẫu tham chiếu có nhãn cho mỗi từ khóa (Support Set $S$).
  - Một đoạn âm thanh truy vấn cần nhận diện (Query sample $x_q$).
- **Đầu ra:**
  - Nhãn từ khóa dự đoán $\hat{y} = \arg\min_k d(f_\theta(x_q), c_k)$.
  - Khoảng cách / độ tương đồng giữa vector embedding của truy vấn và các prototype đại diện.
  - Điểm tự tin (Softmax confidence score).
  - Các metric đo lường: Top-1 Accuracy, Mean Episode Accuracy, 95% Confidence Interval, và Inference Latency.

---

## 3. System Architecture (Kiến trúc hệ thống)

```
[Microphone / Audio WAV]
          │
          ▼
┌────────────────────────────────────────┐
│ Audio Preprocessing Pipeline           │
│ (Mono, Resample 16kHz, Trim, Norm)    │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ Speech Feature Extractor               │
│ (Log Mel-Spectrogram: F=64, T=101)     │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ TD-ResNet Embedding Backbone (f_theta) │
│ (Temporal Dilated Convolutions)        │
└──────────────────┬─────────────────────┘
                   │
                   ▼ Embedding vector z in R^128
┌────────────────────────────────────────┐
│ Prototypical Network Head              │
│ - Compute Prototypes: c_k = Mean(z_ik) │
│ - Compute Euclidean / Cosine Distances │
└──────────────────┬─────────────────────┘
                   │
                   ▼
       [Predicted Keyword Label]
                   │
                   ▼ (Desktop Voice Assistant Bridge)
┌────────────────────────────────────────────────────────┐
│ OS Background Daemon & Antigravity Dispatcher          │
│ - Wake-up Word State Machine                           │
│ - Command Matcher                                      │
│ - Dispatch action: Mở YouTube, VS Code, Kiểm tra giờ...│
└────────────────────────────────────────────────────────┘
```

---

## 4. Few-Shot Learning Concept (Khái niệm Học ít mẫu)
- **$N$-way $K$-shot:** Phân loại giữa $N$ từ khóa khác nhau, mỗi từ khóa chỉ có $K$ mẫu tham chiếu trong Support Set.
- **Support Set ($S$):** Tập dữ liệu mẫu dùng để tính toán tọa độ tâm (Prototype $c_k$) cho từng lớp từ khóa trong không gian đặc trưng.
- **Query Set ($Q$):** Tập dữ liệu cần dự đoán nhãn dựa trên khoảng cách tới các prototype.
- **Episodic Learning:** Quá trình huấn luyện không chia theo mini-batch truyền thống mà chia theo các episode. Mỗi episode là một bài toán con $N$-way $K$-shot mô phỏng đúng điều kiện kiểm thử thực tế.

---

## 5. Dataset (Tập dữ liệu)
Repository quản lý 3 bộ dữ liệu với vai trò chuyên biệt:
1. **Google Speech Commands v2 (Few-Shot GSC) [Chính]:**
   - 35 từ khóa, tần số 16 kHz, định dạng WAV xấp xỉ 1.0 giây.
   - Chia lớp độc lập (Class-disjoint split):
     - **15 Train Classes:** `happy`, `house`, `bird`, `bed`, `backward`, `sheila`, `marvin`, `wow`, `tree`, `follow`, `dog`, `visual`, `forward`, `learn`, `cat`.
     - **10 Validation Classes:** `zero`, `one`, `two`, `three`, `four`, `five`, `six`, `seven`, `eight`, `nine`.
     - **10 Test Classes:** `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`.
2. **AudioMNIST [Bổ sung 1]:** Bộ số nói (0-9) dùng để kiểm tra tính ổn định của pipeline tiền xử lý và chuyển đổi tần số lấy mẫu (48 kHz $\to$ 16 kHz).
3. **Fluent Speech Commands [Bổ sung 2]:** Dữ liệu câu lệnh đa từ phục vụ đánh giá tiền xử lý trên tín hiệu dài (không coi đây là bài toán KWS đơn từ).

---

## 6. Data Preprocessing (Quy trình tiền xử lý)
Hỗ trợ kiểm tra thực nghiệm **RAW vs. NORMALIZED** thông qua cấu hình bật/tắt linh hoạt (`configs/preprocessing/`):
- `to_mono`: Chuyển đổi kênh âm thanh về mono.
- `resample`: Chuẩn hóa tần số lấy mẫu về 16.000 Hz.
- `trim_silence`: Loại bỏ khoảng lặng đầu và cuối dựa trên ngưỡng dB.
- `normalize_amplitude`: Chuẩn hóa biên độ theo giá trị đỉnh (Peak Normalization) hoặc RMS.
- `pad_or_crop`: Cắt tỉa hoặc chèn zero-padding về độ dài đồng nhất $16.000$ mẫu (1.0 giây).

---

## 7. Feature Extraction (Trích xuất đặc trưng tiếng nói)
- **Đặc trưng đưa vào mô hình:** Log Mel-Spectrogram ($64$ mel bands, FFT window 400 mẫu, hop 160 mẫu).
- **Đặc trưng phân tích học thuật (Report 2.4):**
  - Waveform (Dạng sóng thời gian).
  - Linear Spectrogram (Biểu đồ phổ công suất STFT).
  - MFCC (Mel-Frequency Cepstral Coefficients).
  - Short-Time RMS Energy (Đường bao năng lượng).
  - Zero Crossing Rate (Tỷ lệ đổi dấu biên độ).

---

## 8. Model Architecture (Kiến trúc mô hình)
- **Embedding Backbone (TD-ResNet):**
  - Tích chập thời gian giãn nở (Temporal Dilations: 1, 2, 4) giúp bao quát ngữ cảnh phát âm dài.
  - Residual connections ổn định gradient.
  - Global average pooling trích xuất vector embedding cố định kích thước 128 chiều ($d=128$).
  - Chuẩn hóa L2-normalization trước khi tính toán khoảng cách.
- **Classification Head (Prototypical Network):**
  - Tọa độ Prototype của lớp $k$: $c_k = \frac{1}{K} \sum_{i=1}^K f_\theta(x_i)$.
  - Khoảng cách Euclid bình phương: $d(z_q, c_k) = \|z_q - c_k\|_2^2$.

---

## 9. Training (Quy trình huấn luyện)
- Huấn luyện theo cơ chế **Episodic Training**.
- Không sử dụng trọng số phân loại cố định; chỉ cập nhật tham số $\theta$ của TD-ResNet.
- Bộ tối ưu Adam, StepLR scheduler, hàm mất mát Cross Entropy trên khoảng cách âm bản (Negative Distance Logits).
- Toàn bộ tham số được quản lý qua `configs/experiments/exp_001_5way_5shot.yaml`.

---

## 10. Evaluation (Đánh giá mô hình)
- Kiểm thử trên **10 từ khóa hoàn toàn chưa từng xuất hiện trong quá trình huấn luyện**.
- Đo lường trên các cấu hình:
  - 5-way 1-shot & 5-way 5-shot.
  - 10-way 1-shot & 10-way 5-shot.
- Báo cáo kết quả kèm khoảng tin cậy 95% (95% Confidence Interval) qua $200$ episodes kiểm thử.

---

## 11. Raw vs. Normalized Experiment (Thực nghiệm Tiền xử lý)
So sánh định lượng và trực quan ảnh hưởng của các bước tiền xử lý:
- Dạng sóng tín hiệu & tỷ lệ đỉnh biên độ.
- Biểu đồ năng lượng RMS.
- Phổ Log Mel-spectrogram tương ứng.

---

## 12. Demo (Chương trình kiểm thử & Cầu nối Assistant)
Giao diện dòng lệnh kiểm thử nhanh:
```bash
python scripts/demo.py \
    --support demo/support/ \
    --query demo/samples/query.wav \
    --checkpoint checkpoints/exp_001/best.pt
```
Inference Engine này cung cấp hàm API `predict()` có thể nhúng trực tiếp vào vòng lặp mic của ứng dụng Desktop Voice AI Assistant.

---

## 13. Repository Structure (Cấu trúc mã nguồn)
```text
voice-AI-assistant/
├── configs/
│   ├── data/                 # gsc.yaml, audiomnist.yaml, fsc.yaml
│   ├── preprocessing/        # raw.yaml, normalized.yaml
│   ├── model/                # td_resnet.yaml, prototypical.yaml
│   └── experiments/          # exp_001_5way_5shot.yaml
├── src/
│   ├── data/                 # audit.py, preprocessing.py, datasets.py, episodic_sampler.py
│   ├── features/             # audio_features.py, visualization.py
│   ├── models/               # td_resnet.py, prototypical_network.py, distances.py
│   ├── training/             # trainer.py, losses.py, metrics.py
│   ├── evaluation/           # evaluate.py, embedding_analysis.py
│   └── utils/                # config.py
├── scripts/                  # audit_dataset.py, compare_raw_normalized.py, train.py, evaluate.py, demo.py
├── notebooks/                # 6 Notebooks tương ứng báo cáo môn học (01 -> 06)
├── reports/                  # data_audit/, figures/, tables/, results/
├── demo/                     # support/ (thư mục mẫu từ khóa), samples/
├── docs/                     # architecture.md, dataset.md, experiments.md
├── tests/                    # test_pipeline.py
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 14. Installation (Cài đặt)
```bash
# Tạo môi trường Conda
conda env create -f environment.yml
conda activate speech-fewshot-kws

# Hoặc cài đặt qua pip
pip install -r requirements.txt
```

---

## 15. Kaggle Usage (Sử dụng trên Kaggle)
Repository được thiết kế tương thích hoàn toàn với Kaggle Notebooks. Bạn chỉ cần clone repo và import các module từ `src/`. Tất cả các notebook trong `notebooks/` đều có tiêu đề cell đánh dấu rõ ràng (`REPORT 2.2`, `REPORT 2.4`, `REPORT 2.5`, `REPORT 3.x`).

---

## 16. Current Progress (Tiến độ hiện tại)
- [x] Thiết lập cấu trúc repository chuẩn khoa học.
- [x] Xây dựng module kiểm định dữ liệu tự động (`src/data/audit.py`).
- [x] Hoàn thiện pipeline tiền xử lý dạng bật/tắt (`src/data/preprocessing.py`).
- [x] Hoàn thiện trích xuất đặc trưng STFT, Mel, MFCC, RMS, ZCR (`src/features/audio_features.py`).
- [x] Hoàn thiện Episodic Sampler đảm bảo Support và Query không trùng lặp (`src/data/episodic_sampler.py`).
- [x] Xây dựng mô hình TD-ResNet và Prototypical Network (`src/models/`).
- [x] Xây dựng Episodic Trainer và Evaluator (`src/training/`, `src/evaluation/`).
- [x] Xây dựng bộ CLI scripts và 6 Kaggle Notebooks chuẩn báo cáo học thuật.
- [ ] Huấn luyện mô hình trên Kaggle GPU và ghi nhận số liệu thực tế.

---

## 17. Experimental Results (Kết quả thực nghiệm)

> ⚠️ **Nguyên tắc khoa học:** Tuyệt đối không tạo dựng số liệu giả. Các giá trị dưới đây sẽ được cập nhật tự động từ log thực nghiệm trên Kaggle.

| Cấu hình Thử nghiệm | Số Way ($N$) | Số Shot ($K$) | Mean Accuracy (%) | 95% Confidence Interval | Độ trễ suy luận (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 5-way 1-shot | 5 | 1 | `TBD_FROM_EXPERIMENT` | `TBD_FROM_EXPERIMENT` | `TBD_FROM_EXPERIMENT` |
| 5-way 5-shot | 5 | 5 | `TBD_FROM_EXPERIMENT` | `TBD_FROM_EXPERIMENT` | `TBD_FROM_EXPERIMENT` |
| 10-way 1-shot | 10 | 1 | `TBD_FROM_EXPERIMENT` | `TBD_FROM_EXPERIMENT` | `TBD_FROM_EXPERIMENT` |
| 10-way 5-shot | 10 | 5 | `TBD_FROM_EXPERIMENT` | `TBD_FROM_EXPERIMENT` | `TBD_FROM_EXPERIMENT` |

---

## 18. Limitations (Hạn chế hiện tại)
- Mô hình hiện tại giả định các từ khóa được cắt tỉa trong khoảng thời lượng ~1 giây.
- Chưa tích hợp cơ chế phát hiện từ chối ngoài tập (Unknown / Open-set thresholding) trong pha suy luận thời gian thực.

---

## 19. Future Work (Hướng phát triển tiếp theo - Hoàn thiện Voice AI Assistant)
1. **Wake-up Word Background Service:** Xây dựng background listener với mức tiêu thụ CPU thấp để lắng nghe từ khóa kích hoạt máy tính.
2. **Intent & Command Matcher:** Mở rộng không gian Prototype để đối chiếu các khẩu lệnh điều khiển desktop (mở trình duyệt, mở ứng dụng lập trình, chỉnh âm lượng,...).
3. **Antigravity Execution Bridge:** Kết nối API với hệ thống Antigravity để biến khẩu lệnh thành các hành động tự động hóa trên máy tính cá nhân.

---

## 20. References (Tài liệu tham khảo)
1. Snell, J., Swersky, K., & Zemel, R. (2017). *Prototypical networks for few-shot learning*. Advances in Neural Information Processing Systems (NeurIPS).
2. Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*. arXiv preprint arXiv:1804.03209.
3. Vygon, R., & Lysova, N. (2021). *Learning to Detect Open-Set Keywords in Few-Shot Setting*. Interspeech.
