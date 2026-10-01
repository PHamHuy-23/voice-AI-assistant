# Kiến trúc hệ thống: TD-ResNet & Prototypical Network

Tài liệu này mô tả chi tiết nguyên lý hoạt động, cấu tạo toán học và luồng dữ liệu của hệ thống **Few-Shot Keyword Spotting (FS-KWS)** và cách hệ thống liên kết với ứng dụng **Desktop Voice AI Assistant**.

---

## 1. Luồng xử lý tổng thể (End-to-End Pipeline)

```
[Audio Input] (.wav / microphone)
      │
      ▼
[Audio Preprocessor] (Mono, Resample 16kHz, Trim Silence, Amplitude Norm, Pad/Crop 16000)
      │
      ▼
[Feature Extractor] (64 Mel-bands Log Mel-Spectrogram: F=64, T=101)
      │
      ▼
[TD-ResNet Embedding Backbone] (Temporal Dilated Convolutions, Global Pooling)
      │
      ▼ Vector embedding z in R^128 (L2 Normalized)
      │
┌─────┴────────────────────────────────┐
│ Prototypical Network Inference Head  │
│  - Compute class prototypes c_k      │
│  - Compute Euclidean distances       │
│  - Softmax Confidence Scoring        │
└─────┬────────────────────────────────┘
      │
      ▼
[Predicted Keyword & Confidence Score]
      │
      ▼ (Tích hợp tương lai)
[Desktop Voice Assistant / Antigravity Dispatcher]
  -> Phân loại Wake-word ("Hey Assistant")
  -> Phân loại Lệnh ("Open VSCode", "Open YouTube", "Tell Time")
  -> Gửi IPC command tới Antigravity / Hệ điều hành để thực thi.
```

---

## 2. Temporal Dilated ResNet (TD-ResNet)

### 2.1. Đặt vấn đề
Các mạng CNN 2D truyền thống xử lý Mel-spectrogram như ảnh tĩnh, bỏ qua tính chất tuần tự nhân quả và phụ thuộc ngữ cảnh dài theo thời gian của âm vị tiếng nói. Để khắc phục, **TD-ResNet** áp dụng:
1. **Temporal Dilated Convolutions**: Dilation chỉ tăng dọc theo trục thời gian (Time axis $T$), mở rộng vùng quan sát (Receptive Field) mà không làm tăng số lượng tham số hay kích thước ma trận trọng số.
2. **Residual Connections**: Đường truyền tắt dạng $y = \mathcal{F}(x) + x$ giải quyết triệt để vấn đề triệt tiêu gradient (vanishing gradient) khi mạng sâu.
3. **L2 Normalization**: Chuẩn hóa vector đầu ra về mặt cầu đơn vị $||z||_2 = 1$, giúp khoảng cách Euclidean tương đương với Cosine similarity, ổn định không gian embedding.

---

## 3. Prototypical Network (ProtoNet)

### 3.1. Nguyên lý Prototype
Khác với Siamese Network (so sánh từng cặp $O(N \times K)$) hay Matching Network (attention trên toàn bộ Support Set), Prototypical Network biểu diễn mỗi từ khóa bằng một **vector đại diện duy nhất (Prototype)**:

$$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\theta(x_i)$$

Trong đó:
- $S_k$: Tập các mẫu Support của từ khóa thứ $k$ ($|S_k| = K$).
- $f_\theta$: TD-ResNet embedding network với trọng số $\theta$.

### 3.2. Dự đoán và Hàm mất mát
Với mỗi mẫu Query $x_q$, khoảng cách tới từng prototype $c_k$ được tính bằng khoảng cách Euclid bình phương:

$$d(f_\theta(x_q), c_k) = \|f_\theta(x_q) - c_k\|_2^2$$

Xác suất Query thuộc về từ khóa $k$:

$$p(y = k \mid x_q) = \frac{\exp(-d(f_\theta(x_q), c_k) / \tau)}{\sum_{k'} \exp(-d(f_\theta(x_q), c_{k'}) / \tau)}$$

Hàm mất mát được tối ưu qua lan truyền ngược (Backpropagation) để cập nhật tham số $\theta$ của TD-ResNet:

$$\mathcal{L}(\theta) = - \log p(y = y^* \mid x_q)$$

---

## 4. Episodic Training Scheme
- Mỗi epoch gồm $E$ episodes.
- Mỗi episode chọn ngẫu nhiên $N$ từ khóa (ví dụ $N = 5$).
- Với mỗi từ khóa, chọn $K$ mẫu Support và $Q$ mẫu Query sao cho $S \cap Q = \emptyset$.
- Nhãn được gán động trong phạm vi episode từ $0$ đến $N - 1$.
- Giúp mô hình rèn luyện khả năng so sánh khoảng cách và phân loại từ khóa mới mà không phụ thuộc vào lớp Softmax cố định.
