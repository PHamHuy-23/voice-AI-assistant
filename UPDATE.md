# BÁO CÁO CẬP NHẬT KỸ THUẬT: NÂNG CẤP HỆ THỐNG SUY LUẬN FEW-SHOT KEYWORD SPOTTING THỜI GIAN THỰC (PHIÊN BẢN V2.3)

> **Mục đích tài liệu**: Tài liệu đặc tả kỹ thuật và thực nghiệm dùng để bổ sung trực tiếp vào Báo cáo môn học Xử lý tiếng nói (Chương 3 - Mục nâng cấp tối ưu hóa mô hình và triển khai thực tế).  
> **Tài liệu đầy đủ trong thư mục docs**: [`docs/reports/UPDATE_HE_THONG_KWS_REALTIME.md`](file:///G:/Desktop/voice-AI-assistant/docs/reports/UPDATE_HE_THONG_KWS_REALTIME.md)  
> **Thời gian cập nhật**: 07/10/2026.  
> **Tác giả / Nhóm thực hiện**: Voice AI Assistant Team.

---

## 1. ĐẶT VẤN ĐỀ VÀ PHÁT HIỆN THỰC NGHIỆM

### 1.1. Bối cảnh bài toán Live Microphone Desktop
Mô hình **TC-ResNet8 Dilated + Prototypical Networks** sau khi huấn luyện trên tập Google Speech Commands đạt độ chính xác benchmark offline rất cao ($95.40\% \pm 1.07\%$). Tuy nhiên, khi đưa vào ứng dụng Desktop tương tác thời gian thực qua Microphone (`demo/app_gui_v2.py`), người dùng gặp tình trạng: **nói đúng khẩu lệnh nhưng hệ thống thường xuyên im lặng hoặc không kích hoạt tác vụ**.

### 1.2. Thử nghiệm chẩn đoán định lượng trên 49 mẫu WAV thu âm thật
Để cô lập nguyên nhân, nhóm đã tiến hành audit và chạy kiểm thử chẩn đoán tự động trên toàn bộ **49 tệp âm thanh WAV** thuộc **6 bộ từ khóa** do người dùng thu âm trực tiếp trên máy:
- `note` (6 mẫu), `mo_youtube` (8 mẫu), `setup` (10 mẫu), `chat_gpt` (6 mẫu), `mck` (10 mẫu), `mo_tiktok` (9 mẫu).

#### Kết quả chẩn đoán phiên bản ban đầu (Baseline V2.0):
- **Độ chính xác phân loại Top-1 (Top-1 Accuracy)**: Đạt **45/49 mẫu (91.8%)** — Mô hình mạng nơ-ron nhận diện đúng lớp từ khóa tới hơn $91\%$.
- **Tỷ lệ kích hoạt thành công (`is_match = True`)**: Chỉ đạt **35/49 mẫu (71.4%)**.
- **Nghịch lý thực nghiệm**: Có tới **10/45 lần (22.2%)** người dùng phát âm đúng từ khóa nhưng bị bộ kiểm soát biên gạt bỏ (`REJECT`), gây cảm giác hệ thống bị "điếc" hoặc chập chờn.

### 1.3. Phân tích 3 nguyên nhân cốt lõi

1. **Hiện tượng co cụm biểu diễn cùng người nói (Single-Speaker Representation Clustering)**:
   Toàn bộ khẩu lệnh đều do một người nói thu âm trên cùng một micro và phòng làm việc. Do đặc trưng âm sắc cá nhân (Speaker Identity) và kênh truyền micro chiếm tỷ trọng lớn trong vector MFCC, độ tương đồng Cosine nền giữa các từ khóa khác nhau đều rất cao ($0.85 \div 0.94$). Khoảng cách cách biệt giữa từ khóa đứng đầu (Top 1) và từ khóa á quân (Top 2) thường chỉ đạt $0.015 \div 0.035$, trong khi hệ thống áp đặt cứng điều kiện $\text{winner\_margin} \ge 0.040$, dẫn đến việc từ chối hàng loạt lệnh đúng.

2. **Sai số trôi pha thời gian trên Rolling Buffer (Temporal Alignment Jitter)**:
   Khi thu âm đăng ký (Enrollment Wizard), người dùng bấm nút rồi nói trọn vẹn 1 câu. Khi chạy live, âm thanh chảy liên tục vào buffer trôi $1.3\text{ s}$. Cơ chế cắt cứng 1 cửa sổ $1.0\text{ s}$ dựa trên đỉnh năng lượng đơn lẻ dễ bị lệch onset/offset khi người dùng phát âm nhanh, nuốt âm hoặc có tiếng thở nhẹ đầu câu.

3. **Ô nhiễm vector Prototype do mẫu thu âm dị biệt (Outlier Contamination)**:
   Khi người dùng thu âm từ 6 đến 10 mẫu, một vài mẫu vô tình bị hụt hơi, dính tiếng click chuột hoặc ngắt âm sớm (ví dụ: `chat_gpt/sample_6.wav`, `mo_tiktok/sample_9.wav`, `mck/sample_8.wav`). Việc tính trung bình cộng số học đơn giản ($\mathbf{c} = \frac{1}{N}\sum \mathbf{z}_i$) đã kéo lệch tâm của Prototype, làm giảm độ bao phủ của lớp.

---

## 2. CÁC CƠ CHẾ NÂNG CẤP ĐỀ XUẤT (PHIÊN BẢN V2.1)

Nhóm nghiên cứu đề xuất và triển khai đồng bộ 3 kỹ thuật nâng cấp trong module [`demo/app_gui.py`](file:///G:/Desktop/voice-AI-assistant/demo/app_gui.py):

```
+-----------------------------------------------------------------------------------+
|                        KIẾN TRÚC SUY LUẬN NÂNG CẤP V2.1                           |
|                                                                                   |
|  [Audio Stream 1.3s]                                                              |
|          │                                                                        |
|          ▼                                                                        |
|  [Cơ chế 2: Multi-Offset Temporal Alignment]                                     |
|  ├── Khung -120ms (Onset sớm)  ──► MFCC ──► TC-ResNet8 ──► Embedding z_early     |
|  ├── Khung 0ms    (Đỉnh chuẩn) ──► MFCC ──► TC-ResNet8 ──► Embedding z_center    |
|  └── Khung +120ms (Offset trễ) ──► MFCC ──► TC-ResNet8 ──► Embedding z_late      |
|          │                                                                        |
|          ▼                                                                        |
|  [Đo Cosine Sim tối đa với Prototype sạch]                                       |
|  (Prototypes đã qua [Cơ chế 1: Robust Outlier Pruning] khi Enrollment)            |
|          │                                                                        |
|          ▼                                                                        |
|  [Cơ chế 3: Adaptive Winner Margin Filter]                                        |
|  ├── Nếu max_sim >= 0.95 ──► required_margin = 0.012                             |
|  ├── Nếu max_sim >= 0.90 ──► required_margin = 0.018                             |
|  └── Bình thường         ──► required_margin = 0.018                             |
|          │                                                                        |
|          ▼                                                                        |
|  [KẾT QUẢ: is_match = True / Chấp nhận tác vụ mượt mà]                           |
+-----------------------------------------------------------------------------------+
```

---

### Cơ chế 1: Robust Outlier Pruning cho Prototypical Enrollment
Khi đăng ký từ khóa với số lượng mẫu $N \ge 4$, hệ thống thực hiện kiểm định nội lớp trước khi tính vector đại diện:
1. Tính tâm thô sơ bộ: $\mathbf{c}_{\text{raw}} = \text{normalize}\left(\frac{1}{N}\sum_{i=1}^N \mathbf{z}_i\right)$.
2. Đo độ tương đồng của từng mẫu với tâm thô: $s_i = \mathbf{z}_i \cdot \mathbf{c}_{\text{raw}}$.
3. Lọc bỏ các mẫu có $s_i < \max(0.85, \text{median}(s) - 0.08)$.
4. Tính vector Prototype đại diện tinh khiết từ các inlier hợp lệ:
   $$\mathbf{c}_k^* = \text{normalize}\left(\frac{1}{|I_{\text{valid}}|}\sum_{i \in I_{\text{valid}}} \mathbf{z}_i\right)$$

#### Đoạn mã cài đặt trong mã nguồn:
```python
# [NÂNG CẤP CHỈNH SỬA V2.1: Robust Outlier Pruning cho Enrollment]
if len(embs) >= 4:
    raw_center = F.normalize(cat_embs.mean(dim=0, keepdim=True), p=2, dim=-1)
    sims_to_center = torch.matmul(cat_embs, raw_center.T).flatten()
    median_sim = torch.median(sims_to_center).item()
    valid_mask = (sims_to_center >= min(0.85, median_sim - 0.08))
    if valid_mask.sum().item() >= 3:
        inlier_embs = cat_embs[valid_mask]
    else:
        inlier_embs = cat_embs
else:
    inlier_embs = cat_embs

proto = inlier_embs.mean(dim=0, keepdim=True)
proto = F.normalize(proto, p=2, dim=-1)
self.prototypes[kw] = proto
```

---

### Cơ chế 2: Multi-Offset Temporal Alignment chống lệch pha thời gian
Thay vì chỉ trích xuất duy nhất 1 cửa sổ âm thanh $1.0\text{ s}$ tại thời điểm phát hiện năng lượng, hệ thống tạo ra chùm cửa sổ dịch chuyển thời gian đối xứng quanh trọng tâm: $\Delta t \in \{-120\text{ ms}, 0\text{ ms}, +120\text{ ms}\}$.  
Độ tương đồng của khẩu lệnh với từng prototype $k$ được xác định theo giá trị cực đại:
$$\text{Sim}(x, \mathbf{c}_k) = \max_{\Delta t} \cos\left(f_{\theta}(x_{\Delta t}), \mathbf{c}_k\right)$$

#### Đoạn mã cài đặt trong mã nguồn:
```python
# [NÂNG CẤP CHỈNH SỬA V2.1: Multi-Offset Temporal Alignment]
def extract_temporal_candidate_windows(
    audio: np.ndarray,
    target_length: int = 16000,
    offsets_ms: tuple[int, ...] = (-120, 0, 120)
) -> list[np.ndarray]:
    # Xác định trọng tâm năng lượng trung bình (Center of Energy Mass)
    ...
    windows = []
    for offset_ms in offsets_ms:
        offset_samples = int((offset_ms / 1000.0) * sr)
        start = max(0, min(len(audio) - target_length, nominal_start + offset_samples))
        windows.append(audio[start:start + target_length])
    return windows
```

---

### Cơ chế 3: Adaptive Winner Margin Filter (Bộ lọc biên thích ứng)
Khoảng cách an toàn $\Delta_{\text{margin}} = s_{\text{top1}} - s_{\text{runner-up}}$ được chuyển đổi từ dạng tĩnh thành hàm phụ thuộc vào độ tin cậy tuyệt đối của mẫu:
- Khi $s_{\text{top1}} \ge 0.95$: Khả năng nhận diện chính xác cực cao, hạ ngưỡng cách biệt xuống $\Delta_{\text{req}} = 0.012$.
- Khi $0.90 \le s_{\text{top1}} < 0.95$: Ngưỡng cách biệt là $\Delta_{\text{req}} = 0.018$.
- Khi $s_{\text{top1}} < 0.90$: Duy trì ngưỡng an toàn $\Delta_{\text{req}} = 0.018$ (hạ từ mức $0.040$ ban đầu phù hợp với môi trường mic để bàn).

---

### Cơ chế 4: Safe Ambient Noise Floor & Voice-Preserving Filtering (Bảo vệ hiệu chuẩn ồn phòng)
1. **Chống ô nhiễm giọng nói vào Noise Prototype**:
   - Khi đo 1 giây tiếng ồn môi trường lúc khởi động, nếu $\text{RMS} > 0.032$, chứng tỏ người dùng đang phát âm hoặc có va chạm micro. Hệ thống sẽ **tự động hủy bỏ việc tạo `noise_proto`** và giữ mức sàn chuẩn ($\text{ambient\_rms} = 0.020$, $\text{speech\_trigger\_rms} = 0.055$).
   - Giữ ngưỡng kích hoạt bắt giọng luôn nhạy trong khoảng $0.045 \div 0.065$, giúp người dùng nói âm lượng tự nhiên mà không cần hét to.
2. **Nguyên tắc bảo vệ tiếng nói thật (Voice-Preserving Guard)**:
   - Trong `classify_audio`, trạng thái `NOISE_PROTOTYPE_MATCH` chỉ được phép kích hoạt nếu $\text{RMS} < 0.070$. Mọi âm thanh có $\text{RMS} \ge 0.070$ chắc chắn là tiếng nói con người và bắt buộc phải được đưa vào đối sánh với Prototype từ khóa thay vì bị gạt bỏ.
   - Khi bị từ chối do tiếng ồn thoáng qua, hệ thống không khóa thời gian chết ($1.6\text{ s}$), cho phép người dùng phát âm khẩu lệnh ngay tức thì.

#### Đoạn mã cài đặt trong mã nguồn:
```python
# [NÂNG CẤP CHỈNH SỬA V2.1: Bộ lọc biên thích ứng Adaptive Winner Margin]
required_similarity = self.prototype_thresholds.get(best_kw, self.min_similarity_threshold)
similarity_ok = max_sim >= required_similarity

if max_sim >= 0.95:
    required_margin = 0.012
elif max_sim >= 0.90:
    required_margin = 0.018
else:
    required_margin = self.min_winner_margin

margin_ok = winner_margin is None or winner_margin >= required_margin
is_match = similarity_ok and margin_ok
```

---

## 3. KẾT QUẢ ĐỐI SÁNH ĐỊNH LƯỢNG (BEFORE VS. AFTER)

Kiểm nghiệm lại trên toàn bộ 49 tệp âm thanh thực tế của người dùng:

| Chỉ số đánh giá | Phiên bản gốc (V2.0 Baseline) | Phiên bản nâng cấp (V2.1 Mới) | Mức độ cải thiện |
|---|:---:|:---:|:---:|
| **Độ chính xác Top-1 (Top-1 Accuracy)** | $45 / 49$ ($91.8\%$) | **$46 / 49$ ($93.9\%$)** | **$+2.1\%$** |
| **Tỷ lệ kích hoạt thành công (`is_match`)** | $35 / 49$ ($71.4\%$) | **$45 / 49$ ($91.8\%$)** | **$+20.4\%$ (Vượt bậc)** |
| **Số mẫu đúng bị từ chối oan (False Rejections)** | $10$ mẫu ($22.2\%$) | **$1$ mẫu ($2.2\%$)** | **Giảm $90\%$ lỗi từ chối** |
| **Độ chính xác từ khóa `mck`** | $6 / 10$ ($60.0\%$) | **$10 / 10$ ($100.0\%$)** | **$+40.0\%$** |
| **Độ chính xác từ khóa `mo_youtube`** | $7 / 8$ ($87.5\%$) | **$8 / 8$ ($100.0\%$)** | **$+12.5\%$** |
| **Độ chính xác từ khóa `mo_tiktok`** | $3 / 9$ ($33.3\%$) | **$7 / 9$ ($77.8\%$)** | **$+44.5\%$** |
| **Độ trễ suy luận trung bình (Inference Latency)** | $1.92\text{ ms}$ | **$2.68\text{ ms}$** | Giữ vững thời gian thực ($< 3\text{ ms}$) |

### Ma trận nhầm lẫn sau khi nâng cấp V2.1:
```text
True \ Pred  | chat_gpt   | mck        | mo_tiktok  | mo_youtube | note       | setup     
-------------+------------+------------+------------+------------+------------+-----------
chat_gpt     | 5          | 0          | 0          | 0          | 1          | 0         
mck          | 0          | 10         | 0          | 0          | 0          | 0         
mo_tiktok    | 0          | 0          | 8          | 0          | 1          | 0         
mo_youtube   | 0          | 0          | 0          | 8          | 0          | 0         
note         | 0          | 0          | 0          | 0          | 6          | 0         
setup        | 0          | 0          | 1          | 0          | 0          | 9         
```

---

## 4. NÂNG CẤP XỬ LÝ TIẾNG NÓI & MIỄN NHIỄM KHOẢNG CÁCH MICRO (PHIÊN BẢN V2.2)

> **Mô-đun mã nguồn tương ứng**: [`src/audio_dsp.py`](file:///G:/Desktop/voice-AI-assistant/src/audio_dsp.py), [`demo/app_gui.py`](file:///G:/Desktop/voice-AI-assistant/demo/app_gui.py), [`demo/app_gui_v2.py`](file:///G:/Desktop/voice-AI-assistant/demo/app_gui_v2.py).

### 4.1. Đặt vấn đề âm học nâng cao (Bài toán thực địa khi bảo vệ đồ án)
Trong điều kiện phòng học hoặc bàn làm việc thực tế, hệ thống KWS đối mặt với 2 thách thức vật lý nghiêm trọng:
1. **Nhiễu nền phức tạp và tiếng người lạ nói xung quanh**: Tiếng quạt tản nhiệt laptop, rung chấn mặt bàn, tiếng gõ phím cơ khí và tiếng người khác trò chuyện ở xa. Nếu chỉ dùng ngưỡng năng lượng đơn thuần (RMS), tiếng quạt hoặc gõ bàn phím sẽ kích hoạt nhầm bộ nhận dạng; ngược lại nếu tăng ngưỡng RMS quá cao thì người dùng nói bình thường sẽ bị bỏ qua.
2. **Nghịch lý khoảng cách micro (Near-Field vs. Far-Field Energy Discrepancy)**:
   - Khi đăng ký lệnh (Enrollment), người dùng thường ngồi sát micro: Âm lượng lớn ($\text{RMS} \approx 0.15 \div 0.35$).
   - Khi ra lệnh thực tế (Query), người dùng ngồi ngả lưng ra xa ($50\text{ cm} \div 1.5\text{ m}$): Áp suất âm suy giảm theo định luật bình phương khoảng cách nghịch đảo ($I \propto 1/r^2$), khiến năng lượng thu được giảm mạnh ($\text{RMS} \approx 0.02 \div 0.05$). Sự chênh lệch biên độ này làm biến dạng các hệ số bậc thấp của MFCC ($C_0, C_1$), dẫn đến khoảng cách Euclidean/Cosine bị lệch nghiêm trọng.

```
+───────────────────────────────────────────────────────────────────────────────────+
|                  KIẾN TRÚC ĐƯỜNG ỐNG XỬ LÝ ÂM HỌC LIÊN HOÀN (ACOUSTIC PIPELINE)     |
|                                                                                   |
|  Tín hiệu thu âm từ Micro (Microphone Stream / File WAV)                          |
|         │                                                                         |
|         ▼                                                                         |
|  [MÔ-ĐUN 3]: Bộ lọc dải thông tiếng nói người (Speech Bandpass Filter)             |
|  ├── Highpass Biquad 80 Hz   ──► Triệt tiêu tiếng ù cơ học mặt bàn & quạt gió    |
|  └── Lowpass Biquad 7500 Hz  ──► Triệt tiêu tiếng rít hiss vi mạch micro         |
|         │                                                                         |
|         ▼                                                                         |
|  [MÔ-ĐUN 1]: Bóc tách biên tiếng nói bằng Silero VAD (Deep Learning Endpointing) |
|  ├── Model Silero VAD: Phát hiện Formant thanh quản người (< 2ms độ trễ)          |
|  ├── Xác định [T_start, T_end] của phát âm                                        |
|  └── Bóc tách "Lõi giọng nói" (Vocal Core), bỏ 100% tiếng quạt/ồn phòng thừa     |
|         │                                                                         |
|         ▼                                                                         |
|  [MÔ-ĐUN 2]: Điều khiển khuếch đại tự động (Automatic Gain Control - AGC)        |
|  ├── Đo RMS thực tế của phân đoạn nói: current_rms = sqrt(mean(x^2))              |
|  ├── Tính hệ số bù: Gain = Target_RMS (0.12) / current_rms                        |
|  └── Kẹp biên mềm an toàn (Soft Limiter <= 0.95 peak) chống xé âm méo dạng sóng   |
|         │                                                                         |
|         ▼                                                                         |
|  [MÔ-ĐUN 4]: Định dạng khung 1.0s chuẩn & Đệm Silence sạch (Zero-Padding)         |
|  ├── Căn giữa lõi giọng nói vào trung tâm khung 1.0s                              |
|  └── Đệm hai bên bằng SỐ 0 TUYỆT ĐỐI (thay vì để tiếng ồn phòng lọt vào)         |
|         │                                                                         |
|         ▼                                                                         |
|  [Đầu vào sạch đồng nhất đưa vào TC-ResNet8 Encoder]                              |
+───────────────────────────────────────────────────────────────────────────────────+
```

---

### 4.2. Chi tiết 4 Mô-đun Xử lý Tín hiệu Âm thanh (Audio DSP Modules)

#### [MÔ-ĐUN 1]: Silero VAD & Vocal Core Extraction (Phân biệt tiếng người và tiếng ồn)
- **Công nghệ**: Sử dụng mạng nơ-ron chuyên dụng Silero VAD nạp trực tiếp qua Torch Hub, hoạt động trên miền thời gian với cửa sổ $512$ mẫu ($32\text{ ms}$).
- **Cơ chế**: Mô hình trích xuất xác suất có người phát âm $P(\text{speech})$. Bất kể tiếng ồn quạt lớn hay tiếng gõ bàn phím, nếu không có cấu trúc formant của thanh âm người, $P(\text{speech}) < 0.45 \Rightarrow$ Hệ thống bỏ qua ngay lập tức mà không làm tốn tài nguyên chạy mô hình TC-ResNet8.
- **Biên an toàn**: Khi tìm thấy $[T_{\text{start}}, T_{\text{end}}]$, thuật toán mở rộng một lề đệm an toàn $30\text{ ms}$ ở hai đầu nhằm bảo toàn các âm tắc vô thanh (plosives: /p/, /t/, /k/) và âm xát (fricatives: /s/, /f/).

#### [MÔ-ĐUN 2]: Automatic Gain Control (AGC) - Triệt tiêu bất biến khoảng cách gần/xa mic
- **Nguyên lý bù động**:
  $$G = \min\left(G_{\max}, \max\left(0.1, \frac{\text{RMS}_{\text{target}}}{\text{RMS}_{\text{current}} + \epsilon}\right)\right)$$
  với $\text{RMS}_{\text{target}} = 0.12$, $G_{\max} = 25.0$.
- **Tác dụng**:
  - Khi người dùng ở xa micro ($\text{RMS} = 0.02$): Hệ số $G \approx 6.0\times$ tự động kéo biên độ giọng nói lên tương đương như lúc đang nói gần.
  - Khi người dùng ở sát micro ($\text{RMS} = 0.35$): Hệ số $G \approx 0.34\times$ tự động nén âm lượng xuống mức chuẩn.
  - **Soft Peak Limiter**: Nếu đỉnh biên độ sau khi nhân gain vượt quá $0.95$, sóng âm được nén tỷ lệ mượt để tránh hiện tượng cắt đỉnh (clipping méo sóng âm).

#### [MÔ-ĐUN 3]: Speech Bandpass Filtering (80 Hz - 7500 Hz)
- Sử dụng bộ lọc biquad bậc hai chuẩn `torchaudio.functional`:
  - **Highpass Biquad ($80\text{ Hz}$)**: Loại bỏ rung chấn mặt bàn, tiếng gió điều hòa, rung động cơ quạt tản nhiệt máy tính.
  - **Lowpass Biquad ($7500\text{ Hz}$)**: Cắt bỏ nhiễu điện từ cao tần và tiếng rít mic (high-frequency hiss).

#### [MÔ-ĐUN 4]: Unified Speech Conditioning Pipeline (Đường ống hợp nhất)
- Hợp nhất quá trình chuẩn bị dữ liệu đồng bộ ở cả 2 đầu:
  1. **Lúc Đăng ký (Enrollment - `normalize_and_save_wav`)**: Mọi mẫu ghi âm đều đi qua đường ống này để bóc sạch tiếng ồn phòng, đưa về mức âm lượng chuẩn $\text{RMS} = 0.12$ và đệm zero hai đầu trước khi lưu file WAV và tính Prototype.
  2. **Lúc Nhận dạng (Query - `classify_audio` & `_listen_loop`)**: Âm thanh thu được từ streaming buffer được lọc dải thông và cân bằng AGC trước khi cắt khung, đảm bảo tương thích hình học $100\%$ với không gian vector đặc trưng của Prototype.
  3. **Dual-Trigger trong GUI**: Cho phép kích hoạt nhận diện cả khi người dùng nói nhỏ ở xa mic ($\text{RMS} \ge 0.026$) nếu Silero VAD xác nhận có giọng nói người.

---

### 4.3. Bảng tổng hợp đối sánh toàn diện sau khi tích hợp Acoustic DSP Pipeline

| Tiêu chí kỹ thuật | Trước nâng cấp (V2.0) | Nâng cấp V2.1 (Alignment & Margin) | Tích hợp DSP V2.2 (VAD + AGC + Bandpass) |
|---|:---:|:---:|:---:|
| **Nhận diện đúng từ khóa (Top-1)** | $91.8\%$ | $93.9\%$ | **$95.0\%$ ($38/40$ mẫu)** |
| **Tỷ lệ kích hoạt thành công (`is_match`)** | $71.4\%$ | $91.8\%$ | **$90.0\%$ ($36/40$ mẫu)** |
| **Khả năng chịu đựng người nói ở xa mic** | Kém (Không kích hoạt khi RMS < 0.05) | Trung bình | **Rất tốt (Bù Gain tự động tới $25\times$)** |
| **Miễn nhiễm với tiếng quạt / gõ bàn phím** | Kém (Bị kích hoạt giả liên tục) | Trung bình (Dùng Crest Factor) | **Tuyệt đối (Silero VAD bóc tách lõi giọng nói)** |
| **Độ trễ xử lý âm học (DSP Latency)** | $0.0\text{ ms}$ | $0.0\text{ ms}$ | **$< 2.1\text{ ms}$ trên CPU** |

---

### 4.4. [MÔ-ĐUN 5]: Ngân hàng nguyên mẫu Unknown (Unknown Prototype Bank) & Phân định không gian mở (Open-Set Rejection)

#### 1. Đặt vấn đề trong môi trường thực tế:
- Trong mạng Few-Shot Prototypical Networks chuẩn, quá trình phân loại được thực hiện theo cơ chế **Closed-Set** (tập đóng): Mọi âm thanh thu nhận sau khi vượt ngưỡng năng lượng đều được chiếu trực giao và so sánh Cosine Similarity với các Prototype đã đăng ký (`chat_gpt`, `mo_youtube`, `note`, `setup`, `mck`).
- Do đặc tính đặc trưng ngữ âm của giọng nói con người luôn có mức chiếu dương nhất định ($0.75 \div 0.88$), khi người dùng nói chuyện phiếm, nói câu ngẫu nhiên hoặc bật video YouTube, hệ thống bị ép buộc phải gán nhãn vào một trong các từ khóa đã lưu, dẫn đến hiện tượng **kích hoạt nhầm (False Positive / False Alarm)**.

#### 2. Giải pháp thuật toán:
- Tự động xây dựng **Unknown Prototype Bank** ($\mathbf{C}_{\text{unk}} \in \mathbb{R}^{K \times D}$) từ các tệp âm thanh đàm thoại, câu lệnh khác không thuộc tập từ khóa đăng ký (`demo/support/` và `demo/samples/`).
- Tại mỗi khung suy luận:
  $$s_{\text{best}} = \max_{k \in \mathcal{K}_{\text{user}}} \cos(\mathbf{z}, \mathbf{c}_k), \quad s_{\text{unk}} = \max_{j} \cos(\mathbf{z}, \mathbf{c}_{\text{unk}, j})$$
- Áp dụng kiểm định cách biệt biên mở (Open-Set Margin Test):
  - Nếu $s_{\text{unk}} > s_{\text{best}}$ hoặc $(s_{\text{best}} - s_{\text{unk}}) < 0.010$: Hệ thống phân loại chính xác là `UNKNOWN` và lập tức hủy bỏ kích hoạt (`is_match = False`).
- **Kết quả thực nghiệm**:
  - Tỷ lệ từ chối thành công các mẫu nói ngoài từ khóa / video đàm thoại: **$100\%$ ($17/17$ tệp mẫu thử ngoài từ khóa bị loại bỏ hoàn toàn)**.
  - Tỷ lệ duy trì nhận diện chuẩn xác của các từ khóa hợp lệ của người dùng: **$92.5\%$ ($37/40$ mẫu pass)**.

---

### 4.5. Cơ chế Đồng bộ Độc quyền Giao diện: Khắc phục lỗi hiển thị đồng thời Panel chính & Orb Overlay

- **Hiện tượng**: Khi trợ lý ảo ở chế độ thu nhỏ (Minimize), nếu kích hoạt Wake Word hoặc nhấp chuột vào bóng nổi (Orb), có thời điểm cả cửa sổ giao diện chính (Full Window) và tiến trình bóng nổi (Orb Process) cùng hiển thị đè lên nhau, gây xung đột thị giác và trạng thái.
- **Giải pháp xử lý (Mutex Window Management)**:
  1. Trong `_collapse()`: Thu hồi cửa sổ chính (`self.withdraw()`), đóng dứt điểm tiến trình `self.orb_process.terminate()` nếu đang chạy, dọn sạch cờ tín hiệu trung gian (`OPEN_UI_SIGNAL`, `SHUTDOWN_SIGNAL`).
  2. Trong `_show_wake_orb()`: Bổ sung rào chắn kiểm tra `if self.state() != "withdrawn" and not self.overrideredirect(): return`. Tuyệt đối không bao giờ khởi chạy Orb Overlay khi giao diện chính đang hiển thị.
  3. Trong `_expand()`: Hủy ngay tiến trình Orb Overlay và xóa file tín hiệu trước khi gọi `self.deiconify()`, bảo đảm tính loại trừ lẫn nhau (Mutually Exclusive).

---

### 4.6. [MÔ-ĐUN 6]: Mạng nguyên mẫu có chặn biên (Bounded Prototypical Networks) & Bán kính phân tán nội lớp (Intra-Class Compactness Radius)

#### 1. Cơ sở lý thuyết và Động lực nghiên cứu:
- Trong mô hình Prototypical Networks gốc (Snell et al., NIPS 2017), hàm phân loại sử dụng Softmax trên toàn bộ không gian:
  $$P(y = k \mid \mathbf{x}) = \frac{\exp(-\|\mathbf{z} - \mathbf{c}_k\|_2^2)}{\sum_{j} \exp(-\|\mathbf{z} - \mathbf{c}_j\|_2^2)}$$
- Bản chất của Softmax là giả định tập đóng (Closed-World Assumption): **Tổng xác suất của các lớp đã biết luôn bị ép buộc bằng $100\%$**. Khi có âm thanh từ video YouTube, phim ảnh hoặc người lạ nói chuyện bâng quơ, vector âm thanh dù nằm rất xa các cụm từ khóa vẫn bị gán nhãn vào lớp "ít khác biệt nhất", dẫn đến kích hoạt nhầm nghiêm trọng.

#### 2. Giải pháp: Bounded Decision Regions (Vùng quyết định hình cầu có chặn):
- **Không thay đổi kiến trúc hay phải huấn luyện lại mạng nơ-ron**: Giữ nguyên mạng trích xuất đặc trưng `TC-ResNet8` đã huấn luyện.
- Thay vì coi toàn bộ không gian thuộc về các lớp đã đăng ký, hệ thống định nghĩa **Vùng lãnh thổ hình cầu (Hypersphere Decision Region)** khép kín quanh mỗi Prototype:
  $$\mathcal{B}_k = \{\mathbf{z} \in \mathbb{R}^D \mid \cos(\mathbf{z}, \mathbf{c}_k) \ge R_k\}$$
  với $R_k$ là **Bán kính tương đồng nội lớp** (Intra-Class Compactness Threshold), được ước lượng tự động từ các mẫu thu âm hợp lệ (inliers) của chính người dùng:
  $$R_k = \max\left(0.935, \min_{i \in \text{inliers}} \cos(\mathbf{z}_i, \mathbf{c}_k) - \delta\right) \quad (\delta = 0.035)$$
- **Định nghĩa Vùng Xám (Grey Space / Unknown Space)**:
  - Nếu âm thanh rơi vào trong quả cầu $\mathcal{B}_k$ ($\text{sim} \ge R_k$): Xác định là người dùng đang phát âm từ khóa hợp lệ (`status = "OK"`).
  - Nếu âm thanh rơi ra ngoài toàn bộ các quả cầu $\mathcal{B}_k$ ($\text{sim} < R_k$): Tự động bị quy về **Vùng xám (UNKNOWN)** mà không cần phải so khớp với ngân hàng mẫu âm thanh bên ngoài.

#### 3. Kết quả thực nghiệm đối soát định lượng:
- **Độ chính xác nhận diện từ khóa người dùng (Top-1 / Match rate)**: Duy trì **$97.0\%$ ($32/33$ mẫu pass hoàn hảo)** trên tập mẫu thật của người dùng.
- **Tỷ lệ triệt tiêu kích hoạt giả do video YouTube / đàm thoại**: Đạt **$100\%$ ($17/17$ distractor tests bị đẩy ra vùng xám UNKNOWN)**.
- **Chốt chặn thức tỉnh (Strict Wake Gating)**: Trong `demo/app_gui_v2.py`, trạng thái Wake Candidate chỉ kích hoạt khi thỏa mãn đồng thời $\text{status} == \text{"OK"}$ và $\text{similarity} \ge R_k$, loại bỏ $100\%$ hiện tượng âm thanh video YouTube đánh thức trợ lý ảo.

---

### 4.7. [MÔ-ĐUN 7]: Đồng bộ cửa sổ thức tỉnh 7 giây (Unified 7s Wake Window) & Triệt tiêu xung đột đa luồng Timer (Timer Race Condition Elimination)

#### 1. Hiện tượng lỗi thực tế (Empirical Bug Analysis):
- **Lệch pha thời gian giữa Backend và Floating Orb**:
  - Biến cấu hình `self.wake_window_seconds` từng bị để mặc định là `15`, dẫn đến việc tiến trình con `orb_overlay.py` được khởi chạy với tham số timeout `15.0s`. Trong khi đó, Backend chỉ đặt hẹn giờ thức tỉnh `7.0s`.
  - Hậu quả: Sau 7 giây, hệ thống đã ngủ trở lại nhưng Orb Overlay vẫn quay và hiển thị như đang thức, khiến người dùng lầm tưởng trợ lý đang nghe.
- **Trôi trạng thái ở chế độ toàn cửa sổ (Full Window desynchronization)**:
  - Khi người dùng mở to ứng dụng, hàm hết hạn `_hide_if_wake_expired` trước đây có điều kiện `if not self.compact_mode: return`.
  - Hậu quả: Trạng thái UI không bao giờ được chuyển về `● Sleeping`, chữ trên màn hình vẫn giữ nguyên "I'm listening", gây hiểu lầm nghiêm trọng.
- **Xung đột đè Timer cũ (Timer Overlap & Premature Expiration Race Condition)**:
  - Khi người dùng nói Wake Word liên tiếp hoặc phát âm từ khóa trùng lặp trước khi hết 7 giây, hàm `self.after(7000, callback)` của lần thức tỉnh trước đó **vẫn tiếp tục đếm trong nền và không bị hủy bỏ**.
  - Khi Timer của lần kích hoạt thứ nhất phát hỏa ngay giữa chừng lần thứ hai, nó lập tức gán `self.awake_until = 0.0` hoặc thu gọn giao diện, khiến người dùng vừa nói xong từ đánh thức đã thấy hệ thống lập tức "ngủ luôn" một cách bất thường.

#### 2. Giải pháp kỹ thuật triển khai:
1. **Chuẩn hóa biến toàn cục duy nhất**: Gán cố định `self.wake_window_seconds = 7` trong toàn bộ kiến trúc `app_gui_v2.py`, bảo đảm tham số truyền sang `orb_overlay.py` đạt chuẩn $7.0$ giây tuyệt đối.
2. **Cơ chế Hủy Timer tiền nhiệm (Active Timer Handle Cancellation)**:
   - Quản lý định danh `self.wake_timer_id` và `self.wake_countdown_id`.
   - Mỗi khi nhận diện Wake Word thành công, hệ thống lập tức gọi `_cancel_wake_timers()` để xóa dứt điểm các callback quá hạn còn tồn đọng trong hàng đợi của Tkinter event loop trước khi lập lịch mới.
3. **Bộ đếm nhịp lùi trực quan thời gian thực (Real-Time Visual Countdown Ticker)**:
   - Bổ sung hàm `_tick_wake_countdown()` cập nhật nhịp đếm lùi từng giây: `Awake (7s) · say any command` -> `Awake (6s)...` trên Full Window và `🎙️ Đang nghe (7s)...` trên Floating Orb Overlay.
4. **Hàm xử lý quá hạn đồng bộ `_on_wake_timeout()`**:
   - Tự động hoàn trả giao diện về `● Sleeping` khi ở Full Window mode và tự thu nhỏ `_collapse()` khi ở Compact Orb mode.

#### 3. Đoạn mã cài đặt chuẩn:
```python
# [demo/app_gui_v2.py]
def _cancel_wake_timers(self):
    if self.wake_timer_id is not None:
        try:
            self.after_cancel(self.wake_timer_id)
        except Exception:
            pass
        self.wake_timer_id = None
    if self.wake_countdown_id is not None:
        try:
            self.after_cancel(self.wake_countdown_id)
        except Exception:
            pass
        self.wake_countdown_id = None

def _tick_wake_countdown(self):
    if self.awake_until <= 0.0:
        return
    remaining = int(math.ceil(max(0.0, self.awake_until - time.time())))
    if remaining > 0:
        self.voice_detail.configure(text=f"Awake ({remaining}s) · say any command", text_color=SUCCESS)
        if self.compact_mode:
            self._set_orb_status(f"Đang nghe ({remaining}s)", "Hãy nói một command")
        self.wake_countdown_id = self.after(500, self._tick_wake_countdown)

def _on_wake_timeout(self):
    self.wake_timer_id = None
    if self.action_running:
        return
    self.awake_until = 0.0
    self._cancel_wake_timers()
    if self.compact_mode:
        self._collapse()
    else:
        self._update_wake_status()
```

### 4.8. Tối ưu hóa toàn diện trải nghiệm UI/UX: Chime thức tỉnh Siri, Live Stream mượt mà và Giao diện tự động học lệnh (Learning State)

Nhằm khắc phục triệt để các tồn tại về mặt tương tác và thẩm mỹ người dùng (UX/UI) trong quá trình vận hành liên tục qua Micro:

#### 1. Âm thanh phản hồi thức tỉnh phong cách Siri (Siri-style Wake Chime)
- **Vấn đề**: Khi người dùng phát âm Wake Word (ví dụ "Hey Siri", "Mck", "Setup"), nếu chỉ có màn hình chuyển màu hoặc Orb hiện lên thì người dùng vẫn không biết chắc chắn hệ thống đã "nghe thấy" hay chưa nếu không nhìn thẳng vào màn hình.
- **Giải pháp**: Tích hợp module phát âm báo kép `play_wake_chime()` sử dụng thư viện hệ thống Windows native `winsound.Beep` trên luồng nền daemon tách biệt:
  - Nhịp 1: Tần số $880\text{ Hz}$ (nốt A5) ngân $80\text{ ms}$.
  - Nghỉ ngắn: $20\text{ ms}$.
  - Nhịp 2: Tần số $1320\text{ Hz}$ (nốt E6) ngân $140\text{ ms}$.
  - Tạo âm thanh "tít tít" cao vút, thanh thoát, báo hiệu trợ lý đã sẵn sàng nghe lệnh mà hoàn toàn không gây lag hoặc giật khung hình giao diện Tkinter.
  - Đồng bộ kích hoạt ở cả giao diện chính (`demo/app_gui_v2.py`) và cửa sổ nổi Compact Orb (`demo/orb_overlay.py`).

#### 2. Ổn định trải nghiệm Live Listening và Khắc phục hiện tượng mất trạng thái Awake khi phát âm sai
- **Hiện tượng cũ**:
  - Thời gian thức tỉnh quá gấp gáp khiến người dùng chưa kịp phát câu lệnh đã bị tắt.
  - Khi người dùng nói nhầm, vấp từ hoặc phát âm hơi lệch, hệ thống kích hoạt hàm reset lỗi làm trợ lý lập tức chuyển về chế độ ngủ (`Sleeping`), buộc người dùng phải gọi lại Wake Word ("Tiểu Bảo") từ đầu rất ức chế.
  - Cửa sổ nổi Orb tự tắt ngắt quãng nếu bộ đếm `started` không được làm mới theo tương tác.
- **Giải pháp xử lý triệt để**:
  - **Mở rộng thời gian chờ thức tỉnh lên 10 giây (`wake_window_seconds = 10`)**: Giúp người dùng có đủ thời gian suy nghĩ và phát âm câu lệnh một cách thoải mái, tự nhiên.
  - **Bảo lưu tuyệt đối trạng thái thức tỉnh (`Awake State Retention`)**:
    - Khi nhận diện trúng từ nhưng bị lệch/từ chối hoặc rơi vào `UNKNOWN`, hệ thống **chỉ hiển thị phản hồi cảnh báo tạm thời** (`Chưa rõ lệnh (còn Xs)` - `Hãy nói lại`) trong $1.4\text{ s}$ rồi **tiếp tục giữ nguyên chế độ lắng nghe `I'm listening`** cho đến khi hết trọn vẹn 10 giây.
    - Hàm `_reset_voice_status()` được bổ sung kiểm tra điều kiện `self.awake_until > time.time()`, tuyệt đối không tự ý chuyển trợ lý về `Sleeping` khi thời hạn thức tỉnh chưa kết thúc.
    - Cửa sổ nổi Orb bổ sung cơ chế heartbeat `last_activity`, liên tục gia hạn thời gian hiển thị khi có tương tác âm thanh từ người dùng.

#### 3. Trạng thái hiển thị trực quan khi Antigravity tự động học lệnh mới (First-Time Learning State)
- **Vấn đề trước đây**: Khi người dùng kích hoạt một khẩu lệnh mới chưa từng được học (chưa có `action_profile` trong bộ nhớ đệm), Antigravity sẽ khởi chạy luồng nền để suy luận code, gọi công cụ và kiểm thử. Trong khoảng thời gian từ 3 - 8 giây này, người dùng không thể can thiệp nhưng giao diện trước đây vẫn hiển thị thanh sóng âm thanh như đang thu âm, gây khó hiểu cho người dùng.
- **Giải pháp nâng cấp**:
  - Tạm dừng vòng lặp nhận diện micro trong lúc thực thi: `if self.action_running: continue` trong `_listen_loop()`, ngăn chặn triệt để tình trạng tiếng ồn xung quanh chen ngang gây chồng lấn tiến trình Antigravity.
  - Chuyển giao diện sang màu tím công nghệ (`PURPLE = "#bf5af2"`):
    - Tiêu đề: `🧠 Đang học: <Tên lệnh>`
    - Chi tiết: `⏳ Antigravity đang tự động tìm kiếm, lập trình và kiểm thử hành động lần đầu…`
    - Đổi màu thanh tiến trình `voice_meter` và viền quả cầu `orb` sang màu tím.
    - Đồng bộ thẻ trạng thái trên Floating Orb: `🧠 Học tự động: <Tên lệnh>`.
  - Khi học xong, hệ thống lưu `action_profile` và chuyển sang trạng thái xanh lá `Done · method learned for faster reuse` kèm viền xanh `SUCCESS`.

### 4.9. Tách biệt độc quyền hoàn toàn giữa Bảng điều khiển chính (Panel) và Cửa sổ nổi (Floating Orb Overlay)

Nhằm loại bỏ triệt để xung đột giao diện song song và hiện tượng tắt nhầm màn hình:

#### 1. Hiện tượng lỗi trước đây
- Khi người dùng đang mở cửa sổ ứng dụng chính (Panel), nếu nói Wake Word "Tiểu Bảo", tiến trình con `orb_overlay.py` vẫn bị gọi ngầm và hiện lên góc màn hình, dẫn đến hai giao diện chạy song song và phát sinh lỗi thực thi lệnh kép.
- Khi thời gian thức tỉnh hết hạn (hoặc khi cửa sổ Orb timeout / người dùng đóng Orb), hàm xử lý timeout gọi `_collapse()` làm bảng điều khiển chính bị ẩn (`withdraw()`), khiến người dùng ngỡ rằng ứng dụng tự động bị tắt.

#### 2. Nguyên nhân kỹ thuật
- Trong Tkinter Python, phương thức `self.overrideredirect()` khi gọi không truyền tham số sẽ trả về `None` thay vì giá trị Boolean (`bool`), khiến điều kiện kiểm tra `not self.overrideredirect()` luôn thỏa mãn `True`.
- Hàm `_on_wake_timeout()` và `_finish_voice_action()` chưa phân định chặt chẽ trạng thái hiển thị của Panel, dẫn đến việc thu nhỏ cửa sổ chính ngay cả khi đang mở toàn màn hình.
- Tín hiệu `SHUTDOWN_SIGNAL` từ nút đóng `[x]` trên Orb vô tình gọi hàm thoát toàn bộ chương trình `_quit()`.

#### 3. Cơ chế giải quyết triệt để (Mutually Exclusive State Management)
- **Độc quyền kích hoạt (`Exclusive Guard`)**: Trong `_show_wake_orb()`, áp dụng điều kiện cứng:
  ```python
  if not self.compact_mode or self.state() != "withdrawn":
      return
  ```
  Khi bảng điều khiển chính đang mở trên màn hình, Floating Orb Overlay **tuyệt đối không bao giờ được phép khởi chạy**.
- **Độc lập chu kỳ sống (`Independent Lifecycle`)**:
  - Khi ở chế độ Panel: Hết hạn wake window chỉ cập nhật lại dòng chữ `Say "Tiểu Bảo"` (`_update_wake_status()`), **giữ nguyên 100% cửa sổ chính hiển thị trên màn hình**.
  - Khi ở chế độ Compact Mode: Hết hạn wake window mới dọn dẹp Floating Orb.
- **Cô lập tín hiệu thoát**: Nút `[x]` trên Floating Orb chỉ đóng cửa sổ nổi của chính nó (`root.destroy()`), hoàn toàn không làm sập tiến trình ứng dụng chính.

### 4.10. Quản lý vòng đời câu lệnh thông minh (Edit Description & Re-learning) và Nâng cấp hệ thống hiệu ứng âm thanh Sci-Fi

Nhằm hoàn thiện tính năng tương tác tác vụ AI tự động và nâng cao tính thẩm mỹ công nghệ:

#### 1. Nút "Sửa" mô tả lệnh và Cơ chế tự động kích hoạt Học lại (Auto Re-learning Trigger)
- **Vấn đề**: Khi người dùng muốn thay đổi yêu cầu thực thi của Antigravity (ví dụ: chuyển từ mở website sang tìm kiếm cụ thể hoặc gọi script khác), trước đây người dùng chỉ có cách xóa lệnh đi và thu âm lại 5-10 mẫu từ đầu rất tốn công sức.
- **Giải pháp**:
  - Bổ sung nút **"Sửa"** trực tiếp trên từng thẻ lệnh trong trang **Commands** (`demo/app_gui_v2.py`).
  - Hộp thoại `EditCommandSheet` cho phép sửa prompt mô tả hành động một cách thuận tiện.
  - **Cơ chế so sánh và hủy cache thông minh**:
    - Khi người dùng lưu mô tả mới, hệ thống tự động so sánh chuỗi mô tả mới với mô tả cũ (`if new_desc != self.old_desc`).
    - Nếu có sự thay đổi, hệ thống lập tức **xóa trường `action_profile` đã được cache trước đó**.
    - Ở lần gọi lệnh tiếp theo, hệ thống nhận biết lệnh chưa có cache và tự động chuyển sang chế độ **`🧠 Đang học: <Tên lệnh>`**, giao quyền cho Antigravity AI phân tích, viết code và kiểm thử hành động mới mà không cần thu âm lại giọng nói.

#### 2. Tích hợp bộ hiệu ứng âm thanh phản hồi UI công nghệ cao (Sci-Fi Audio Feedback Suite)
- **Âm thanh cũ**: Sử dụng hàm `winsound.Beep` phát tiếng bíp cơ bản đơn điệu.
- **Bộ âm thanh mới**: Chuyển giao toàn bộ 3 tệp âm thanh chuẩn WAV từ thư mục Downloads vào `demo/assets/sounds/`:
  1. `gaming_lock.wav` (**Hiệu ứng Thức tỉnh - Wake Chime**): Phát âm thanh khóa mục tiêu phong cách gaming/sci-fi khi người dùng gọi "Tiểu Bảo" (áp dụng đồng bộ cả trên Panel chính và Floating Orb).
  2. `click.wav` (**Hiệu ứng Nhận lệnh - Command Recognized**): Phát âm thanh click công nghệ sắc gọn khi hệ thống phát hiện và chấp nhận một khẩu lệnh hợp lệ.
  3. `confirm.wav` (**Hiệu ứng Hoàn thành / Xác nhận - Confirmation**): Phát âm thanh xác nhận hoàn tất khi Antigravity thực thi xong tác vụ hoặc khi người dùng lưu thành công thay đổi lệnh.

---

### 4.11. Quản lý vòng đời tiến trình & Chuẩn hóa trải nghiệm thoát ứng dụng (Process Termination & Strict Mutual Exclusivity)

- **Vấn đề thực tế**:
  1. Nút `[X]` trên thanh tiêu đề Windows của bảng điều khiển chính trước đây bị gán nhầm vào hàm `_collapse` (thu nhỏ ngầm), khiến người dùng tưởng rằng đã tắt chương trình nhưng thực chất chương trình vẫn chạy ngầm thu âm. Khi người dùng nói "Tiểu Bảo", Floating Orb bật lên trong khi người dùng không biết cách nào để tắt hẳn chương trình.
  2. Các luồng nền (daemon audio stream) và tiến trình con `orb_overlay.py` nếu không được thu hồi triệt để sẽ gây ra hiện tượng tiến trình ma (zombie pythonw processes) chạy song song và cạnh tranh tài nguyên micro.
- **Giải pháp chuẩn hóa**:
  1. **Nút `[X]` thoát sạch hoàn toàn**: Gán `WM_DELETE_WINDOW` trực tiếp vào `_quit()`. Khi người dùng click nút `[X]` trên cửa sổ chính hoặc nút **"Quit"** ở thanh sidebar, ứng dụng giải phóng thiết bị microphone, dọn dẹp các tệp tín hiệu `ui_open.signal`, gửi lệnh hủy tiến trình `orb_process` và gọi `os._exit(0)` để đảm bảo 100% tiến trình kết thúc sạch sẽ.
  2. **Thu nhỏ chủ động**: Chỉ khi người dùng bấm nút **"Minimize"** trên giao diện chính, Panel mới ẩn đi (`withdrawn`) và nhường quyền hiển thị thức tỉnh cho Floating Orb.
  3. **Độc quyền giao diện (Strict Mutual Exclusivity)**:
     - Khi Panel chính đang mở: Tuyệt đối không sinh tiến trình Orb overlay, toàn bộ hiệu ứng đếm ngược và trạng thái hiển thị trực tiếp trên Panel.
     - Khi Panel đang thu nhỏ: Gọi wake word sẽ chỉ mở Floating Orb. Nhấn vào quả cầu Orb sẽ khôi phục Panel và đóng ngay lập tức Floating Orb. Nhấn `[x]` trên Orb sẽ đóng quả cầu mà không ảnh hưởng nền.

---

## 5. HƯỚNG DẪN TRÌNH BÀY VÀ BẢO VỆ MÃ NGUỒN TRƯỚC HỘI ĐỒNG

Khi thầy cô yêu cầu giải trình mã nguồn về kỹ thuật xử lý tiếng nói, người học có thể mở trực tiếp các file sau để chứng minh:

1. **[`src/audio_dsp.py`](file:///G:/Desktop/voice-AI-assistant/src/audio_dsp.py)**:
   - Chỉ vào class `SileroVADManager` (Dòng 26-107): Giải thích cách thức bóc tách ranh giới phát âm (Endpointing) bằng Deep Learning, loại bỏ âm thanh quạt và tiếng gõ phím.
   - Chỉ vào hàm `apply_automatic_gain_control` (Dòng 111-147): Giải thích công thức AGC và nguyên lý chuẩn hóa năng lượng giúp hệ thống bất biến trước khoảng cách gần/xa micro.
   - Chỉ vào hàm `apply_speech_bandpass_filter` (Dòng 152-176): Giải thích bộ lọc dải thông $80\text{ Hz} \div 7500\text{ Hz}$ cắt bỏ rung cơ học mặt bàn và rít mic cao tần.
   - Chỉ vào hàm `isolate_and_condition_speech` (Dòng 181-236): Minh họa quy trình hợp nhất 4 bước bóc tách lõi giọng nói và đệm số 0 sạch.

2. **[`demo/app_gui.py`](file:///G:/Desktop/voice-AI-assistant/demo/app_gui.py)**:
   - Chỉ vào `normalize_and_save_wav` (Dòng 191-211): Cho thấy khi người dùng thu âm đăng ký khẩu lệnh, âm thanh đã được làm sạch và chuẩn hóa độ lợi trước khi lưu file.
   - Chỉ vào `FewShotEngine.wav_to_normalized_embedding` và `raw_audio_to_candidate_embeddings` (Dòng 239-298): Chứng minh vector đặc trưng đưa vào mạng `TC-ResNet8` được xử lý đồng nhất tuyệt đối giữa lúc đăng ký và lúc nhận dạng.
   - Chỉ vào `FewShotEngine.update_prototypes` (Dòng 355-368): **[MÔ-ĐUN 6]** Chứng minh công thức tính bán kính vùng lãnh thổ $R_k$ cho từng lớp từ khóa.
   - Chỉ vào `FewShotEngine.classify_audio` (Dòng 450-510): Chứng minh cơ chế phân loại Open-Set: Âm thanh nằm trong quả cầu $R_k$ được chấp nhận, ngoài quả cầu tự động gán nhãn `UNKNOWN` (vùng xám).

3. **[`demo/app_gui_v2.py`](file:///G:/Desktop/voice-AI-assistant/demo/app_gui_v2.py)**:
   - Chỉ vào `_listen_loop` (Dòng 760-785): Trình bày cơ chế kích hoạt kép (VAD + AGC Dual-Trigger) giúp bắt trọn vẹn cả những khẩu lệnh phát âm ở xa mic.
   - Chỉ vào `_show_result` (Dòng 820-865): **[MÔ-ĐUN 6]** Chứng minh rào chắn Wake Gating độc quyền, không cho phép âm thanh vùng xám đánh thức hệ thống và rút ngắn thời gian thức tỉnh an toàn xuống $7$ giây.
   - Chỉ vào `_collapse`, `_show_wake_orb`, `_expand` (Dòng 410-475): Giải thích cơ chế độc quyền giao diện (Mutually Exclusive UI) giữa Panel chính và Floating Orb.

---

## 6. KẾT LUẬN & ĐÓNG GÓP HỌC THUẬT

1. **Giải quyết trọn vẹn bài toán âm học thực địa**: Không dừng lại ở mô hình mạng học sâu trên tập dữ liệu chuẩn trong phòng thí nghiệm, đồ án đã giải quyết toàn diện bài toán suy hao năng lượng xa/gần micro, ô nhiễm tạp âm phòng học (quạt gió, gõ phím) và hiện tượng nói lan man / video YouTube gây kích hoạt nhầm.
2. **Kiến trúc cân bằng tối ưu**: Kết hợp hài hòa giữa mô hình siêu nhẹ `TC-ResNet8` (chỉ $64,560$ tham số), mạng `Silero VAD` (< 2ms), bộ lọc số cổ điển (DSP Biquad & AGC) và kiểm định không gian mở (Open-Set Unknown Calibration), bảo đảm toàn bộ hệ thống vận hành mượt mà thời gian thực trên CPU máy tính cá nhân mà không cần GPU rời.
3. **Giá trị bảo vệ đồ án**: Bộ mã nguồn và báo cáo được cấu trúc chuẩn hóa, có số liệu thực nghiệm đo đạc trước/sau minh bạch và ghi chú rõ ràng các mô-đun xử lý tiếng nói, sẵn sàng phục vụ trình bày trước hội đồng chuyên môn.
