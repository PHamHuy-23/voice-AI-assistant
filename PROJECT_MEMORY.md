# Project Memory: Few-Shot Keyword Spotting & Voice AI Assistant

## Core Mission
- Tái hiện và phát triển hệ thống Few-Shot Keyword Spotting (FS-KWS) dựa trên kiến trúc **TC-ResNet8 + Prototypical Networks**.
- Ứng dụng mô hình vào Voice AI Assistant chạy ngầm trên Desktop để kích hoạt và điều khiển tác vụ bằng giọng nói với số lượng mẫu thu âm cực ít (1-5 samples).

## Current Status & Critical Findings (2026-10-03)
- **Chương 1 (Tổng quan & Cơ sở lý thuyết)**: Hoàn thành 100% (Mục 1.1 -> 1.5, 9 hình vector chuẩn xuất bản).
- **Chương 2 (Dữ liệu & Đặc trưng tiếng nói)**: Hoàn thành 100% (Mục 2.1 -> 2.5 với 100% dữ liệu audit thật từ Kaggle cho GSC v0.02, AudioMNIST, FSC; 18 hình ảnh thực nghiệm, trích xuất MFCC [1, 51, 40]).
- **Chương 3 (Giải pháp đề xuất, Cài đặt, Thực nghiệm & Demo giữa kỳ)**: **HOÀN THÀNH 100% TOÀN DIỆN (3.1 -> 3.5)**:
  - **Mục 3.1 (Kiến trúc & Quy trình đề xuất)**: Hoàn thành 100% (20 tiểu mục, chuẩn hóa danh xưng TC-ResNet, hình vector 3.1).
  - **Mục 3.2 (Cài đặt & Kiểm tra hoạt động ban đầu)**: Hoàn thành 100% (11 tiểu mục, vá lỗi `torchaudio.load`, Smoke test 5-way 1-shot 2-query đạt Loss 1.9866, Acc 20.0%, hình vector 3.2).
  - **Mục 3.3 (Hiện thực hóa mã nguồn hệ thống trên Kaggle GPU)**: Hoàn thành 100% (6 tiểu mục trích xuất từ `dataset (1).ipynb`: Chuẩn hóa âm học, Làm sạch FS-GSC, Trích xuất MFCC [1, 51, 40], Mạng TC-ResNet8 Dilated 64.560 tham số, Phân loại metric ProtoNet, Bộ điều phối ExperimentQueueRunner).
  - **Mục 3.4 (Kết quả thực nghiệm & Đối sánh công bố gốc)**: Hoàn thành 100% với **160 / 160 cấu hình hoàn thành 100%** (tiêu tốn 30.3 giờ Kaggle GPU Tesla T4). 3 bảng ma trận (Clean 4 Way x 5 Shot, Full Ambient, Xếp hạng 8 điều kiện môi trường), phân tích 4 quy luật học thuật sâu sắc, Bảng 3.4 so sánh 12 tiêu chí với bài báo gốc Parnami & Lee (*arXiv:2007.14463*), hình vector 3.3.
  - **Mục 3.5 (Demo các chức năng cơ bản giữa kỳ)**: Hoàn thành 100% (Kiến trúc 3 tầng Enrollment -> Inference Engine -> Action Dispatcher; Độ trễ suy luận siêu thấp 1.67-3.03 ms; Đăng ký 4 từ khóa trong 36.64 ms; Cơ chế kiểm soát biên $d_{max}=1.10$ loại bỏ 100% OOV/nhiễu; Điều khiển Windows: Notepad, Browser, Mute, Stop; hình vector 3.4).

### 🏆 Các mô hình tiêu biểu (Checkpoints xác thực từ 160 cấu hình)
Dữ liệu lưu tại `data/paper_reproduction_109of160_20261002_1643/results/` và `docs/reproduction_160_experiments.csv`:
- **Đỉnh cao toàn cục (Clean)**:
  - `exp_025` (2-way 15-shot): Acc $95.40\% \pm 1.07\%$, Loss $0.1240$
  - `exp_033` (2-way 20-shot): Acc $95.23\% \pm 1.01\%$, Loss $0.1312$
  - `exp_073` (3-way 20-shot): Acc $93.04\% \pm 0.97\%$, Loss $0.1962$
  - `exp_113` (4-way 20-shot): Acc $91.02\% \pm 0.73\%$, Loss $0.2431$
  - `exp_145` (5-way 15-shot): Acc $88.97\% \pm 0.63\%$, Loss $0.3293$
- **Đỉnh cao môi trường Trợ lý ảo thực tế (Full Ambient: BG + Silence + Unknown)**:
  - `exp_040` (2-way 20-shot): Acc $90.32\% \pm 0.87\%$, Loss $0.2645$
  - `exp_080` (3-way 20-shot): Acc $89.73\% \pm 0.96\%$, Loss $0.3146$
  - `exp_112` (4-way 15-shot): Acc $85.78\% \pm 0.61\%$, Loss $0.4296$
  - `exp_160` (5-way 20-shot): Acc $85.61\% \pm 0.67\%$, Loss $0.4389$

### ⚠️ Ghi chú thực nghiệm Live Microphone Desktop App & Thách thức kỹ thuật
- **Tình trạng thực tế**: Trên tập benchmark kiểm thử chuẩn offline (FS-GSC / file WAV cố định), mô hình TC-ResNet8 Dilated đạt độ chính xác cao $95.40\% \pm 1.07\%$ (Exp 025). Tuy nhiên khi thử nghiệm trực tiếp qua Microphone trong ứng dụng Desktop thời gian thực (`demo/app_gui.py`):
  - **Độ ổn định nhận diện giọng thật vẫn chưa đạt yêu cầu (tỉ lệ lỗi còn cao)**.
  - **Nguyên nhân chính**:
    1. **Domain Mismatch**: Mô hình được huấn luyện trên 35 từ khóa tiếng Anh đơn âm tiết chuẩn phòng thu của Google Speech Commands, khi người dùng nói tiếng Việt (từ ghép, thanh điệu) hoặc ngữ điệu tự nhiên thì biểu diễn MFCC bị lệch khỏi phân phối học của mạng.
    2. **Đồng bộ hóa thời gian (Temporal Alignment)**: Khẩu lệnh người dùng nói qua mic có thể lệch onset/offset trong cửa sổ 1.0s (16.000 mẫu); việc cắt/pad cứng hoặc căn đỉnh năng lượng đơn giản chưa đủ để khớp chính xác với 51 time-frames của TC-ResNet8 Dilated.
    3. **Ngưỡng biên khoảng cách Prototypical**: Độ nhạy của khoảng cách Euclidean đối với âm lượng micro (gain), nhiễu phòng và khoảng cách miệng khiến khoảng cách nội lớp ($d_{within}$) dao động mạnh ($1.1 \div 5.0$), khó áp dụng ngưỡng tĩnh.
  - **Kết luận & Định hướng giải quyết giai đoạn Cuối kỳ**:
    - Bản Demo kịch bản chuẩn (`demo/run_desktop_demo.py`) dùng để báo cáo hội đồng giữa kỳ đảm bảo tính trung thực khoa học và khả năng tái lập.
    - Cần tích hợp Voice Activity Detection cao cấp (như **Silero VAD**) để bắt khung từ khóa chính xác.
    - Áp dụng **Dynamic Time Warping (DTW)** hoặc cơ chế Fine-tuning / Domain Adaptation trên dữ liệu âm vị tiếng Việt.

## Deliverables & Artifacts Hoàn Thành
- **Báo cáo giữa kỳ (Word .docx)**: File chuẩn academic hoàn chỉnh (~9.36 MB, bao gồm toàn bộ Chương 1, 2, 3 từ 3.1 đến 3.5) tại:
  - `G:\Desktop\docs\Bao_cao_hang_muc_cham_giua_ky_Xu_ly_tieng_noi.docx`
  - `G:\Desktop\voice-AI-assistant\docs\report\Bao_cao_hang_muc_cham_giua_ky_Xu_ly_tieng_noi.docx`
- **Dữ liệu thực nghiệm & Bảng số liệu**:
  - `docs/reproduction_160_experiments.csv`: Đầy đủ 160 cấu hình chi tiết.
  - `data/paper_reproduction_109of160_20261002_1643/`: Checkpoint và log của 109 thí nghiệm bổ sung.
  - `data/paper_reproduction_51of160_20261001_1952/`: Checkpoint và log của 51 thí nghiệm ban đầu.
- **Mã nguồn hệ thống & Demo**:
  - `scripts/update_report_section_33_34.py`: Script tự động chèn mục 3.3 và 3.4 vào báo cáo.
  - `scripts/find_best_model.py`: Script truy vấn top mô hình và kiểm định checkpoint.
  - `demo/run_desktop_demo.py`: Kịch bản demo kiểm thử tự động & tương tác điều khiển Windows (5 ca thử nghiệm).
  - `demo/app_gui.py`: Giao diện Desktop GUI VoiceLink AI.
  - `docs/nhat_ky_tien_hanh.md`: Cập nhật đầy đủ từ mục #1 đến mục #32.

## Next Steps (Giai đoạn Cuối kỳ)
1. Tiến hành thực nghiệm mở rộng: Cross-dataset trên AudioMNIST và Cross-domain trên Fluent Speech Commands.
2. Tích hợp mô-đun Silero VAD vào bộ thu âm Microphone để cắt ghép chính xác khung thời gian khẩu lệnh.
3. Nghiên cứu giải pháp Dynamic Time Warping (DTW) hoặc Domain Adaptation nhẹ trên dữ liệu tiếng Việt.
