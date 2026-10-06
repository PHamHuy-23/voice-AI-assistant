# Project Memory: Few-Shot Keyword Spotting & Voice AI Assistant

## Core Mission
- Tái hiện và phát triển hệ thống Few-Shot Keyword Spotting (FS-KWS) dựa trên kiến trúc **TC-ResNet8 + Prototypical Networks**.
- Ứng dụng mô hình vào Voice AI Assistant chạy ngầm trên Desktop để kích hoạt và điều khiển tác vụ bằng giọng nói với số lượng mẫu thu âm cực ít (1-5 samples).

## Current Status & Critical Findings (2026-10-06)
- **Chương 1 (Tổng quan & Cơ sở lý thuyết)**: Hoàn thành 100% (Mục 1.1 -> 1.5, 9 hình vector chuẩn xuất bản).
- **Chương 2 (Dữ liệu & Đặc trưng tiếng nói)**: Hoàn thành 100% (Mục 2.1 -> 2.5 với 100% dữ liệu audit thật từ Kaggle cho GSC v0.02, AudioMNIST, FSC; 18 hình ảnh thực nghiệm, trích xuất MFCC [1, 51, 40]).
- **Chương 3 (Giải pháp đề xuất, Cài đặt, Thực nghiệm & Demo giữa kỳ)**: **HOÀN THÀNH 100% TOÀN DIỆN (3.1 -> 3.5)**:
  - **Mục 3.1 (Kiến trúc & Quy trình đề xuất)**: Hoàn thành 100% (20 tiểu mục, chuẩn hóa danh xưng TC-ResNet8 Dilated 64.560 tham số, hình vector 3.1).
  - **Mục 3.2 (Cài đặt & Kiểm tra hoạt động ban đầu)**: Hoàn thành 100% (11 tiểu mục, vá lỗi `torchaudio.load`, Smoke test 5-way 1-shot 2-query đạt Loss 1.9866, Acc 20.0%, hình vector 3.2).
  - **Mục 3.3 (Hiện thực hóa mã nguồn hệ thống trên Kaggle GPU)**: Hoàn thành 100% (6 tiểu mục trích xuất từ `dataset (1).ipynb`: Chuẩn hóa âm học, Làm sạch FS-GSC, Trích xuất MFCC [1, 51, 40], Mạng TC-ResNet8 Dilated, Phân loại metric ProtoNet, Bộ điều phối ExperimentQueueRunner).
  - **Mục 3.4 (Kết quả thực nghiệm & Đối sánh công bố gốc)**: Hoàn thành 100% với **160 / 160 cấu hình hoàn thành 100%** (tiêu tốn 30.3 giờ Kaggle GPU Tesla T4). 3 bảng ma trận (Clean 4 Way x 5 Shot, Full Ambient, Xếp hạng 8 điều kiện môi trường), phân tích 4 quy luật học thuật sâu sắc, Bảng 3.4 so sánh 12 tiêu chí với bài báo gốc Parnami & Lee (*arXiv:2007.14463*), hình vector 3.3.
  - **Mục 3.5 (Demo các chức năng cơ bản giữa kỳ)**: Hoàn thành 100% (Kiến trúc 3 tầng Enrollment -> Inference Engine -> Action Dispatcher; Độ trễ suy luận siêu thấp 1.67-3.03 ms; Đăng ký 4 từ khóa trong 36.64 ms; Cơ chế kiểm soát biên $d_{max}=1.10$ loại bỏ OOV/nhiễu; Điều khiển Windows: Notepad, Browser, Mute, Stop; hình vector 3.4).
- **Chuẩn hóa Báo cáo Word học thuật**:
  - Tự động chuyển đổi công thức toán từ chuỗi LaTeX sang định dạng Microsoft Word OMML chuẩn (`scripts/fix_report_equations.py`).
  - Chuẩn hóa tên mô hình thành TC-ResNet8 Dilated và đồng bộ đoạn mã kiến trúc mạng (`scripts/update_report_model_naming.py`). File báo cáo tối ưu nhất: `docs/reports/midterm/Bao_cao_hang_muc_cham_giua_ky_Xu_ly_tieng_noi_cong_thuc_va_ten_mo_hinh_da_chinh.docx` (~9.94 MB).
- **Nâng cấp Hệ thống Desktop GUI v2 & Live Assistant**:
  - `demo/app_gui_v2.py`: Hỗ trợ giao diện thu âm bổ sung mẫu (`RecordMoreSheet`), tự động cân chỉnh độ ồn phòng (`_auto_calibrate_noise` / `ambient_noise_rms`) và cơ chế `NOISE_PROTOTYPE_MATCH` chống nhận diện nhầm tiếng ồn.
  - Tích hợp Antigravity CLI (`demo/antigravity_runner.py`): Stream JSON, cờ `--dangerously-skip-permissions`, cập nhật thời gian thực qua `step_update` delta và `result.response`.
  - Khởi động ngầm qua `CHAY_VOICE_AI.bat` với cờ `--daemon`.

### 🏆 Các mô hình tiêu biểu (Checkpoints xác thực từ 160 cấu hình)
Dữ liệu lưu tại `experiments/runs/paper_reproduction_109of160_20261002_1643/results/` và `reports/tables/reproduction_160_experiments.csv`:
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

### 🛠️ Tiến bộ giải pháp Live Microphone Desktop App & Noise Suppression (Cập nhật 2026-10-07)
- **Kiến trúc đường ống âm học hoàn chỉnh (`src/audio_dsp.py`)**:
  1. **Silero VAD Endpointing**: Bóc tách lõi giọng nói (< 2ms độ trễ), loại bỏ 100% tiếng ồn quạt/bàn phím.
  2. **Automatic Gain Control (AGC)**: Bù động biên độ với mục tiêu RMS = 0.12, triệt tiêu suy hao năng lượng giữa nói gần mic (RMS ~0.30) và nói xa mic (RMS ~0.02).
  3. **Speech Bandpass Filter (80Hz - 7500Hz)**: Triệt rung cơ học và rít mạch điện tử.
  4. **Zero-Padding & Centering**: Căn giữa giọng nói trong khung 1.0s và đệm số 0 sạch thay vì đệm tạp âm phòng.
- **Nâng cấp Hệ thống Nhận diện Khẩu lệnh & Phản xạ (Few-Shot Engine)**:
  1. **Open-Set Bounded Prototypes**: Xác định bán kính biên $R_k$ cho từng lớp từ khóa; mọi âm thanh ngoài bán kính được gán nhãn `UNKNOWN` an toàn.
  2. **Wake Gating an toàn (Wake window 10s)**: Duy trì trạng thái Awake 10 giây; nếu phát âm lệch nhẹ trong lúc thức, hệ thống nhắc nhở và giữ nguyên trạng thái thức để người dùng nói lại mà không bắt gọi lại wake word.
- **Nâng cấp Giao diện, Quản lý Vòng đời & Tương tác UI/UX (`demo/app_gui_v2.py`, `demo/orb_overlay.py`)**:
  1. **Độc quyền giao diện (Strict Mutual Exclusivity)**:
     - Khi mở Panel chính: Tuyệt đối không sinh Orb overlay.
     - Khi bấm "Minimize": Panel ẩn đi (`withdraw`), chỉ hiển thị Floating Orb khi thức giấc. Bấm vào Orb mở lại Panel; bấm `[x]` trên Orb đóng Orb.
  2. **Quản lý thoát ứng dụng sạch sẽ**: Gán `WM_DELETE_WINDOW` trực tiếp vào `_quit()`. Bấm `[X]` trên thanh tiêu đề Windows hoặc nút "Quit" sidebar sẽ hủy triệt để luồng mic, tiến trình Orb con và gọi `os._exit(0)` sạch sẽ, loại bỏ hoàn toàn tiến trình ma (zombie processes).
  3. **Bộ âm thanh Sci-Fi (`demo/assets/sounds/`)**:
     - `gaming_lock.wav`: Âm thanh khóa mục tiêu thức tỉnh (Wake chime).
     - `click.wav`: Nhận diện lệnh hợp lệ.
     - `confirm.wav`: Hoàn tất tác vụ Antigravity và lưu sửa mô tả lệnh.
  4. **Quản lý sửa mô tả lệnh & Tự động học lại (Auto Re-learning Trigger)**: Thêm nút "Sửa" (`EditCommandSheet`). Khi người dùng đổi mô tả, tự động hủy `action_profile` đã cache để Antigravity AI tự động học lại hành động mới ở lần gọi tiếp theo.
  5. **Giao diện tự động học lệnh (Learning State)**: Hiển thị màu tím (`PURPLE`) `🧠 Đang học: <Tên lệnh>` với trạng thái chờ minh bạch, khóa mic tránh nhận diện rác chen ngang.

## Deliverables & Artifacts Hoàn Thành
- **Tài liệu & Báo cáo nâng cấp**:
  - `UPDATE.md` & `docs/reports/UPDATE_HE_THONG_KWS_REALTIME.md`: Báo cáo chi tiết 11 mục cập nhật kỹ thuật âm học, thực nghiệm, giao diện và kiến trúc KWS thời gian thực.
  - Báo cáo giữa kỳ Word OMML: `docs/reports/midterm/Bao_cao_hang_muc_cham_giua_ky_Xu_ly_tieng_noi_cong_thuc_va_ten_mo_hinh_da_chinh.docx`.
- **Mã nguồn hệ thống & Demo**:
  - `src/audio_dsp.py`: Pipeline xử lý âm thanh 4 mô-đun (Silero VAD, AGC, Bandpass, Zero-Padding).
  - `demo/app_gui_v2.py`: Ứng dụng Desktop Voice Assistant toàn diện.
  - `demo/orb_overlay.py`: Cửa sổ nổi Compact Orb công nghệ cao.
  - `demo/assets/sounds/`: Bộ hiệu ứng âm thanh Sci-Fi (`gaming_lock.wav`, `click.wav`, `confirm.wav`).
  - `demo/antigravity_runner.py`: Bộ kết nối hai chiều với Antigravity CLI.
  - `CHAY_VOICE_AI.bat`: Trình khởi chạy hệ thống.

## Next Steps (Giai đoạn Cuối kỳ)
1. Mở rộng thực nghiệm Cross-dataset trên AudioMNIST và Cross-domain trên Fluent Speech Commands.
2. Thử nghiệm tích hợp Dynamic Time Warping (DTW) hoặc Domain Adaptation nhẹ trên dữ liệu âm vị tiếng Việt.
3. Hoàn thiện báo cáo toàn diện cuối kỳ dựa trên các mục kỹ thuật đã ghi lại trong `UPDATE.md`.
