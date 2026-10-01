# Project Memory: Few-Shot Keyword Spotting & Voice AI Assistant

## Core Mission
- Tái hiện và phát triển hệ thống Few-Shot Keyword Spotting (FS-KWS) dựa trên kiến trúc **TC-ResNet8 + Prototypical Networks**.
- Ứng dụng mô hình vào Voice AI Assistant chạy ngầm trên Desktop để kích hoạt và điều khiển tác vụ bằng giọng nói với số lượng mẫu thu âm cực ít (1-5 samples).

## Current Status (2026-10-01)
- **Chương 1 (Tổng quan & Cơ sở lý thuyết)**: Hoàn thành 100% (Mục 1.1 -> 1.5, 9 hình vector chuẩn xuất bản).
- **Chương 2 (Dữ liệu & Đặc trưng tiếng nói)**: Hoàn thành 100% (Mục 2.1 -> 2.5 với 100% dữ liệu audit thật từ Kaggle cho GSC v0.02, AudioMNIST, FSC; 18 hình ảnh thực nghiệm, trích xuất MFCC [1, 51, 40]).
- **Chương 3 (Giải pháp đề xuất & Cài đặt ban đầu)**:
  - **Mục 3.1 (Kiến trúc & Quy trình đề xuất)**: Hoàn thành 100% (20 tiểu mục, chuẩn hóa danh xưng TC-ResNet, hình vector 3.1).
  - **Mục 3.2 (Cài đặt & Kiểm tra hoạt động ban đầu)**: Hoàn thành 100% (11 tiểu mục, vá lỗi `torchaudio.load`, Smoke test 5-way 1-shot 2-query đạt Loss 1.9866, Acc 20.0%, hình vector 3.2).
  - **Mục 3.4 & 3.5**: Giữ nguyên tiêu đề rỗng theo chuẩn barem chấm giữa kỳ.

## Last Session Achievements
- Kiểm toán toàn diện 105.835 file WAV GSC raw và 33.790 file FS-GSC prepared.
- Bổ sung các bảng CSV audit và metadata kiểm toán vào Báo cáo giữa kỳ.
- Hoàn thiện trọn vẹn Mục 3.1 và Mục 3.2 trong file Word (`Bao_cao_hang_muc_cham_giua_ky_Xu_ly_tieng_noi.docx` ~8.61 MB).
- Cập nhật nhật ký tiến trình (`docs/nhat_ky_tien_hanh.md`) đến mục #28.

## Next Steps
- Chuẩn bị pipeline huấn luyện chính thức (Training loop & episodic evaluation).
- Tiến hành thực nghiệm Baseline (Exp 1) trên FS-GSC và các kịch bản mở rộng (Exp 2, 3, 4).
- Thu thập số liệu accuracy/loss thực nghiệm để điền vào Mục 3.4 và xây dựng kịch bản demo cho Mục 3.5.
