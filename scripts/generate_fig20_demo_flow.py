import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT = Path(r"G:\Desktop\docs\report_assets")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['figure.dpi'] = 300

fig, ax = plt.subplots(figsize=(14, 7.0))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.5)
ax.axis('off')

# Color palette
NAVY = "#1a365d"
BLUE = "#2b6cb0"
LIGHT_BLUE = "#ebf8ff"
BORDER_BLUE = "#3182ce"

TEAL = "#234e52"
CYAN = "#319795"
LIGHT_TEAL = "#e6fffa"
BORDER_TEAL = "#38b2ac"

PURPLE = "#44337a"
LIGHT_PURPLE = "#faf5ff"
BORDER_PURPLE = "#805ad5"

ORANGE = "#c05621"
LIGHT_ORANGE = "#fffaf0"
BORDER_ORANGE = "#dd6b20"

GRAY_BG = "#f7fafc"
BORDER_GRAY = "#cbd5e0"

# Main title
ax.text(7.0, 7.2, "KIẾN TRÚC VẬN HÀNH DEMO VOICE AI ASSISTANT (FEW-SHOT KWS)",
        ha='center', va='center', fontsize=12, fontweight='bold', color=NAVY)

# 3 Main Pillars (Columns)
# Column 1: Enrollment Phase (Đăng ký từ khóa ít mẫu)
box1 = patches.FancyBboxPatch((0.5, 0.6), 3.8, 6.1, boxstyle="round,pad=0.08,rounding_size=0.1",
                             facecolor=LIGHT_BLUE, edgecolor=BORDER_BLUE, linewidth=1.8)
ax.add_patch(box1)
ax.text(2.4, 6.4, "GIAI ĐOẠN 1: ENROLLMENT", ha='center', va='center', fontweight='bold', fontsize=10.5, color=BLUE)
ax.text(2.4, 6.05, "(Đăng ký khẩu lệnh tùy biến 1-5 mẫu)", ha='center', va='center', fontstyle='italic', fontsize=8.5, color='#4a5568')

steps_c1 = [
    ("1. Thu âm mẫu từ khóa", "Người dùng nói 3-5 lần\n('open_notepad', 'system_mute')", 5.1),
    ("2. Tiền xử lý & MFCC", "Sampling 16 kHz, Frame 40ms\nTrích xuất Tensor [1, 51, 40]", 3.9),
    ("3. Frozen TC-ResNet8", "Chuyển tiếp qua mạng nơ-ron\nSinh vector embedding z ∈ R^48", 2.7),
    ("4. Tính tâm cụm Prototype", "c_k = (1/K) Σ z_{k,i}\nLưu vào Prototype Memory Cache", 1.5)
]

for title, desc, y_pos in steps_c1:
    subbox = patches.FancyBboxPatch((0.8, y_pos - 0.45), 3.2, 0.9, boxstyle="round,pad=0.04,rounding_size=0.06",
                                    facecolor='white', edgecolor='#90cdf4', linewidth=1.2)
    ax.add_patch(subbox)
    ax.text(2.4, y_pos + 0.15, title, ha='center', va='center', fontweight='bold', fontsize=9, color=NAVY)
    ax.text(2.4, y_pos - 0.18, desc, ha='center', va='center', fontsize=8, color='#2d3748', linespacing=1.2)
    if y_pos > 1.6:
        ax.annotate('', xy=(2.4, y_pos - 0.52), xytext=(2.4, y_pos - 0.45),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))

# Column 2: Live Inference Engine (Nhận diện trực tuyến)
box2 = patches.FancyBboxPatch((4.8, 0.6), 4.4, 6.1, boxstyle="round,pad=0.08,rounding_size=0.1",
                             facecolor=LIGHT_TEAL, edgecolor=BORDER_TEAL, linewidth=1.8)
ax.add_patch(box2)
ax.text(7.0, 6.4, "GIAI ĐOẠN 2: INFERENCE ENGINE", ha='center', va='center', fontweight='bold', fontsize=10.5, color=TEAL)
ax.text(7.0, 6.05, "(Nhận diện thời gian thực & Metric Learning)", ha='center', va='center', fontstyle='italic', fontsize=8.5, color='#4a5568')

steps_c2 = [
    ("1. Thu âm Query thời gian thực", "Microphone bắt đoạn âm thanh 1s\n(Audio Stream Buffer)", 5.1),
    ("2. Trích xuất Query Embedding", "MFCC -> Frozen TC-ResNet8\nVector truy vấn z_query ∈ R^48", 3.9),
    ("3. Đối chiếu khoảng cách Euclidean", "d_k = ||z_query - c_k||^2\nSo sánh với mọi Prototype trong Cache", 2.7),
    ("4. Phân loại & Kiểm định ngưỡng", "Softmax xác suất P(y=k) ≥ 70%\nNếu không đạt -> Gán nhãn UNKNOWN", 1.5)
]

for title, desc, y_pos in steps_c2:
    subbox = patches.FancyBboxPatch((5.1, y_pos - 0.45), 3.8, 0.9, boxstyle="round,pad=0.04,rounding_size=0.06",
                                    facecolor='white', edgecolor='#81e6d9', linewidth=1.2)
    ax.add_patch(subbox)
    ax.text(7.0, y_pos + 0.15, title, ha='center', va='center', fontweight='bold', fontsize=9, color=TEAL)
    ax.text(7.0, y_pos - 0.18, desc, ha='center', va='center', fontsize=8, color='#2d3748', linespacing=1.2)
    if y_pos > 1.6:
        ax.annotate('', xy=(7.0, y_pos - 0.52), xytext=(7.0, y_pos - 0.45),
                    arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.5))

# Column 3: Action Dispatcher (Điều khiển Desktop Assistant)
box3 = patches.FancyBboxPatch((9.7, 0.6), 3.8, 6.1, boxstyle="round,pad=0.08,rounding_size=0.1",
                             facecolor=LIGHT_ORANGE, edgecolor=BORDER_ORANGE, linewidth=1.8)
ax.add_patch(box3)
ax.text(11.6, 6.4, "GIAI ĐOẠN 3: ACTION DISPATCHER", ha='center', va='center', fontweight='bold', fontsize=10.5, color=ORANGE)
ax.text(11.6, 6.05, "(Ánh xạ hành động & Điều khiển Desktop OS)", ha='center', va='center', fontstyle='italic', fontsize=8.5, color='#4a5568')

actions = [
    ("Khẩu lệnh: 'open_notepad'", "Mở trình soạn thảo Notepad\n(Launch notepad.exe)", 5.1),
    ("Khẩu lệnh: 'open_browser'", "Mở trình duyệt mặc định\n(Launch default web browser)", 3.9),
    ("Khẩu lệnh: 'system_mute'", "Tắt / Bật âm lượng hệ thống\n(Toggle Windows Master Volume)", 2.7),
    ("Khẩu lệnh: 'stop_task'", "Dừng tác vụ / Đóng cửa sổ\n(Hủy bỏ tiến trình đang chạy)", 1.5)
]

for title, desc, y_pos in actions:
    subbox = patches.FancyBboxPatch((10.0, y_pos - 0.45), 3.2, 0.9, boxstyle="round,pad=0.04,rounding_size=0.06",
                                    facecolor='white', edgecolor='#fbd38d', linewidth=1.2)
    ax.add_patch(subbox)
    ax.text(11.6, y_pos + 0.15, title, ha='center', va='center', fontweight='bold', fontsize=9, color=ORANGE)
    ax.text(11.6, y_pos - 0.18, desc, ha='center', va='center', fontsize=8, color='#2d3748', linespacing=1.2)

# Inter-column Connecting Arrows
ax.annotate('Nạp Prototype\nvào bộ nhớ', xy=(4.8, 1.5), xytext=(4.3, 1.5),
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.0),
            ha='right', va='center', fontsize=8, fontweight='bold', color=NAVY)

ax.annotate('Kích hoạt\nhành động', xy=(9.7, 3.5), xytext=(9.2, 3.5),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.0),
            ha='right', va='center', fontsize=8, fontweight='bold', color=ORANGE)

plt.tight_layout()
out_file = OUT / "fig_20_demo_assistant_flow.png"
plt.savefig(out_file, dpi=300)
plt.close()
print(f"[+] Saved {out_file}")
