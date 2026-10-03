import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(r"G:\Desktop\docs\report_assets")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.dpi'] = 300

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Subplot 1: Scaling law of K-shot
shots = [1, 5, 10, 15, 20]
acc_clean = [85.40, 92.67, 94.83, 95.40, 95.23]
ci_clean = [2.75, 1.74, 1.13, 1.07, 1.01]

acc_ambient = [78.17, 87.67, 87.95, 88.97, 90.32]
ci_ambient = [2.22, 1.33, 1.26, 1.26, 0.87]

ax1.errorbar(shots, acc_clean, yerr=ci_clean, fmt='-o', color='#2b6cb0', linewidth=2.2,
             capsize=4, capthick=1.5, markersize=7, label='Clean (Không nhiễu/Silence/Unknown)')
ax1.errorbar(shots, acc_ambient, yerr=ci_ambient, fmt='-s', color='#c05621', linewidth=2.2,
             capsize=4, capthick=1.5, markersize=7, label='Full Ambient (BG + Silence + Unknown)')

# Annotations
ax1.annotate('Bão hòa tại 10-15 shot\n(Đạt đỉnh 95.40%)', xy=(15, 95.40), xytext=(11, 98),
             arrowprops=dict(arrowstyle='->', color='#1a365d', lw=1.2),
             fontsize=9, fontweight='bold', color='#1a365d', ha='center')

ax1.annotate('+7.27% nhảy vọt\n(1-shot -> 5-shot)', xy=(3, 89.0), xytext=(3.5, 84),
             arrowprops=dict(arrowstyle='->', color='#2b6cb0', lw=1.2),
             fontsize=9, fontweight='bold', color='#2b6cb0', ha='center')

ax1.set_title('(a) Quy luật bão hòa số lượng mẫu hỗ trợ (K-shot)', fontsize=11, fontweight='bold', pad=12)
ax1.set_xlabel('Số lượng mẫu hỗ trợ K (Shot)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Độ chính xác kiểm thử (%)', fontsize=10, fontweight='bold')
ax1.set_xticks(shots)
ax1.set_ylim(70, 102)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower right', frameon=True, facecolor='#f7fafc', edgecolor='#cbd5e0', fontsize=9)

# Subplot 2: Environmental conditions impact (1-shot vs 5-shot)
cond_labels = ['Clean', 'BG Noise', 'Silence only\n(Silence Trap)', 'Silence + Unk\n(Bù trừ OOV)', 'Full Ambient']
acc_1shot = [85.40, 87.20, 66.20, 80.10, 78.17]
ci_1shot = [2.75, 2.23, 2.92, 2.43, 2.22]

acc_5shot = [92.67, 91.97, 77.40, 89.55, 87.67]
ci_5shot = [1.74, 1.61, 2.88, 1.20, 1.33]

x = np.arange(len(cond_labels))
width = 0.35

rects1 = ax2.bar(x - width/2, acc_1shot, width, yerr=ci_1shot, capsize=3,
                 color='#4299e1', edgecolor='#2b6cb0', label='1-shot (K = 1)')
rects2 = ax2.bar(x + width/2, acc_5shot, width, yerr=ci_5shot, capsize=3,
                 color='#ed8936', edgecolor='#c05621', label='5-shot (K = 5)')

# Highlight Silence trap
ax2.axvspan(1.6, 2.4, color='#fed7d7', alpha=0.3, zorder=0)
ax2.text(2, 60, 'Sụt giảm mạnh\n(-19.2%)', ha='center', fontsize=8.5, color='#9b2c2c', fontweight='bold')

# Value labels on bars
for r in rects1:
    h = r.get_height()
    ax2.text(r.get_x() + r.get_width()/2., h + 3.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=8, color='#1a365d')
for r in rects2:
    h = r.get_height()
    ax2.text(r.get_x() + r.get_width()/2., h + 3.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=8, color='#7b341e')

ax2.set_title('(b) Tác động của điều kiện môi trường âm học (1-shot vs 5-shot)', fontsize=11, fontweight='bold', pad=12)
ax2.set_xlabel('Điều kiện môi trường kiểm thử', fontsize=10, fontweight='bold')
ax2.set_ylabel('Độ chính xác kiểm thử (%)', fontsize=10, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(cond_labels, fontsize=8.5)
ax2.set_ylim(50, 105)
ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
ax2.legend(loc='lower right', frameon=True, facecolor='#f7fafc', edgecolor='#cbd5e0', fontsize=9)

plt.tight_layout()
out_file = OUT / "fig_19_interim_experimental_results.png"
plt.savefig(out_file, dpi=300)
plt.close()
print(f"[+] Saved {out_file}")
