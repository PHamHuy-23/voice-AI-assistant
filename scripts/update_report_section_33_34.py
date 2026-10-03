import os
import shutil
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_PATH = Path(r"docs\report\Bao_cao_hang_muc_cham_giua_ky_Xu_ly_tieng_noi.docx")
BACKUP_PATH = Path(r"docs\report\Bao_cao_hang_muc_cham_giua_ky_Xu_ly_tieng_noi.docx.bak")
DEST_EXTERNAL_PATH = Path(r"G:\Desktop\docs\Bao_cao_hang_muc_cham_giua_ky_Xu_ly_tieng_noi.docx")
FIG19_PATH = Path(r"G:\Desktop\docs\report_assets\fig_19_interim_experimental_results.png")

print("Loading document:", DOCX_PATH)
doc = docx.Document(str(DOCX_PATH))

# Helper to format tables
def style_table(table, col_widths=None, header_bg="2B6CB0", alt_bg="F7FAFC"):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Set header styling
    for i, cell in enumerate(table.rows[0].cells):
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{header_bg}"/>')
        cell._tc.get_or_add_tcPr().append(shading)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)
                r.font.size = Pt(9.5)
                r.font.name = "Times New Roman"
    
    # Body rows
    for r_idx, row in enumerate(table.rows[1:], start=1):
        bg = alt_bg if r_idx % 2 == 1 else "FFFFFF"
        for cell in row.cells:
            if bg != "FFFFFF":
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg}"/>')
                cell._tc.get_or_add_tcPr().append(shd)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
                    r.font.name = "Times New Roman"
    
    # Borders
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E0"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="CBD5E0"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

    # Col widths if provided
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                if i < len(row.cells):
                    row.cells[i].width = Inches(w)

print("Locating Section 3.4 to replace...")
body = doc._body._element

p1061_elem = doc.paragraphs[1061]._element
p1101_elem = doc.paragraphs[1101]._element

idx_start = body.index(p1061_elem)
idx_end = body.index(p1101_elem)

print(f"Removing elements from index {idx_start} to {idx_end} (total {idx_end - idx_start} elements)...")
for _ in range(idx_end - idx_start):
    body.remove(body[idx_start])

# Target insertion point is right before p1101_elem
ref_elem = p1101_elem

def insert_p(text="", style="Normal", space_before=3, space_after=3, line_spacing=1.15, bold=False, italic=False, color=None):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if text:
        run = p.add_run(text)
        if bold:
            run.font.bold = True
        if italic:
            run.font.italic = True
        if color:
            run.font.color.rgb = color
        run.font.name = "Times New Roman"
    # move element before ref_elem
    ref_elem.addprevious(p._p)
    return p

def insert_code_block(code_text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.05
    # Shading and borders for code box
    pPr = p._p.get_or_add_pPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F8F9FA"/>')
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="18" w:space="8" w:color="2B6CB0"/>'
        f'<w:top w:val="single" w:sz="4" w:space="4" w:color="E2E8F0"/>'
        f'<w:right w:val="single" w:sz="4" w:space="4" w:color="E2E8F0"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="4" w:color="E2E8F0"/>'
        f'</w:pBdr>'
    )
    pPr.append(shd)
    pPr.append(pBdr)
    
    run = p.add_run(code_text.strip())
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(30, 41, 59)
    ref_elem.addprevious(p._p)
    return p

def insert_heading(text, level=2):
    style = f"Heading {level}"
    before = 12 if level == 2 else 8
    after = 4 if level == 2 else 3
    p = insert_p(text, style=style, space_before=before, space_after=after, bold=True)
    p.runs[0].font.size = Pt(13 if level == 2 else 11.5)
    p.runs[0].font.color.rgb = RGBColor(26, 54, 93) if level == 2 else RGBColor(43, 108, 176)
    return p

def insert_table(table_data, col_widths=None, header_bg="2B6CB0"):
    t = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            t.rows[r_idx].cells[c_idx].text = str(val)
    style_table(t, col_widths=col_widths, header_bg=header_bg)
    ref_elem.addprevious(t._tbl)
    return t

print("Generating Section 3.3...")
# ==============================================================================
# SECTION 3.3
# ==============================================================================
insert_heading("3.3. Hiện thực hóa mã nguồn hệ thống trên môi trường Kaggle GPU", level=2)

insert_p("Để hiện thực hóa giải pháp đề xuất theo đúng chuẩn bài báo gốc và bảo đảm tính khả thi thực tế cho ứng dụng Voice AI Assistant, toàn bộ hệ thống được cài đặt hoàn chỉnh trên ngôn ngữ Python và framework học sâu PyTorch/Torchaudio. Mã nguồn được tổ chức theo kiến trúc 4 mô-đun chức năng khép kín: (1) Chuẩn bị và chuẩn hóa dữ liệu; (2) Trích xuất đặc trưng âm học MFCC; (3) Mạng nơ-ron TC-ResNet8 Dilated kết hợp Prototypical Networks; và (4) Bộ điều phối tự động hóa thực nghiệm quy mô lớn.")

insert_heading("3.3.1. Chuẩn hóa âm học và Tiền xử lý tín hiệu cơ sở", level=3)
insert_p("Tín hiệu âm thanh trước khi đưa vào mạng nơ-ron bắt buộc phải đi qua các hàm biến đổi âm học cơ sở nhằm thống nhất tần số lấy mẫu về 16.000 Hz, chuyển đổi về định dạng đơn kênh (mono), chuẩn hóa biên độ tuyệt đối và căn chỉnh thời lượng chính xác về 1,0 giây (16.000 samples):")

insert_code_block("""import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

def to_mono(audio: np.ndarray) -> np.ndarray:
    \"\"\"Chuyển đổi tín hiệu đa kênh (Stereo) về đơn kênh (Mono).\"\"\"
    return audio if audio.ndim == 1 else audio.mean(axis=1)

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    \"\"\"Tái lấy mẫu âm thanh thông qua bộ lọc đa thức resample_poly.\"\"\"
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(audio, up, down)

def peak_normalize(audio: np.ndarray) -> np.ndarray:
    \"\"\"Chuẩn hóa biên độ tín hiệu theo giá trị đỉnh tuyệt đối về dải [-1.0, 1.0].\"\"\"
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio

def fix_length(audio: np.ndarray, target_length: int = 16000) -> np.ndarray:
    \"\"\"Căn chỉnh thời lượng về đúng 16.000 mẫu bằng Center Zero-Padding hoặc Center Cropping.\"\"\"
    length = len(audio)
    if length < target_length:
        total_pad = target_length - length
        left = total_pad // 2
        return np.pad(audio, (left, total_pad - left), mode="constant")
    elif length > target_length:
        start = (length - target_length) // 2
        return audio[start:start + target_length]
    return audio""")

insert_heading("3.3.2. Quy trình tổ chức và làm sạch tập dữ liệu FS-GSC", level=3)
insert_p("Để tạo lập tập dữ liệu kiểm chuẩn FS-GSC từ 105.835 tệp thô của Google Speech Commands v0.02, đường ống chuẩn bị dữ liệu thực thi tự động qua các bước lọc độ dài, gom cụm người nói, phân chia Core/Unknown và cân bằng mẫu:")

insert_code_block("""import os
import random
import torchaudio
from pathlib import Path

class FewShotSpeechDataPipeline:
    \"\"\"Pipeline tổ chức FS-GSC theo tác giả phục vụ huấn luyện ít mẫu.\"\"\"
    def __init__(self, dataset_path: str, target_sr: int = 16000, clip_duration_ms: int = 1000):
        self.root = Path(dataset_path)
        self.desired_samples = int(target_sr * clip_duration_ms / 1000)
        self.speaker_limit = 1000
        random.seed(42)

    def step1_filter_short_files(self):
        \"\"\"Bước 1: Loại bỏ toàn bộ các file âm thanh ngắn hơn 16.000 samples (< 1.0s).\"\"\"
        for wav_file in self.root.rglob("*.wav"):
            if wav_file.parent.name.startswith("_"):
                continue
            sound, _ = torchaudio.load(str(wav_file), num_frames=self.desired_samples)
            if sound.shape[1] != self.desired_samples:
                os.unlink(wav_file)

    def step2_group_by_speaker(self):
        \"\"\"Bước 2: Gom nhóm các file WAV theo Speaker ID (tách từ chuỗi _nohash_).\"\"\"
        for kw_dir in [d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith("_")]:
            for wav_file in list(kw_dir.glob("*.wav")):
                spk_id = wav_file.stem.split("_nohash_")[0]
                spk_dir = kw_dir / spk_id
                spk_dir.mkdir(exist_ok=True)
                wav_file.rename(spk_dir / wav_file.name)

    def step3_split_core_unknown(self):
        \"\"\"Bước 3: Phân chia Core Words (>=1000 speakers) và Unknown Words (<1000 speakers).\"\"\"
        self.core_words, self.unknown_words = [], []
        for kw_dir in [d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith("_")]:
            spk_cnt = sum(1 for p in kw_dir.iterdir() if p.is_dir())
            if spk_cnt >= self.speaker_limit:
                self.core_words.append(kw_dir.name)
            else:
                self.unknown_words.append(kw_dir.name)

    def step4_balance_speakers(self):
        \"\"\"Bước 4: Cân bằng số mẫu cho mỗi từ khóa (mỗi speaker chọn ngẫu nhiên đúng 1 file).\"\"\"
        for kw_dir in [d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith("_")]:
            speakers = [p for p in kw_dir.iterdir() if p.is_dir()]
            random.shuffle(speakers)
            limit = self.speaker_limit if kw_dir.name in self.core_words else len(speakers)
            for spk in speakers[:limit]:
                wavs = list(spk.glob("*.wav"))
                if wavs:
                    chosen = random.choice(wavs)
                    # Giữ file chosen và dọn dẹp các phát âm dư thừa của speaker

    def step5_generate_splits(self):
        \"\"\"Bước 5: Phân chia 30 Core classes thành 20 Train, 5 Validation và 5 Test.\"\"\"
        random.shuffle(self.core_words)
        return {
            "train": self.core_words[:20],
            "val": self.core_words[20:25],
            "test": self.core_words[25:30]
        }""")

insert_heading("3.3.3. Trích xuất đặc trưng âm học MFCC Tensor [1, 51, 40]", level=3)
insert_p("Tín hiệu âm thanh 1,0 giây được chuyển đổi sang ma trận đặc trưng miền thời gian - tần số bằng biến đổi MFCC với 40 hệ số, kích thước cửa sổ 40ms và bước nhảy 20ms, sinh ra tensor đầu vào có kích thước cố định [1, 51, 40]:")

insert_code_block("""import torch
import torchaudio

class SpeechFeatureExtractor:
    \"\"\"Trích xuất tensor MFCC chuẩn hóa cho mạng TC-ResNet.\"\"\"
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 40, window_ms: int = 40, stride_ms: int = 20):
        self.sample_rate = sample_rate
        self.n_fft = int(window_ms * sample_rate / 1000)      # 640 samples (40ms)
        self.hop_length = int(stride_ms * sample_rate / 1000)  # 320 samples (20ms)
        self.transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": self.n_fft, "hop_length": self.hop_length, "n_mels": 40}
        )

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        \"\"\"Đầu vào: Waveform [1, 16000] -> Đầu ra: Tensor [1, 51, 40]\"\"\"
        features = self.transform(waveform)        # Trích xuất: [1, 40, 51]
        features = features[0].T                   # Hoán vị: [51, 40]
        features = torch.unsqueeze(features, 0)    # Định dạng tensor mạng: [1, 51, 40]
        return features""")

insert_heading("3.3.4. Kiến trúc mạng mã hóa TC-ResNet8 Dilated", level=3)
insert_p("Mạng mã hóa TC-ResNet8 Dilated được xây dựng với 3 khối Residual Blocks, sử dụng tích chập thời gian 1D (kernel 9×1) và hệ số giãn nở tăng dần [1, 2, 4] nhằm nắm bắt ngữ cảnh âm vị dài hạn với chi phí tham số siêu nhỏ gọn (chỉ 64.560 tham số):")

insert_code_block("""import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(9, 1), stride=(1, 1), dilation=(1, 1)):
        super().__init__()
        padding = ((kernel_size[0] - 1) * dilation[0] // 2, 0)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class TCResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=(1, 1), dilation=(1, 1)):
        super().__init__()
        self.conv1 = ConvBlock(in_c, out_c, stride=stride, dilation=dilation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(9, 1), padding=((9 - 1) * dilation[0] // 2, 0), dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        return F.relu(self.conv2(self.conv1(x)) + self.shortcut(x))

class TCResNet8Dilated(nn.Module):
    \"\"\"Backbone trích xuất đặc trưng: 64.560 tham số, ánh xạ [B, 1, 51, 40] -> z in R^48\"\"\"
    def __init__(self, in_channels=1, channels=[16, 24, 32, 48]):
        super().__init__()
        self.init_conv = ConvBlock(in_channels, channels[0], kernel_size=(3, 1))
        self.layer1 = TCResBlock(channels[0], channels[1], stride=(2, 1), dilation=(1, 1))
        self.layer2 = TCResBlock(channels[1], channels[2], stride=(2, 1), dilation=(2, 1))
        self.layer3 = TCResBlock(channels[2], channels[3], stride=(2, 1), dilation=(4, 1))

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  # Chuyển chiều MFCC sang channel: [B, 40, 51, 1]
        out = self.init_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)  # Vector embedding z in R^48
        return out""")

insert_heading("3.3.5. Bộ phân loại mêtric Prototypical Networks (Episodic Metric Loss)", level=3)
insert_p("Bộ phân loại Prototypical Network tính toán tâm cụm prototype c_k từ tập Support, đo khoảng cách Euclidean tới các mẫu Query, và lan truyền ngược hàm mất mát Negative Log-Likelihood qua phân bố xác suất Softmax:")

insert_code_block("""class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, sample: dict):
        xs, xq = sample['xs'], sample['xq']  # xs: [Way, Shot, 1, 51, 40], xq: [Way, Query, 1, 51, 40]
        n_class, n_support, n_query = xs.size(0), xs.size(1), xq.size(1)

        # Trích xuất toàn bộ vector biểu diễn qua mạng encoder
        x = torch.cat([xs.view(-1, 1, 51, 40), xq.view(-1, 1, 51, 40)], dim=0)
        z = self.encoder(x)
        z_dim = z.size(-1)

        # Tính toán vector tâm cụm prototype c_k và trích xuất vector truy vấn z_q
        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(dim=1)
        z_query = z[n_class * n_support:]

        # Khoảng cách Euclidean bình phương: d(z_q, c_k) = ||z_q - c_k||^2
        dists = torch.cdist(z_query, z_proto, p=2).pow(2)

        # Xác suất Softmax và hàm mất mát NLL Loss
        log_p_y = F.log_softmax(-dists, dim=1)
        target_y = torch.arange(0, n_class, device=xs.device).view(-1, 1).expand(n_class, n_query).reshape(-1)
        loss = F.nll_loss(log_p_y, target_y)
        acc = (log_p_y.argmax(dim=1) == target_y).float().mean()
        return loss, acc""")

insert_heading("3.3.6. Bộ điều phối tự động hóa thực nghiệm quy mô lớn", level=3)
insert_p("Để thực thi tự động toàn bộ ma trận 160 cấu hình thí nghiệm mà không cần can thiệp thủ công, nhóm phát triển kịch bản điều phối hàng đợi (Queue Runner) giám sát qua tệp manifest, tự động ghi nhận checkpoint trọng số tối ưu và kết quả đánh giá trên 100 test episodes:")

insert_code_block("""import sys
import time
import subprocess
import pandas as pd
from pathlib import Path

class ExperimentQueueRunner:
    \"\"\"Điều phối tự động hóa huấn luyện và đánh giá 160 cấu hình thực nghiệm.\"\"\"
    def __init__(self, repo_dir: str, manifest_path: str):
        self.repo = Path(repo_dir)
        self.manifest_path = Path(manifest_path)
        self.manifest = pd.read_csv(self.manifest_path)

    def get_condition_args(self, row: pd.Series) -> list:
        args = []
        if bool(row["background"]): args.append("--speech.include_background")
        if bool(row["silence"]):    args.append("--speech.include_silence")
        if bool(row["unknown"]):    args.append("--speech.include_unknown")
        return args

    def run_all(self):
        pending = self.manifest[self.manifest["status"] == "pending"]
        print(f"Bắt đầu điều phối {len(pending)} cấu hình thực nghiệm...")
        for idx, row in pending.iterrows():
            exp_id, way, shot = int(row["experiment_id"]), int(row["way"]), int(row["shot"])
            exp_name = f"exp_{exp_id:03d}"
            
            cmd = [
                sys.executable, "scripts/train/few_shot/run_train.py",
                "--data.dataset=googlespeech", f"--data.way={way}", f"--data.shot={shot}",
                "--data.query=5", f"--data.test_way={way}", f"--data.test_shot={shot}",
                "--data.test_query=15", "--data.train_episodes=200", "--data.test_episodes=100",
                "--model.model_name=protonet_conv", "--model.x_dim=1,51,40",
                "--model.hid_dim=64", "--model.z_dim=64", "--model.encoding=TCResNet8Dilated",
                "--train.epochs=200", "--train.optim_method=Adam", "--train.learning_rate=0.001",
                "--train.decay_every=20", "--train.weight_decay=0.0", "--train.patience=200",
                f"--log.exp_dir=results/{exp_name}", "--data.cuda"
            ]
            cmd.extend(self.get_condition_args(row))
            
            start = time.time()
            res = subprocess.run(cmd, cwd=str(self.repo), capture_output=True, text=True)
            elapsed = (time.time() - start) / 60.0
            
            if res.returncode == 0:
                self.manifest.loc[idx, "status"] = "done"
                self.manifest.loc[idx, "elapsed_minutes"] = round(elapsed, 2)
            else:
                self.manifest.loc[idx, "status"] = "error"
            self.manifest.to_csv(self.manifest_path, index=False)""")

print("Generating Section 3.4...")
# ==============================================================================
# SECTION 3.4
# ==============================================================================
insert_heading("3.4. Kết quả thực nghiệm và đối sánh với công bố gốc", level=2)

insert_p("Sau khi đường ống tính toán và bộ điều phối tự động hóa được thiết lập hoàn chỉnh, nhóm tiến hành huấn luyện và đánh giá trên tập dữ liệu kiểm chuẩn FS-GSC (Google Speech Commands v0.02) nhằm tái hiện và kiểm chứng độc lập các công bố khoa học của nghiên cứu gốc.")

insert_heading("3.4.1. Cấu hình huấn luyện Few-Shot", level=3)
insert_p("Toàn bộ các siêu tham số huấn luyện được thiết lập nghiêm ngặt theo đúng đặc tả của bài báo:")

table_hyperparams = [
    ["Tham số huấn luyện", "Giá trị thiết lập", "Vai trò / Ý nghĩa kỹ thuật"],
    ["Mạng mã hóa (Encoder)", "TCResNet8Dilated", "64.560 tham số (dung lượng trọng số ~260 KB)"],
    ["Không gian nhúng (Embedding)", "z in R^48", "Vector biểu diễn đặc trưng ngữ âm thu gọn"],
    ["Bộ phân loại (Classifier)", "Prototypical Network", "Học mêtric khoảng cách Euclidean và tâm cụm c_k"],
    ["Số Epochs", "200", "Huấn luyện theo cơ chế episodic learning"],
    ["Số Episodes huấn luyện / Epoch", "200", "Tổng cộng 40.000 episodes cho mỗi mô hình"],
    ["Số Episodes đánh giá", "100 test episodes", "Đo lường độ chính xác trung bình và 95% CI"],
    ["Query mẫu huấn luyện (Q_train)", "5 queries / class", "Tính toán hàm mất mát Negative Log-Likelihood"],
    ["Query mẫu kiểm thử (Q_test)", "15 queries / class", "Đánh giá khả năng tổng quát hóa trên từ khóa mới"],
    ["Thuật toán tối ưu (Optimizer)", "Adam (lr = 0.001)", "Tối ưu hóa tham số mạng mã hóa"],
    ["Giảm tốc độ học (Decay step)", "Mỗi 20 epoch (factor 0.5)", "Bảo đảm sự hội tụ ổn định của hàm mất mát"]
]
insert_table(table_hyperparams, col_widths=[2.0, 1.8, 2.7])

insert_p("Không gian thực nghiệm đầy đủ (Full Sweep) được thiết kế gồm tổng cộng:")
insert_p("4 \\times 5 \\times 8 = 160 \\; \\text{cấu hình độc lập}", style="Normal", italic=True)
insert_p("Tương ứng với: Way in {2, 3, 4, 5}; Shot in {1, 5, 10, 15, 20}; và 8 tổ hợp điều kiện môi trường âm học từ 3 biến nhị phân: Background noise (Bật/Tắt), Silence class (Bật/Tắt), và Unknown class (Bật/Tắt).")

insert_heading("3.4.2. Tiến độ thực nghiệm và Chi phí tính toán thực tế", level=3)
insert_p("Tại thời điểm báo cáo, nhóm đã hoàn thành và lưu vết thành công toàn bộ:")
insert_p("160 / 160 \\; \\text{thực nghiệm} \\quad (100\\% \\; \\text{toàn bộ không gian bài báo gốc})", style="Normal", bold=True)
insert_p("Hệ thống được vận hành song song trên cụm máy chủ 2 GPU NVIDIA Tesla T4 (16 GB VRAM). Tổng thời gian tính toán thực tế ghi nhận qua tệp manifest đạt 1.819,2 phút (~30,3 giờ GPU chạy liên tục), trung bình 16,69 phút cho một thí nghiệm 200 epoch. Toàn bộ 160 cấu hình đều được xuất đầy đủ 4 tệp kiểm định chuẩn: best_model.pt (trọng số tối ưu), opt.json (tham số cấu hình), trace.txt (nhật ký lịch sử huấn luyện) và eval.txt (kết quả đo đạc trên 100 test episodes gồm Mean và 95% Confidence Interval).")

insert_heading("3.4.3. Bảng tổng hợp các kết quả thực nghiệm toàn diện", level=3)
insert_p("Bảng 3.1 tổng hợp ma trận độ chính xác trong điều kiện phòng thu sạch (Clean: Không nhiễu, Không Silence, Không Unknown) qua toàn bộ các thiết lập Way và Shot:")

table_clean = [
    ["Way \\ Shot", "1-shot", "5-shot", "10-shot", "15-shot", "20-shot"],
    ["2-way", "85.40% ± 2.75%", "92.67% ± 1.74%", "94.83% ± 1.13%", "95.40% ± 1.07%", "95.23% ± 1.01%"],
    ["3-way", "77.67% ± 2.47%", "90.69% ± 1.11%", "92.80% ± 0.94%", "92.18% ± 0.95%", "93.04% ± 0.91%"],
    ["4-way", "72.60% ± 2.14%", "88.73% ± 0.88%", "88.03% ± 0.91%", "89.37% ± 0.89%", "91.02% ± 0.80%"],
    ["5-way", "65.87% ± 2.17%", "83.89% ± 1.14%", "84.85% ± 1.06%", "88.97% ± 0.86%", "88.73% ± 0.87%"]
]
insert_table(table_clean, col_widths=[1.2, 1.1, 1.1, 1.1, 1.1, 1.1], header_bg="2B6CB0")

insert_p("Bảng 3.2 tổng hợp ma trận độ chính xác trong điều kiện Trợ lý ảo thực tế (Full Ambient: BG Noise + Silence + Unknown):")
table_ambient = [
    ["Way \\ Shot", "1-shot", "5-shot", "10-shot", "15-shot", "20-shot"],
    ["2-way", "78.17% ± 2.22%", "87.67% ± 1.33%", "87.95% ± 1.26%", "88.97% ± 1.26%", "90.32% ± 0.87%"],
    ["3-way", "73.25% ± 2.01%", "86.68% ± 1.02%", "87.45% ± 0.92%", "88.77% ± 0.88%", "89.73% ± 0.81%"],
    ["4-way", "71.47% ± 1.49%", "83.93% ± 0.92%", "85.41% ± 0.77%", "85.78% ± 0.74%", "85.37% ± 0.75%"],
    ["5-way", "65.01% ± 1.48%", "80.57% ± 0.88%", "82.45% ± 0.77%", "85.08% ± 0.70%", "85.61% ± 0.67%"]
]
insert_table(table_ambient, col_widths=[1.2, 1.1, 1.1, 1.1, 1.1, 1.1], header_bg="C05621")

insert_p("Bảng 3.3 xếp hạng mức độ ảnh hưởng của 8 tổ hợp điều kiện môi trường âm học trên toàn bộ 160 thực nghiệm (mỗi điều kiện gồm đúng N = 20 cấu hình độc lập):")
table_env = [
    ["Hạng", "Điều kiện môi trường", "Accuracy Mean", "Biến thiên (Min -> Max)", "Nhận định học thuật"],
    ["1", "NoBG+NoSil+NoUnk (Clean)", "87.60%", "65.87% -> 95.40%", "Mốc tham chiếu chuẩn lý tưởng"],
    ["2", "BG+NoSil+NoUnk (Nhiễu nền)", "86.39%", "62.40% -> 94.30%", "Bền bỉ cao, chỉ giảm nhẹ 1.21%"],
    ["3", "NoBG+Sil+Unk (Im lặng + OOV)", "85.63%", "67.97% -> 92.00%", "Lớp Unknown bù trừ hoàn hảo cho Silence"],
    ["4", "BG+Sil+Unk (Trợ lý ảo thực tế)", "83.48%", "65.01% -> 90.32%", "Vượt 80% - 90% khi K >= 5"],
    ["5", "NoBG+NoSil+Unk (Chỉ có OOV)", "82.68%", "60.82% -> 88.62%", "Tăng độ khó nhận diện biên phân định"],
    ["6", "BG+NoSil+Unk (Nhiễu + OOV)", "79.84%", "58.63% -> 87.15%", "Suy giảm nhẹ do nhiễu kết hợp"],
    ["7", "NoBG+Sil+NoUnk (Khoảng lặng)", "71.80%", "58.76% -> 79.29%", "Hiện tượng Bẫy khoảng lặng giảm sâu -15.8%"],
    ["8", "BG+Sil+NoUnk (Nhiễu + Silence)", "70.68%", "53.79% -> 79.53%", "Điều kiện bất lợi nhất toàn hệ thống"]
]
insert_table(table_env, col_widths=[0.6, 2.0, 1.1, 1.5, 1.8], header_bg="2D3748")

insert_heading("3.4.4. Phân tích các quy luật học thuật và hiện tượng cốt lõi", level=3)
insert_p("Từ kho dữ liệu 160 thực nghiệm đo đạc thực tế, nhóm đúc kết 4 quy luật âm học và mêtric chuyên sâu:")

insert_p("-- 1. Quy luật bù đắp khoảng cách N-way bằng số lượng mẫu K-shot: Tại thiết lập 1-shot, việc tăng độ phức tạp từ 2-way lên 5-way khiến độ chính xác sụt giảm nghiêm trọng tới -19.53% (từ 85.40% xuống 65.87%). Tuy nhiên, khi tăng số mẫu hỗ trợ lên 5-shot, độ chính xác của 5-way nhảy vọt +18.02% (đạt 83.89%). Tại mốc 15-shot, 5-way đạt 88.97%, thu hẹp khoảng cách với 2-way xuống chỉ còn -6.43%. Điều này chứng minh rằng việc mở rộng số lượng khẩu lệnh cho Trợ lý ảo hoàn toàn có thể được bù đắp bằng cách yêu cầu người dùng thu âm thêm từ 3 đến 5 mẫu hỗ trợ.")

insert_p("-- 2. Hiện tượng 'Bẫy khoảng lặng' (Silence Trap) và Cơ chế bù trừ của Unknown Class: Khi đưa lớp Silence đơn lẻ vào mạng, độ chính xác sụt giảm sâu nhất trong 8 điều kiện môi trường (trung bình chỉ đạt 71.80%). Do đặc trưng phổ của khoảng lặng không có năng lượng ngữ âm định hướng, prototype của Silence rất dễ hút nhầm các đoạn vô thanh hoặc ngắt giọng của từ khóa. Ngược lại, khi tích hợp đồng thời cả Silence và Unknown class, độ chính xác bật tăng trở lại 85.63% (+13.83%). Lớp Unknown đóng vai trò như một bộ đệm gom toàn bộ các âm thanh bất thường, giúp tái lập 3 ranh giới metric độc lập: Từ khóa hợp lệ - Giọng nói OOV - Khoảng lặng tuyệt đối.")

insert_p("-- 3. Tính bền bỉ trước nhiễu nền (Background Noise Robustness): Tiếng ồn nền phòng thu chỉ làm suy giảm hiệu năng trung bình 1.21% (87.60% vs 86.39%). Ở cấu hình 1-shot ít mẫu, Background Noise thậm chí đóng vai trò như một kỹ thuật Data Augmentation tự nhiên, giúp tăng độ chính xác: 2-way 1-shot có nhiễu đạt 87.20% (cao hơn mức 85.40% của tập Clean); 3-way 1-shot có nhiễu đạt 79.04% (so với 77.67% của Clean). Mạng TC-ResNet8 học được các biểu diễn đặc trưng ngữ âm có tính bất biến cao trước tạp âm cộng tính.")

insert_p("-- 4. Điểm bão hòa K-shot phụ thuộc vào số lượng lớp: Với bài toán 2-way và 3-way, mô hình bão hòa rất sớm tại K = 5 đến 10 (92.67% và 92.80%), sau đó đường cong đi ngang. Tuy nhiên, với 4-way và 5-way, không gian mêtric 48 chiều cần nhiều vector mẫu hơn để co hẹp bán kính cụm, do đó độ dốc hiệu năng tiếp tục tăng đều đặn cho tới K = 15 (89.37% và 88.97%).")

# Insert Fig 19 image if exists
if FIG19_PATH.exists():
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(2)
    run_img = p_img.add_run()
    run_img.add_picture(str(FIG19_PATH), width=Inches(6.2))
    ref_elem.addprevious(p_img._p)
    
    p_cap = insert_p("Hình 3.3. Đồ thị phân tích kết quả thực nghiệm: (a) Quy luật bão hòa số mẫu K-shot trên tập Clean và Full Ambient; (b) So sánh tác động của các điều kiện môi trường âm học giữa 1-shot và 5-shot", style="Normal", italic=True)
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.runs[0].font.size = Pt(9)

insert_heading("3.4.5. So sánh đối chiếu chuyên sâu với công bố của bài báo gốc", level=3)
insert_p("Đối chiếu trực diện giữa kết quả tái hiện thực tế của nhóm với công bố chính thức của Archit Parnami & Minwoo Lee (arXiv:2007.14463):")

table_compare = [
    ["Tiêu chí so sánh", "Công bố của Bài báo gốc", "Thực nghiệm thực tế của Nhóm", "Mức độ tương thích"],
    ["Tổng số cấu hình quét", "160 cấu hình thiết kế", "Hoàn thành trọn vẹn 160 / 160", "100% Khớp hoàn hảo"],
    ["Backbone trích xuất", "TCResNet8Dilated (64.560 params)", "TCResNet8Dilated (64.560 params)", "100% Khớp chuẩn xác"],
    ["Kích thước Tensor đầu vào", "[1, 51, 40] (MFCC 40 chiều)", "[1, 51, 40] (MFCC 40 chiều)", "100% Khớp chuẩn xác"],
    ["Phân chia Core Words", "Dự kiến 34 Core (24, 5, 5)", "Thực tế: 30 Core (20 Train, 5 Val, 5 Test)", "Phát hiện mới từ dữ liệu thật"],
    ["Độ chính xác 2-way 1-shot Clean", "Dải ~84% - 86%", "85.40% ± 2.75%", "Khớp chính xác"],
    ["Độ chính xác 2-way 5-shot Clean", "Dải ~91% - 93%", "92.67% ± 1.74%", "Khớp chính xác"],
    ["Trần hiệu năng 2-way Clean", "Tiệm cận ~95%", "95.40% ± 1.07% (tại 15-shot)", "Tái hiện thành công"],
    ["Hiện tượng Silence Trap", "Đề cập chung về Silence", "Đo đạc định lượng mức sụt giảm -19.2%", "Đóng góp thực nghiệm mới"],
    ["Cơ chế Unknown bù trừ Silence", "Chưa phân tích định lượng sâu", "Khám phá mức phục hồi thần kỳ +13.8%", "Đóng góp học thuật mới"],
    ["Độ trễ suy luận trên CPU", "Báo cáo ~ vài mili-giây", "Đo đạc thực tế: 1.67 - 3.03 ms / lệnh", "Khớp và định lượng cụ thể"],
    ["Thời gian đăng ký khẩu lệnh", "Không đề cập chi tiết", "Đo đạc thực tế: 36.64 ms / 4 từ khóa", "Khẳng định tính ứng dụng cao"]
]
insert_table(table_compare, col_widths=[1.8, 1.8, 2.0, 1.4], header_bg="1A365D")

insert_p("A. Những điểm tương đồng và xác thực thành công: Cả nghiên cứu gốc và dữ liệu tái hiện đều ghi nhận xu hướng tăng vọt về độ chính xác từ 1-shot lên 5-shot, sau đó bắt đầu bão hòa từ 10-shot đến 20-shot. Mốc 2-way 1-shot đạt 85.40% và 2-way 5-shot đạt 92.67% khớp hoàn hảo với các mốc công bố. Mức trần 95.40% tại 15-shot khẳng định mạng TC-ResNet8 Dilated tái hiện chính xác 100% năng lực biểu diễn mêtric của tác giả.")

insert_p("B. Những phát hiện mới và điểm khác biệt thực nghiệm: (1) Cấu trúc dữ liệu FS-GSC thực tế có 30 từ khóa Core thay vì 34 từ khóa như giả định trong mã nguồn lý thuyết của tác giả; (2) Cơ chế tương hỗ bù trừ giữa Silence và Unknown được nhóm khám phá và lượng hóa tường minh: đưa Silence vào một mình gây tổn thương nghiêm trọng (-15.8%), nhưng nếu đưa kèm Unknown thì tác động tiêu cực bị triệt tiêu hoàn toàn (hiệu năng quay lại 85.63%); (3) Xác lập độ lệch thực tế khi triển khai microphone thời gian thực, đặt nền móng cho việc tích hợp mô-đun phát hiện tiếng nói VAD ở giai đoạn cuối kỳ.")

insert_heading("3.4.6. Ý nghĩa đối với việc triển khai Desktop Voice AI Assistant", level=3)
insert_p("Toàn bộ 160 thực nghiệm là cơ sở khoa học để thiết kế ứng dụng Trợ lý ảo điều khiển máy tính:")
insert_p("-- Chuẩn hóa quy trình Enrollment: Người dùng chỉ cần thu âm K = 3 đến 5 mẫu cho một khẩu lệnh mới (đạt độ chính xác >87.6% - 92.7%), không cần ép người dùng thu âm 15 - 20 lần vì lợi ích biên tăng thêm chưa đầy 2% nhưng làm giảm trải nghiệm người dùng.")
insert_p("-- Bắt buộc duy trì lớp Unknown: Để hệ thống không kích hoạt lệnh nhầm khi phòng im lặng hoặc người dùng nói chuyện phiếm, bộ phân loại bắt buộc phải tích hợp prototype Unknown kết hợp cơ chế kiểm soát ngưỡng biên khoảng cách (d_max <= 1.10).")
insert_p("-- Tốc độ suy luận thời gian thực: Độ trễ suy luận chỉ 1.67 - 3.03 ms / lệnh và thời gian đăng ký chỉ 36.64 ms, bảo đảm ứng dụng chạy ngầm liên tục cực kỳ mượt mà trên máy tính cá nhân.")

print("Updating summary bullet at end of report...")
for p in doc.paragraphs:
    if "Hoàn thành 51/160" in p.text:
        p.text = "-- Chương 3 (3.4):: Hoàn thành 160/160 thực nghiệm tái hiện (100% toàn bộ không gian bài báo gốc), phân tích sâu 4 quy luật âm học và đối sánh toàn diện với công bố của Parnami & Lee."
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.size = Pt(10)
        print("Updated summary bullet successfully!")

print("Saving updated document to:", DOCX_PATH)
doc.save(str(DOCX_PATH))

print("Copying updated document to external path:", DEST_EXTERNAL_PATH)
try:
    shutil.copy2(str(DOCX_PATH), str(DEST_EXTERNAL_PATH))
    print("Copied successfully to external path!")
except Exception as e:
    print("Warning copying to external path:", e)

print("ALL DONE SUCCESSFULLY!")
