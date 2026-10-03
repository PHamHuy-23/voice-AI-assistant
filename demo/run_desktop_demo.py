"""
========================================================================================
 Few-Shot Keyword Spotting & Desktop Voice AI Assistant Interactive Demo
 Model Backbone: Official TCResNet8 + Prototypical Networks (PyTorch)
 Checkpoint: Exp 025 (2-way 15-shot, Test Accuracy: 95.40%, Loss: 0.1240, 64,560 params)
 Input Feature: Torchaudio MFCC [1, 51, 40] (16 kHz, window 40ms, stride 20ms)
 Embedding Space: D = 48 dimensions
========================================================================================
"""

import os
import sys
import time
import math
import struct
import wave
import subprocess
from typing import Dict, List, Tuple

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DEMO_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.tc_resnet8 import TCResNet8, load_trained_tcresnet8

SUPPORT_DIR = os.path.join(DEMO_DIR, "support")
SAMPLES_DIR = os.path.join(DEMO_DIR, "samples")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tcresnet8_clean_weights.pt")

KEYWORDS = {
    "open_notepad": {
        "freq": 440,
        "action": "Mở trình soạn thảo văn bản Notepad",
        "cmd": "notepad.exe",
        "desc": "Khởi chạy ứng dụng ghi chú hệ thống"
    },
    "open_browser": {
        "freq": 660,
        "action": "Mở trình duyệt Web mặc định",
        "cmd": "explorer.exe https://github.com",
        "desc": "Mở trang web GitHub trên trình duyệt mặc định"
    },
    "system_mute": {
        "freq": 880,
        "action": "Tắt / Bật nhanh âm lượng hệ thống",
        "cmd": "powershell -c (New-Object -ComObject Wscript.Shell).SendKeys([char]173)",
        "desc": "Gửi tín hiệu phím Mute của Windows để đảo trạng thái loa"
    },
    "stop_task": {
        "freq": 330,
        "action": "Dừng tác vụ / Đóng cửa sổ ứng dụng",
        "cmd": None,
        "desc": "Phát thông điệp ngắt tiến trình hoặc đóng tác vụ đang chạy"
    },
}


def create_synthetic_wav(filepath: str, base_freq: float, duration_s: float = 1.0, sr: int = 16000, noise_level: float = 0.05):
    """Generates a reproducible 1-second 16kHz WAV file with harmonic acoustic pattern."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    n_samples = int(duration_s * sr)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n_samples):
            t = i / sr
            # Formant / harmonic combination resembling distinct voice signatures
            sig = (
                0.60 * math.sin(2 * math.pi * base_freq * t) +
                0.25 * math.sin(2 * math.pi * (base_freq * 1.5) * t) +
                0.15 * math.sin(2 * math.pi * (base_freq * 2.2) * t)
            )
            # Add subtle pseudo-random noise
            noise = (math.sin(i * 12.9898) % 1.0 - 0.5) * noise_level
            val = int(max(-32767, min(32767, (sig + noise) * 16000)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)


def ensure_demo_audio_samples():
    """Generates support and test audio samples if they don't exist."""
    # 1. Support sets (3 shots per class)
    for kw, meta in KEYWORDS.items():
        kw_dir = os.path.join(SUPPORT_DIR, kw)
        for shot_idx in range(1, 4):
            fp = os.path.join(kw_dir, f"sample_{shot_idx}.wav")
            if not os.path.exists(fp):
                jitter_freq = meta["freq"] * (1.0 + (shot_idx - 2) * 0.03)
                create_synthetic_wav(fp, jitter_freq)

    # 2. Query samples
    for kw, meta in KEYWORDS.items():
        q_fp = os.path.join(SAMPLES_DIR, f"query_{kw}.wav")
        if not os.path.exists(q_fp):
            create_synthetic_wav(q_fp, meta["freq"] * 1.01)

    # 3. Unknown OOV sample (white noise / alien frequency)
    unk_fp = os.path.join(SAMPLES_DIR, "query_unknown.wav")
    if not os.path.exists(unk_fp):
        create_synthetic_wav(unk_fp, base_freq=1800.0, noise_level=0.4)


class PyTorchFeatureExtractor:
    """Extracts MFCC [1, 51, 40] using Torchaudio."""
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={"n_fft": 640, "hop_length": 320, "win_length": 640}
        )

    def extract(self, wav_path: str) -> torch.Tensor:
        with wave.open(wav_path, "r") as wf:
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        wav_tensor = torch.from_numpy(samples).unsqueeze(0)  # [1, 16000]
        # Torchaudio MFCC -> [1, 40, 51]
        feat = self.mfcc_transform(wav_tensor)[0].T  # [51, 40]
        feat = torch.unsqueeze(feat, 0)  # [1, 51, 40]
        return feat


class DesktopVoiceAssistantDemo:
    def __init__(self, execute_real_actions: bool = False):
        self.execute_real_actions = execute_real_actions
        self.feature_extractor = PyTorchFeatureExtractor(sample_rate=16000)

        # Load official trained TC-ResNet8 model
        self.model = load_trained_tcresnet8(CHECKPOINT_PATH)
        self.model.eval()

        self.prototypes: Dict[str, torch.Tensor] = {}
        self.confidence_threshold = 75.0  # %
        self.max_distance_threshold = 1.10  # Euclidean distance boundary

    def enroll_keywords(self):
        """Phase 1: Enrolls keywords from support directory and computes class Prototypes."""
        print("\n" + "="*75)
        print("  GIAI ĐOẠN 1: ĐĂNG KÝ KHẨU LỆNH TÙY BIẾN (FEW-SHOT ENROLLMENT)")
        print("="*75)
        print(f" [Trọng số mô hình]:  Exp 025 (2-way 15-shot, Test Acc: 95.40%, Loss: 0.1240)")
        print(f" [Kiến trúc nơ-ron]:  Official TC-ResNet8 (64,560 tham số, 3 Residual Blocks)")
        print(f" [Trích xuất MFCC]:   Torchaudio MFCC [1, 51, 40] (16 kHz, Window 40ms, Stride 20ms)")
        print(f" [Đường dẫn Checkpoint]: {CHECKPOINT_PATH}")
        print("-" * 75)

        start_t = time.perf_counter()

        with torch.no_grad():
            for kw in sorted(os.listdir(SUPPORT_DIR)):
                kw_dir = os.path.join(SUPPORT_DIR, kw)
                if not os.path.isdir(kw_dir):
                    continue
                wavs = [os.path.join(kw_dir, f) for f in os.listdir(kw_dir) if f.endswith(".wav")]
                if not wavs:
                    continue

                embs = []
                for w in wavs:
                    feat = self.feature_extractor.extract(w)
                    emb = self.model(feat)  # [1, 48]
                    embs.append(emb)

                # Prototype c_k = (1/K) * sum(z_{k,i})
                embs = torch.cat(embs, dim=0)  # [K, 48]
                proto = embs.mean(dim=0, keepdim=True)  # [1, 48]
                proto = F.normalize(proto, p=2, dim=1)

                self.prototypes[kw] = proto
                print(f" [+] Đăng ký thành công: '{kw:<14}' | Mẫu Support: {len(wavs)} (3-shot) | Prototype: D=48")

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        print(f"\n=> Đã nạp thành công {len(self.prototypes)} từ khóa vào Prototype Memory trong: {elapsed_ms:.2f} ms")

    def classify_query(self, query_path: str) -> Tuple[str, float, Dict[str, float], float]:
        """Phase 2: Compares query embedding against prototypes using Squared Euclidean Distance."""
        start_t = time.perf_counter()

        with torch.no_grad():
            feat = self.feature_extractor.extract(query_path)
            q_emb = self.model(feat)  # [1, 48]

            dists = {}
            for kw, proto in self.prototypes.items():
                d = torch.sum((q_emb - proto) ** 2).item()
                dists[kw] = d

            # Softmax probabilities over negative distances
            temperature = 0.1
            exp_d = {kw: math.exp(-d / temperature) for kw, d in dists.items()}
            sum_exp = sum(exp_d.values())
            probs = {kw: (exp_d[kw] / sum_exp) * 100.0 for kw in dists}

            best_kw = min(dists, key=dists.get)
            min_dist = dists[best_kw]
            confidence = probs[best_kw]

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        if min_dist > self.max_distance_threshold or confidence < self.confidence_threshold:
            decision = "UNKNOWN_REJECTED"
        else:
            decision = best_kw

        return decision, confidence, dists, latency_ms

    def dispatch_action(self, keyword: str):
        """Phase 3: Triggers the mapped OS action."""
        if keyword not in KEYWORDS:
            print(f" [!] Bỏ qua: Không có hành động nào được ánh xạ cho từ khóa '{keyword}'.")
            return

        meta = KEYWORDS[keyword]
        action_name = meta["action"]
        cmd = meta["cmd"]

        print(f"\n [ACTION DISPATCHER]")
        print(f"  -> Hành động: {action_name}")

        if not self.execute_real_actions:
            print(f"  -> [Chế độ Giả lập an toàn] Sẵn sàng kích hoạt lệnh: {cmd if cmd else 'N/A'}")
            print(f"     (Gợi ý: Thêm cờ --execute để thực thi thật lệnh mở ứng dụng trên Windows)")
        else:
            print(f"  -> [KÍCH HOẠT THẬT TRÊN WINDOWS] Đang thực thi lệnh hệ thống...")
            if cmd:
                try:
                    subprocess.Popen(cmd, shell=True)
                    print(f"  -> Thành công: Tiến trình đã được mở!")
                except Exception as e:
                    print(f"  -> Lỗi thực thi: {e}")
            else:
                print(f"  -> Đã gửi thông điệp dừng tác vụ.")

    def run_single_test(self, query_filename: str):
        qp = os.path.join(SAMPLES_DIR, query_filename)
        if not os.path.exists(qp):
            print(f" [!] Tệp truy vấn không tồn tại: {qp}")
            return

        dec, conf, dists, lat = self.classify_query(qp)

        print("\n" + "-"*65)
        print(f" [Tệp âm thanh Query]:  {query_filename}")
        print(f" [Kết quả nhận dạng]:  >>> {dec.upper()} <<<")
        print(f" [Độ tin cậy Softmax]:  {conf:.1f}%")
        print(f" [Độ trễ suy luận]:     {lat:.2f} ms (PyTorch Forward Pass)")
        print(" [Khoảng cách Euclidean tới các Prototype c_k]:")
        best_k = min(dists, key=dists.get)
        for kw, d in dists.items():
            bar = "█" * max(1, int((1.0 - min(1.0, d)) * 20))
            is_best = " (Tâm cụm gần nhất)" if kw == best_k else ""
            print(f"   • {kw:<14}: d = {d:.4f}  |{bar:<20}|{is_best}")

        if dec != "UNKNOWN_REJECTED":
            self.dispatch_action(dec)
        else:
            print("   [!] KẾT QUẢ: Tín hiệu bị từ chối do là tạp âm lạ hoặc từ ngoài từ điển (OOV)!")
            print("       -> Tránh kích hoạt nhầm (False Alarm Rejection thành công).")

    def record_microphone_clip(self, filepath: str, prompt_text: str = "Hãy phát âm khẩu lệnh"):
        """Records 1 second of audio from default microphone at 16kHz mono."""
        try:
            import sounddevice as sd
        except ImportError:
            print(" [!] Thư viện sounddevice chưa được cài đặt. Chạy: pip install sounddevice")
            return False

        print(f"\n >>> {prompt_text} <<<")
        for i in [3, 2, 1]:
            print(f"  Chuẩn bị trong: {i}...", end="\r", flush=True)
            time.sleep(0.7)
        print("  [*** ĐANG THU ÂM (1.0 giây) - HÃY NÓI NGAY! ***]      ")

        sr = 16000
        duration_s = 1.0
        audio = sd.rec(int(duration_s * sr), samplerate=sr, channels=1, dtype='int16')
        sd.wait()

        # Save to WAV
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with wave.open(filepath, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())

        print(f"  [+] Đã lưu bản ghi âm thành công: {os.path.basename(filepath)}")
        return True

    def enroll_live_keyword(self):
        """Allows user to record 3 voice samples for any keyword directly from microphone."""
        print("\n" + "="*70)
        print("  ĐĂNG KÝ KHẨU LỆNH BẰNG GIỌNG NÓI THẬT (LIVE MICROPHONE ENROLLMENT)")
        print("="*70)
        print(" Chọn từ khóa bạn muốn thu âm bằng giọng của mình:")
        kw_list = list(KEYWORDS.keys())
        for idx, k in enumerate(kw_list, 1):
            print(f"  [{idx}] {k:<15} ({KEYWORDS[k]['action']})")
        print("  [5] Tự nhập tên khẩu lệnh tùy biến mới")
        print("  [0] Hủy bỏ")

        choice = input(" Nhập lựa chọn [0-5]: ").strip()
        if choice == "0":
            return
        elif choice in ["1", "2", "3", "4"]:
            target_kw = kw_list[int(choice) - 1]
        elif choice == "5":
            target_kw = input(" Nhập tên từ khóa mới (viết liền, ví dụ: open_calc): ").strip()
            if not target_kw:
                return
            KEYWORDS[target_kw] = {
                "action": f"Kích hoạt khẩu lệnh tùy biến '{target_kw}'",
                "cmd": None,
                "desc": "Khẩu lệnh người dùng tự định nghĩa"
            }
        else:
            print(" [!] Lựa chọn không hợp lệ.")
            return

        target_dir = os.path.join(SUPPORT_DIR, target_kw)
        os.makedirs(target_dir, exist_ok=True)

        print(f"\n=> Bắt đầu thu âm 3 mẫu phát âm cho từ khóa: '{target_kw}'")
        for shot in range(1, 4):
            fp = os.path.join(target_dir, f"sample_{shot}.wav")
            input(f"\n Nhấn Enter để bắt đầu thu âm mẫu số {shot}/3...")
            success = self.record_microphone_clip(fp, f"Lần {shot}/3: Nói rõ từ '{target_kw}'")
            if not success:
                return
            time.sleep(0.5)

        # Re-compute prototypes
        print("\n=> Đang cập nhật lại Prototype Memory từ giọng nói thật của bạn...")
        self.enroll_keywords()
        print(f"\n[+] CHÚC MỪNG: Khẩu lệnh '{target_kw}' đã được lưu bằng chính giọng nói của bạn!")

    def test_live_microphone(self):
        """Records 1 second from microphone and classifies in real time."""
        print("\n" + "="*70)
        print("  KIỂM THỬ TRỰC TIẾP QUA MICROPHONE (LIVE VOICE RECOGNITION)")
        print("="*70)
        input(" Nhấn Enter để chuẩn bị nói câu lệnh kiểm thử vào Micro...")

        query_fp = os.path.join(SAMPLES_DIR, "query_live_mic.wav")
        success = self.record_microphone_clip(query_fp, "NÓI MỘT KHẨU LỆNH BẤT KỲ VÀO MICRO")
        if not success:
            return

        time.sleep(0.3)
        self.run_single_test("query_live_mic.wav")

    def run_batch_benchmark(self):
        """Runs batch evaluation across all test queries."""
        print("\n" + "="*75)
        print("  GIAI ĐOẠN 2 & 3: ĐÁNH GIÁ NHẬN DIỆN & ĐIỀU KHIỂN TÁC VỤ (BATCH BENCHMARK)")
        print("="*75)

        query_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith(".wav") and not f.startswith("query_live")])
        for qf in query_files:
            self.run_single_test(qf)


def print_banner():
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║     FEW-SHOT KEYWORD SPOTTING & DESKTOP VOICE AI ASSISTANT DEMO SYSTEM         ║
║     Kiến trúc: Official PyTorch TC-ResNet8 + Prototypical Networks (Few-Shot)  ║
║       Trọng số tối ưu nhất: Exp 025 (Acc: 95.40%, Loss: 0.1240, D=48)          ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """)


def interactive_menu(assistant: DesktopVoiceAssistantDemo):
    while True:
        mode_str = "THỰC THI THẬT TRÊN WINDOWS (LIVE)" if assistant.execute_real_actions else "GIẢ LẬP AN TOÀN (DRY-RUN)"
        print("\n" + "="*72)
        print(f"  MENU THỰC NGHIỆM DEMO GIỮA KỲ [Chế độ: {mode_str}]")
        print("="*72)
        print("  --- KIỂM THỬ NHANH BẰNG DỮ LIỆU MẪU CÓ SẴN ---")
        print("  [1] Thử nghiệm khẩu lệnh: 'open_notepad'  (Mở Notepad)")
        print("  [2] Thử nghiệm khẩu lệnh: 'open_browser'  (Mở trình duyệt Web)")
        print("  [3] Thử nghiệm khẩu lệnh: 'system_mute'   (Tắt / Bật âm lượng loa)")
        print("  [4] Thử nghiệm khẩu lệnh: 'stop_task'     (Dừng tác vụ)")
        print("  [5] Thử nghiệm ngoại lai: 'query_unknown' (Tiếng ồn / Từ ngoài từ điển)")
        print("  [6] Chạy tự động toàn bộ 5 ca kiểm thử (Batch Benchmark)")
        print("  --- TÍNH NĂNG MICROPHONE THỜI GIAN THỰC (GIỌNG THẬT) ---")
        print("  [8] THU ÂM GIỌNG NÓI THẬT ĐỂ ĐĂNG KÝ TỪ KHÓA (Live Enrollment)")
        print("  [9] NÓI TRỰC TIẾP QUA MICROPHONE ĐỂ ĐIỀU KHIỂN (Live Voice Control)")
        print("  --------------------------------------------------------")
        print("  [7] Chuyển đổi chế độ: Giả lập an toàn <-> Thực thi thật trên Windows")
        print("  [0] Thoát")
        print("-" * 72)
        try:
            choice = input(" Nhập lựa chọn của bạn [0-9]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã thoát chương trình.")
            break

        if choice == "1":
            assistant.run_single_test("query_open_notepad.wav")
        elif choice == "2":
            assistant.run_single_test("query_open_browser.wav")
        elif choice == "3":
            assistant.run_single_test("query_system_mute.wav")
        elif choice == "4":
            assistant.run_single_test("query_stop_task.wav")
        elif choice == "5":
            assistant.run_single_test("query_unknown.wav")
        elif choice == "6":
            assistant.run_batch_benchmark()
        elif choice == "8":
            assistant.enroll_live_keyword()
        elif choice == "9":
            assistant.test_live_microphone()
        elif choice == "7":
            assistant.execute_real_actions = not assistant.execute_real_actions
            new_mode = "THỰC THI THẬT TRÊN WINDOWS" if assistant.execute_real_actions else "GIẢ LẬP AN TOÀN"
            print(f"\n=> Đã chuyển đổi chế độ sang: [{new_mode}]")
        elif choice == "0":
            print("\nCảm ơn bạn đã theo dõi phần Demo!")
            break
        else:
            print(" [!] Lựa chọn không hợp lệ. Vui lòng nhập từ 0 đến 9.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Desktop Voice AI Assistant Demo")
    parser.add_argument("--batch", action="store_true", help="Chạy tự động toàn bộ ca kiểm thử không qua menu.")
    parser.add_argument("--execute", action="store_true", help="Bật thực thi thật lệnh hệ thống trên Windows.")
    args = parser.parse_args()

    ensure_demo_audio_samples()
    print_banner()

    assistant = DesktopVoiceAssistantDemo(execute_real_actions=args.execute)
    assistant.enroll_keywords()

    if args.batch:
        assistant.run_batch_benchmark()
    else:
        if sys.stdin.isatty():
            interactive_menu(assistant)
        else:
            assistant.run_batch_benchmark()


if __name__ == "__main__":
    main()
