import os
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import wave
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo.app_gui import FewShotEngine, KEYWORDS_DIR, calculate_rms, center_voice_energy


def load_wav_as_float32(wav_path: str) -> np.ndarray:
    with wave.open(wav_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    print("=" * 70)
    print("KIỂM TRA ĐỘ CHÍNH XÁC NHẬN DẠNG TRÊN CÁC TỆP WAV ĐÃ THU CỦA NGƯỜI DÙNG")
    print("=" * 70)

    engine = FewShotEngine()
    keywords = sorted(list(engine.prototypes.keys()))
    print(f"\n[1] Các từ khóa đã đăng ký ({len(keywords)} từ khóa):")
    for kw in keywords:
        thresh = engine.prototype_thresholds.get(kw, 0.0)
        kw_dir = os.path.join(KEYWORDS_DIR, kw)
        wav_count = len([f for f in os.listdir(kw_dir) if f.endswith(".wav")])
        print(f"  - {kw:<12}: {wav_count:2d} mẫu | Ngưỡng required_similarity: {thresh:.4f}")

    print("\n" + "=" * 70)
    print("[2] KIỂM TRA TRỰC TIẾP QUA classify_audio() (Mô phỏng luồng nhận dạng thực tế):")
    print("=" * 70)

    total_samples = 0
    correct_top1 = 0
    accepted_matches = 0
    results_by_kw = {kw: {"total": 0, "top1": 0, "accepted": 0, "rejections": []} for kw in keywords}

    header = f"{'True Class':<12} | {'File':<14} | {'Predicted':<12} | {'Sim':<7} | {'ReqSim':<7} | {'Margin':<7} | {'Status':<10} | {'Match?'}"
    print(header)
    print("-" * len(header))

    confusion_matrix = {k1: {k2: 0 for k2 in keywords} for k1 in keywords}

    for kw in keywords:
        kw_dir = os.path.join(KEYWORDS_DIR, kw)
        wav_files = sorted([f for f in os.listdir(kw_dir) if f.endswith(".wav")])

        for fname in wav_files:
            total_samples += 1
            results_by_kw[kw]["total"] += 1
            wav_path = os.path.join(kw_dir, fname)
            audio = load_wav_as_float32(wav_path)

            res = engine.classify_audio(audio)
            status = res.get("status", "ERR")
            pred = res.get("best_keyword", "NONE")
            sim = res.get("similarity", 0.0)
            req_sim = res.get("required_similarity", 0.0)
            margin = res.get("winner_margin", 0.0)
            is_match = res.get("is_match", False)

            if pred in confusion_matrix[kw]:
                confusion_matrix[kw][pred] += 1

            is_top1 = (pred == kw)
            if is_top1:
                correct_top1 += 1
                results_by_kw[kw]["top1"] += 1

            if is_match and is_top1:
                accepted_matches += 1
                results_by_kw[kw]["accepted"] += 1
                match_str = "✅ PASS"
            elif is_top1 and not is_match:
                reason = []
                if sim < req_sim:
                    reason.append(f"Sim {sim:.3f} < {req_sim:.3f}")
                if margin is not None and margin < res.get("required_margin", 0.04):
                    reason.append(f"Margin {margin:.3f} < req")
                match_str = f"⚠️ REJECT ({', '.join(reason)})"
                results_by_kw[kw]["rejections"].append((fname, match_str))
            else:
                match_str = f"❌ WRONG -> {pred}"
                results_by_kw[kw]["rejections"].append((fname, match_str))

            m_val = f"{margin:.3f}" if margin is not None else "N/A"
            print(f"{kw:<12} | {fname:<14} | {pred:<12} | {sim:6.3f}  | {req_sim:6.3f}  | {m_val:<7} | {status:<10} | {match_str}")

    print("\n" + "=" * 70)
    print("[3] TỔNG HỢP THEO TỪ KHÓA:")
    print("=" * 70)
    for kw in keywords:
        st = results_by_kw[kw]
        top1_pct = (st["top1"] / st["total"]) * 100 if st["total"] > 0 else 0
        acc_pct = (st["accepted"] / st["total"]) * 100 if st["total"] > 0 else 0
        print(f"  • {kw:<12}: Top-1 Accuracy: {st['top1']}/{st['total']} ({top1_pct:.1f}%) | Accepted Match: {st['accepted']}/{st['total']} ({acc_pct:.1f}%)")
        if st["rejections"]:
            for fname, reason in st["rejections"]:
                print(f"      - {fname}: {reason}")

    top1_overall = (correct_top1 / total_samples) * 100 if total_samples > 0 else 0
    accepted_overall = (accepted_matches / total_samples) * 100 if total_samples > 0 else 0
    print("-" * 70)
    print(f"Tổng số mẫu thử: {total_samples}")
    print(f"Top-1 Correct (nhận diện đúng từ khóa): {correct_top1}/{total_samples} ({top1_overall:.1f}%)")
    print(f"Vượt qua bộ lọc ngưỡng (is_match = True): {accepted_matches}/{total_samples} ({accepted_overall:.1f}%)")

    print("\n" + "=" * 70)
    print("[4] MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
    print("=" * 70)
    header_cm = f"{'True \\ Pred':<12} | " + " | ".join([f"{k:<10}" for k in keywords])
    print(header_cm)
    print("-" * len(header_cm))
    for true_k in keywords:
        row = f"{true_k:<12} | " + " | ".join([f"{confusion_matrix[true_k][pred_k]:<10d}" for pred_k in keywords])
        print(row)

    print("\n" + "=" * 70)
    print("[5] KIỂM TRA LEAVE-ONE-OUT (LOOCV - Mẫu mới chưa từng thấy khi tạo prototype):")
    print("=" * 70)
    loocv_correct = 0
    loocv_total = 0

    # For each keyword and each wav, build prototype on all other wavs of that keyword
    # and all wavs of other keywords.
    for kw in keywords:
        kw_dir = os.path.join(KEYWORDS_DIR, kw)
        wav_files = sorted([f for f in os.listdir(kw_dir) if f.endswith(".wav")])
        if len(wav_files) <= 1:
            continue

        for i, test_fname in enumerate(wav_files):
            loocv_total += 1
            test_path = os.path.join(kw_dir, test_fname)
            test_emb = engine.wav_to_normalized_embedding(test_path)

            # Build temporary prototypes
            temp_prototypes = {}
            for other_kw in keywords:
                o_dir = os.path.join(KEYWORDS_DIR, other_kw)
                o_wavs = sorted([f for f in os.listdir(o_dir) if f.endswith(".wav")])
                if other_kw == kw:
                    train_wavs = [w for j, w in enumerate(o_wavs) if j != i]
                else:
                    train_wavs = o_wavs

                embs = [engine.wav_to_normalized_embedding(os.path.join(o_dir, w)) for w in train_wavs]
                cat_embs = torch.cat(embs, dim=0)
                proto = F.normalize(cat_embs.mean(dim=0, keepdim=True), p=2, dim=-1)
                temp_prototypes[other_kw] = proto

            # Classify
            sims = {k: torch.sum(test_emb * p).item() for k, p in temp_prototypes.items()}
            best_k = max(sims, key=sims.get)
            if best_k == kw:
                loocv_correct += 1
            else:
                print(f"  [LOOCV Mismatch] {kw}/{test_fname} nhầm thành {best_k} (Sim({kw})={sims[kw]:.3f}, Sim({best_k})={sims[best_k]:.3f})")

    loocv_acc = (loocv_correct / loocv_total) * 100 if loocv_total > 0 else 0
    print(f"\nKết quả LOOCV: {loocv_correct}/{loocv_total} ({loocv_acc:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
