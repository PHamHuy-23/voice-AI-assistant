import os
import glob
import json
import csv

csv_file = r"docs\reproduction_160_experiments.csv"
with open(csv_file, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    for k in ["way", "shot"]:
        r[k] = int(r[k])
    for k in ["acc_mean", "acc_ci", "loss_mean", "loss_ci"]:
        r[k] = float(r[k])
    for k in ["bg", "silence", "unknown"]:
        r[k] = (r[k] == "True")

# 1. Top 10 by raw accuracy
sorted_all = sorted(rows, key=lambda x: x["acc_mean"], reverse=True)
print("=== TOP 10 ACCURACY OVER ALL 160 EXPS ===")
for r in sorted_all[:10]:
    cond = ("BG" if r["bg"] else "NoBG") + "+" + ("Sil" if r["silence"] else "NoSil") + "+" + ("Unk" if r["unknown"] else "NoUnk")
    print(f"{r['exp']}: {r['way']}-way {r['shot']:2d}-shot | {cond:<18} | Acc: {r['acc_mean']*100:6.2f}% +/- {r['acc_ci']*100:4.2f}% | Loss: {r['loss_mean']:.4f}")

# 2. Top in Full Ambient (BG + Sil + Unk)
ambient = [r for r in rows if r["bg"] and r["silence"] and r["unknown"]]
sorted_amb = sorted(ambient, key=lambda x: x["acc_mean"], reverse=True)
print("\n=== TOP 10 IN FULL AMBIENT (BG + Sil + Unk) ===")
for r in sorted_amb[:10]:
    print(f"{r['exp']}: {r['way']}-way {r['shot']:2d}-shot | Acc: {r['acc_mean']*100:6.2f}% +/- {r['acc_ci']*100:4.2f}% | Loss: {r['loss_mean']:.4f}")

# 3. Top in 4-way
four_way = [r for r in rows if r["way"] == 4]
sorted_4w = sorted(four_way, key=lambda x: x["acc_mean"], reverse=True)
print("\n=== TOP 5 IN 4-WAY ===")
for r in sorted_4w[:5]:
    cond = ("BG" if r["bg"] else "NoBG") + "+" + ("Sil" if r["silence"] else "NoSil") + "+" + ("Unk" if r["unknown"] else "NoUnk")
    print(f"{r['exp']}: 4-way {r['shot']:2d}-shot | {cond:<18} | Acc: {r['acc_mean']*100:6.2f}% +/- {r['acc_ci']*100:4.2f}% | Loss: {r['loss_mean']:.4f}")

# 4. Top in 5-way
five_way = [r for r in rows if r["way"] == 5]
sorted_5w = sorted(five_way, key=lambda x: x["acc_mean"], reverse=True)
print("\n=== TOP 5 IN 5-WAY ===")
for r in sorted_5w[:5]:
    cond = ("BG" if r["bg"] else "NoBG") + "+" + ("Sil" if r["silence"] else "NoSil") + "+" + ("Unk" if r["unknown"] else "NoUnk")
    print(f"{r['exp']}: 5-way {r['shot']:2d}-shot | {cond:<18} | Acc: {r['acc_mean']*100:6.2f}% +/- {r['acc_ci']*100:4.2f}% | Loss: {r['loss_mean']:.4f}")

# 5. Check checkpoint paths for the best candidates
print("\n=== CHECKING MODEL CHECKPOINT PATHS ===")
candidate_ids = ["exp_025", "exp_040", "exp_104", "exp_112", "exp_120", "exp_152", "exp_160"]
base_dirs = [
    r"data\paper_reproduction_109of160_20261002_1643\results",
    r"data\paper_reproduction_51of160_20261001_1952"
]

for cid in candidate_ids:
    found = False
    for b in base_dirs:
        edir = os.path.join(b, cid)
        if os.path.exists(edir):
            subs = sorted(glob.glob(os.path.join(edir, "*")))
            for s in reversed(subs):
                ckpt = os.path.join(s, "best_model.pt")
                if os.path.exists(ckpt):
                    print(f"[{cid}] Found: {ckpt} (Size: {os.path.getsize(ckpt):,} bytes)")
                    found = True
                    break
        if found:
            break
    if not found:
        print(f"[{cid}] NOT FOUND")
