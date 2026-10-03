import os
import glob
import json

base_dir = r"G:\Desktop\voice-AI-assistant\data\paper_reproduction_51of160_20261001_1952"
exp_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("exp_")])

results = []
for exp in exp_dirs:
    exp_path = os.path.join(base_dir, exp)
    subfolders = sorted(glob.glob(os.path.join(exp_path, "*")))
    found = False
    for sub in reversed(subfolders):
        eval_file = os.path.join(sub, "eval.txt")
        opt_file = os.path.join(sub, "opt.json")
        if os.path.exists(eval_file) and os.path.exists(opt_file):
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                with open(opt_file, "r", encoding="utf-8") as f:
                    opt_data = json.load(f)

                loss_m = eval_data.get("test", {}).get("loss", {}).get("mean", None)
                loss_ci = eval_data.get("test", {}).get("loss", {}).get("confidence", None)
                acc_m = eval_data.get("test", {}).get("acc", {}).get("mean", None)
                acc_ci = eval_data.get("test", {}).get("acc", {}).get("confidence", None)

                results.append({
                    "exp": exp,
                    "way": opt_data.get("data.way"),
                    "shot": opt_data.get("data.shot"),
                    "bg": opt_data.get("speech.include_background"),
                    "silence": opt_data.get("speech.include_silence"),
                    "unknown": opt_data.get("speech.include_unknown"),
                    "acc": acc_m,
                    "acc_ci": acc_ci,
                    "loss": loss_m,
                    "loss_ci": loss_ci,
                })
                found = True
                break
            except Exception as e:
                print(f"Error {sub}: {e}")
    if not found:
        results.append({"exp": exp, "error": True})

print(f"Total parsed: {len(results)}")

# Save to CSV
csv_path = r"G:\Desktop\voice-AI-assistant\docs\reproduction_51_experiments.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("exp,way,shot,bg,silence,unknown,acc_mean,acc_ci,loss_mean,loss_ci\n")
    for r in results:
        if "error" not in r:
            f.write(f"{r['exp']},{r['way']},{r['shot']},{r['bg']},{r['silence']},{r['unknown']},{r['acc']},{r['acc_ci']},{r['loss']},{r['loss_ci']}\n")

print(f"Saved CSV to {csv_path}")

# Print sorted by Way, Shot, Conditions
for r in results:
    if "error" not in r:
        bg_s = "BG" if r["bg"] else "NoBG"
        sil_s = "Sil" if r["silence"] else "NoSil"
        unk_s = "Unk" if r["unknown"] else "NoUnk"
        cond = f"{bg_s}+{sil_s}+{unk_s}"
        print(f"{r['exp']}: {r['way']}-way {r['shot']:2d}-shot | {cond:<18} | Acc: {r['acc']*100:6.2f}% +/- {r['acc_ci']*100:4.2f}% | Loss: {r['loss']:.4f}")
