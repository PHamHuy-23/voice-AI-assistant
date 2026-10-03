import csv
from collections import defaultdict

data = []
with open(r'G:\Desktop\voice-AI-assistant\docs\reproduction_51_experiments.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for k in ['way', 'shot']:
            row[k] = int(row[k])
        for k in ['acc_mean', 'acc_ci', 'loss_mean', 'loss_ci']:
            row[k] = float(row[k])
        for k in ['bg', 'silence', 'unknown']:
            row[k] = (row[k] == 'True')
        data.append(row)

print("=== 1. SCALING LAW: 2-WAY CLEAN ===")
for r in data:
    if r['way'] == 2 and not r['bg'] and not r['silence'] and not r['unknown']:
        print(f"{r['shot']:2d}-shot: Acc = {r['acc_mean']*100:6.2f}% +/- {r['acc_ci']*100:4.2f}%, Loss = {r['loss_mean']:.4f}")

print("\n=== 2. AVERAGE ACC BY SHOT ACROSS ALL 8 CONDITIONS (2-WAY) ===")
shot_groups = defaultdict(list)
for r in data:
    if r['way'] == 2:
        shot_groups[r['shot']].append(r['acc_mean'])

for shot in sorted(shot_groups.keys()):
    vals = shot_groups[shot]
    avg_acc = sum(vals)/len(vals)*100
    print(f"{shot:2d}-shot (Mean of 8 conditions): {avg_acc:6.2f}%")

print("\n=== 3. 2-WAY VS 3-WAY COMPARISON (1-SHOT, ALL 8 CONDITIONS) ===")
conds = [
    (False, False, False, 'Clean (NoBG, NoSil, NoUnk)'),
    (True, False, False, 'BG only'),
    (False, True, False, 'Silence only'),
    (False, False, True, 'Unknown only'),
    (True, True, False, 'BG + Silence'),
    (True, False, True, 'BG + Unknown'),
    (False, True, True, 'Silence + Unknown'),
    (True, True, True, 'BG + Silence + Unknown'),
]
for bg, sil, unk, label in conds:
    m2 = [r for r in data if r['way']==2 and r['shot']==1 and r['bg']==bg and r['silence']==sil and r['unknown']==unk]
    m3 = [r for r in data if r['way']==3 and r['shot']==1 and r['bg']==bg and r['silence']==sil and r['unknown']==unk]
    if m2 and m3:
        a2 = m2[0]['acc_mean']*100
        a3 = m3[0]['acc_mean']*100
        diff = a3 - a2
        print(f"{label:<25}: 2-way = {a2:6.2f}% | 3-way = {a3:6.2f}% | Drop: {diff:6.2f}%")

print("\n=== 4. 2-WAY VS 3-WAY COMPARISON (5-SHOT, AVAILABLE 3 CONDITIONS) ===")
for bg, sil, unk, label in conds[:3]:
    m2 = [r for r in data if r['way']==2 and r['shot']==5 and r['bg']==bg and r['silence']==sil and r['unknown']==unk]
    m3 = [r for r in data if r['way']==3 and r['shot']==5 and r['bg']==bg and r['silence']==sil and r['unknown']==unk]
    if m2 and m3:
        a2 = m2[0]['acc_mean']*100
        a3 = m3[0]['acc_mean']*100
        diff = a3 - a2
        print(f"{label:<25}: 2-way = {a2:6.2f}% | 3-way = {a3:6.2f}% | Drop: {diff:6.2f}%")

print("\n=== 5. ENVIRONMENTAL IMPACT RANKING (2-WAY, AVERAGED OVER ALL SHOTS) ===")
cond_accs = defaultdict(list)
for r in data:
    if r['way'] == 2:
        key = ("BG" if r['bg'] else "NoBG", "Sil" if r['silence'] else "NoSil", "Unk" if r['unknown'] else "NoUnk")
        cond_accs[key].append(r['acc_mean'])

sorted_conds = sorted([(k, sum(v)/len(v)*100) for k, v in cond_accs.items()], key=lambda x: x[1], reverse=True)
for cond, avg in sorted_conds:
    name = "+".join(cond)
    print(f"{name:<20}: Avg Acc = {avg:6.2f}%")
