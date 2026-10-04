import os
import sys
import types

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch

import torch.nn as nn
import torch.nn.functional as F

# Mock protonets modules so torch.load can unpickle the full model
class Dummy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def __setstate__(self, state):
        self.__dict__.update(state)

for mod in [
    'protonets',
    'protonets.models',
    'protonets.models.few_shot',
    'protonets.models.encoder',
    'protonets.models.encoder.TCResNet',
    'protonets.models.encoder.baseUtil'
]:
    m = types.ModuleType(mod)
    m.Protonet = Dummy
    m.TCResNet = Dummy
    m.ResidualBlock = Dummy
    m.TC = Dummy
    m.Flatten = Dummy
    sys.modules[mod] = m

def load_sd(ckpt_path):
    obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if hasattr(obj, 'state_dict'):
        return obj.state_dict()
    return obj

def extract_clean_sd(raw_sd):
    new_sd = {}
    new_sd['conv1.weight'] = raw_sd['encoder.conv1.weight']
    for res_idx, res_name in [(1, 'res1'), (2, 'res2'), (3, 'res3')]:
        prefix = f'encoder.resnet.Res_{res_idx}.'
        new_sd[f'{res_name}.conv1.weight'] = raw_sd[f'{prefix}conv1.weight']
        new_sd[f'{res_name}.bn1.weight'] = raw_sd[f'{prefix}bn1.weight']
        new_sd[f'{res_name}.bn1.bias'] = raw_sd[f'{prefix}bn1.bias']
        new_sd[f'{res_name}.bn1.running_mean'] = raw_sd[f'{prefix}bn1.running_mean']
        new_sd[f'{res_name}.bn1.running_var'] = raw_sd[f'{prefix}bn1.running_var']

        new_sd[f'{res_name}.conv2.weight'] = raw_sd[f'{prefix}conv2.weight']
        new_sd[f'{res_name}.bn2.weight'] = raw_sd[f'{prefix}bn2.weight']
        new_sd[f'{res_name}.bn2.bias'] = raw_sd[f'{prefix}bn2.bias']
        new_sd[f'{res_name}.bn2.running_mean'] = raw_sd[f'{prefix}bn2.running_mean']
        new_sd[f'{res_name}.bn2.running_var'] = raw_sd[f'{prefix}bn2.running_var']

        new_sd[f'{res_name}.conv3.weight'] = raw_sd[f'{prefix}conv3.weight']
        new_sd[f'{res_name}.bn3.weight'] = raw_sd[f'{prefix}bn3.weight']
        new_sd[f'{res_name}.bn3.bias'] = raw_sd[f'{prefix}bn3.bias']
        new_sd[f'{res_name}.bn3.running_mean'] = raw_sd[f'{prefix}bn3.running_mean']
        new_sd[f'{res_name}.bn3.running_var'] = raw_sd[f'{prefix}bn3.running_var']
    return new_sd

p_exp025 = r'checkpoints\best_model_exp025_95.40pct.pt'
p_exp040 = r'checkpoints\best_robust_exp040_90.32pct.pt'

sd025 = extract_clean_sd(load_sd(p_exp025))
sd040 = extract_clean_sd(load_sd(p_exp040))

print('=' * 85)
print(' SO SÁNH TRỌNG SỐ: EXP_025 (Clean Benchmark 95.40%) vs EXP_040 (Ambient Robust 90.32%)')
print('=' * 85)
print(f'{"Tầng trọng số (Layer)":<28} | {"Kích thước Tensor":<18} | {"Cosine Sim":<12} | {"L1 Diff":<10} | {"Nhận định tương quan"}')
print('-' * 85)

cos_sims = []
l1_diffs = []
for k in sd025:
    w1 = sd025[k].float().flatten()
    w2 = sd040[k].float().flatten()
    
    # Cosine similarity
    cos = F.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0)).item()
    l1 = torch.mean(torch.abs(w1 - w2)).item()
    
    cos_sims.append(cos)
    l1_diffs.append(l1)
    
    if cos > 0.8:
        eval_txt = "Rất tương đồng (>=0.80)"
    elif cos > 0.5:
        eval_txt = "Tương đồng khá (>0.50)"
    elif cos > 0.2:
        eval_txt = "Hơi tương đồng (>0.20)"
    elif cos > -0.2:
        eval_txt = "Trực giao / Phân kỳ (~0)"
    else:
        eval_txt = "Đối nghịch âm (<0)"
        
    print(f'{k:<28} | {str(list(sd025[k].shape)):<18} | {cos:+10.4f}   | {l1:8.5f}   | {eval_txt}')

print('=' * 85)
print(f'>> Độ tương đồng Cosin trung bình trên toàn bộ mạng : {sum(cos_sims)/len(cos_sims):+.4f}')
print(f'>> Sai số tuyệt đối L1 trung bình (Mean Absolute Diff) : {sum(l1_diffs)/len(l1_diffs):.5f}')
print('=' * 85)

# Kiểm tra độ tương đồng embedding khi đưa dữ liệu âm thanh qua cả 2 mô hình
# Add project root
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.append(proj_root)

from src.models.tc_resnet8 import TCResNet8


m025 = TCResNet8()
m025.load_state_dict(sd025)
m025.eval()

m040 = TCResNet8()
m040.load_state_dict(sd040)
m040.eval()

# Test with 10 random MFCC inputs
torch.manual_seed(42)
fake_mfcc = torch.randn(10, 1, 51, 40)
with torch.no_grad():
    z025 = m025(fake_mfcc)
    z040 = m040(fake_mfcc)
    emb_sim = F.cosine_similarity(z025, z040, dim=1).mean().item()

print(f'\n[+] ĐỘ TƯƠNG ĐỒNG KHÔNG GIAN NHÚNG EMBEDDING (D=48):')
print(f'    - Thử nghiệm trên các đặc trưng âm học:')
print(f'    - Cosine Similarity trung bình của vector embedding z: {emb_sim:+.4f}')
print('=' * 85)
