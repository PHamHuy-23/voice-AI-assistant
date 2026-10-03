import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

class Dummy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def __setstate__(self, state):
        self.__dict__.update(state)

for mod in ['protonets', 'protonets.models', 'protonets.models.few_shot', 'protonets.models.encoder', 'protonets.models.encoder.TCResNet', 'protonets.models.encoder.baseUtil']:
    m = types.ModuleType(mod)
    m.Protonet = Dummy
    m.TCResNet = Dummy
    m.ResidualBlock = Dummy
    m.TC = Dummy
    m.Flatten = Dummy
    sys.modules[mod] = m

raw_model = torch.load(r'G:\Desktop\voice-AI-assistant\checkpoints\best.pt', map_location='cpu', weights_only=False)
raw_sd = raw_model.state_dict()

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

out_path = r'G:\Desktop\voice-AI-assistant\checkpoints\tcresnet8_clean_weights.pt'
torch.save(new_sd, out_path)
print(f"[+] Successfully extracted and saved clean PyTorch weights to: {out_path} ({len(new_sd)} tensors)")
