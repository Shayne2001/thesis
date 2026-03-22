import torch
from thop import profile

# Model
from model import *

device = 'cuda'
print('==> Building model..')
# model = CTCSN(snr=0, cr=1).to(device)

model = SSANet(cr=1)  
state_dict = torch.load('ckpt/35_690.pth' ,map_location=device)

if device=='cpu':
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
else:
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(state_dict)


dummy_input = torch.randn(1, 172, 256, 4).to(device)
output = model(dummy_input)
# print("模型输出:", output.shape)

# 临时解包 DataParallel
if isinstance(model, torch.nn.DataParallel):
    model = model.module  # 获取原始模型

flops, params = profile(model, inputs=(dummy_input,))

# flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))