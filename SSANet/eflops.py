
import torch
from thop import profile

# Model
from model import *
# 1. 实例化完整的 SSANet 模型以获取其编码器部分
# 您可以更改 cr 的值来测试不同配置下的编码器
model = SSANet(cr=1)

# 2. 创建一个符合编码器输入尺寸的虚拟输入
# 编码器的输入尺寸为 (batch_size, bands, height, width)
dummy_input = torch.randn(1, 172, 128, 4)

# 3. 核心改动：将模型的 .encoder 部分传入 profile 函数
# 这样可以确保只计算编码器的 FLOPs 和参数量
# 添加 verbose=False 来避免 thop 库打印每一层的详细信息
flops, params = profile(model.encoder, (dummy_input,), verbose=False)

# 4. 打印结果，并明确指出这是编码器的数据
print('--- SSANet Encoder Analysis ---')
print(f'Input shape: {list(dummy_input.shape)}')
print(f'Encoder GFLOPs: {flops / 1e9:.6f} G')
print(f'Encoder Params: {params / 1e6:.2f} M')
print('-------------------------------')