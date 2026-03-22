import torch
from thop import profile

# Model
from model.CTCSN import CTCSN

device = torch.device('cuda:0')
print('==> Building model..')
model = CTCSN(snr=0, cr=1).to(device)

# Create dummy input for the encoder
# Assuming the encoder takes input of shape (1, 172, 256, 4)
dummy_input = torch.randn(1, 172, 256, 4).to(device)

# Get the encoder part of the model
# Assuming your CTCSN model has an 'encoder' attribute
encoder = model.encoder.to(device)

# Profile only the encoder
flops, params = profile(encoder, (dummy_input,))
print("Encoder FLOPs and Parameters:")
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))