import time
from dataset import *
from tensorboardX import SummaryWriter
from trainOps import *
from model.CTCSN import CTCSN


# torch.backends.cudnn.benchmark=True
# Hyperparameters
batch_size = 16
device = 'cuda'  ## cpu or cuda (set cuda if gpu avaiilable)
VAL_HR = 256
INTERVAL = 4
WIDTH = 4
BANDS = 172
CR = 1  ## CR = 1, 5, 10, 15, 20
SIGMA = 0.0  ## Noise free -> SIGMA = 0.0
             ## Noise mode -> SIGMA > 0.0
TARGET = '27C'
SNR = 50

prefix = 'CTCSN'

# if not os.path.isdir('Rec'):
#     os.mkdir('Rec')

# ## Load test files from specified text with TARGET name (you can replace it with other path u want)
# testdata = loadTxt('testpath/val_%s.txt' % TARGET)

# ## Setup the dataloader
# val_loader = torch.utils.data.DataLoader(dataset_h5(testdata, mode='Validation', root=''), batch_size=batch_size, shuffle=False,
#                                          pin_memory=False)

model = CTCSN(cr=CR).to(device)

state_dict = torch.load('./ckpt/Train-C_allfig_allfig_cr_1_epoch_400.pth', map_location=device)

def count_parameters(model):
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model = model.module
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total_params, trainable_params = count_parameters(model)
print(f"\nModel Summary:")
print(f"Total parameters: {total_params/1e6:.2f}M")
print(f"Trainable parameters: {trainable_params/1e6:.2f}M")