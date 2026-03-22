import os
import glob
import time
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import torch.nn as nn
import json
from sklearn.metrics import log_loss
import pdb
import random as rn
from model import *
from utils import *
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from dataset import *
from scipy.io import savemat, loadmat
from math import acos, degrees
from tensorboardX import SummaryWriter 
from trainOps import *


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch.backends.cudnn.benchmark=True
# Hyperparameters
batch_size = 64
device = 'cuda' ## cpu or cuda (set cuda if gpu avaiilable)
VAL_HR = 256
INTERVAL= 4
WIDTH=4
BANDS = 172
CR = 1        ## CR = 1, 5, 10, 15, 20
SIGMA = 0.0   ## Noise free -> SIGMA = 0.0
              ## Noise mode -> SIGMA > 0.0
TARGET = 'vis'
SNR = 99999



if not os.path.isdir('Rec'):
    os.mkdir('Rec')

## Load test files from specified text with TARGET name (you can replace it with other path u want)
valfn = loadTxt('./test_path/val_%s.txt' % TARGET)  

## Setup the dataloader
val_loader = torch.utils.data.DataLoader(dataset_h5(valfn, mode='Validation',root=''), batch_size=batch_size, shuffle=False, pin_memory=False, num_workers = 40)

model = SSANet(cr=CR)  
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

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# total_params = count_parameters(model.module if device == 'cuda' else model)
# print(f"\nTotal trainable parameters: {total_params:,}\n")

def count_encoder_parameters(model):
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model = model.module
    # 假设编码器通过model.encoder访问
    encoder = model.encoder
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    return total_params, trainable_params

# 调用示例
encoder_params, trainable_encoder_params = count_encoder_parameters(model)
print(f"\nEncoder Summary:")
print(f"Total encoder parameters: {encoder_params/1e6:.2f}M")
print(f"Trainable encoder parameters: {trainable_encoder_params/1e6:.2f}M")