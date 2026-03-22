
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
import json
import pickle
from thop import profile
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
TARGET = 'allfig'
SNR = 99999

if not os.path.isdir('Rec'):
    os.mkdir('Rec')

## Load test files from specified text with TARGET name (you can replace it with other path u want)
valfn = loadTxt('./test_path/val_%s.txt' % TARGET)  

## Setup the dataloader
val_loader = torch.utils.data.DataLoader(dataset_h5(valfn, mode='Validation',root=''), batch_size=batch_size, shuffle=False, pin_memory=False, num_workers = 40)

model = SSANet(cr=CR)  
state_dict = torch.load('ckpt/35_247.pth' ,map_location=device)

model = torch.nn.DataParallel(model).to(device) ## GPU并行计算
total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(f"参数量: {total_params / 1e6:.3f}M")
input_tensor = torch.randn(240, 172, 128, 4)  # 替换为合适的输入尺寸
input_tensor = input_tensor.to(device)
flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
print(f"FLOPs: {flops / 1e9:.3f}G")

# model.eval()
# with torch.no_grad():
#     rmses, sams, fnames, psnrs = [], [], [], []
#     c_a = [[],[],[]]
#     start_time = time.time()
#     for ind2, (vx, vfn) in enumerate(val_loader):
#         model.eval()
#         vx = vx.view(vx.size()[0]*vx.size()[1], vx.size()[2], vx.size()[3], vx.size()[4])
#         vx= vx.to(device).permute(0,3,1,2).float()
#         if SIGMA>0:
#             val_dec = model(awgn(model(vx, mode=1), SNR), mode=2)
            
#         else:
#             val_dec,_, c_a_l = model(vx)
#             c_a[0].append(c_a_l[0])
#             c_a[1].append(c_a_l[1])
#             c_a[2].append(c_a_l[2])
#     with open('data.pkl', 'wb') as file:
#         pickle.dump(c_a, file)

#----------------------------------------------------------------------------
    # c_a_1 = []
    # for i in c_a:
    #     i = [t.tolist() for t in i]
    #     c_a_1.append(i)
    # # 将列表存储到文件
    # with open('data.json', 'w') as file:
    #     json.dump(c_a_1, file)

    # 读取存储的列表
    # with open('data.json', 'r') as file:
    #     loaded_c_a = json.load(file)



