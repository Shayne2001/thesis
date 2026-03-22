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
import yaml
from thop import profile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch.backends.cudnn.benchmark=True
# Hyperparameters
# torch.cuda.set_device(1)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('config.yaml')

batch_size = config["batch_size"]
device = config["device"]
MAX_EP = config["MAX_EP"]  ## 最大迭代次数
VAL_HR = config["VAL_HR"]    ## 超分辨率图像的大小
INTERVAL = config["INTERVAL"]     ## 采样间隔
WIDTH = config["WIDTH"]         ## 采样宽度
BANDS = config["BANDS"]     ## 波段数
SIGMA = config["SIGMA"]     ## Noise free -> SIGMA = 0.0
                ## Noise mode -> SIGMA > 0.0
SOURCE = config["SOURCE"] ## 训练集输入图像路径
TARGET = config["TARGET"] ## 验证集输入图像路径
CR = config["CR"]         ## CR = 1, 5, 10, 15, 20  CompressionRatio 压缩率大小
prefix = config["prefix"]
SNR = config["SNR"]        ## 信噪比


# def model_metrics(model, input_shape = (172, 128, 4)):
#     # 参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     # FLOPs
#     input_tensor = torch.rand(240, *input_shape).to(device)
#     flops, _ = profile(model, inputs=(input_tensor, ))
#     # 内存占用
#     mem_alloc = torch.cuda.max_memory_allocated() / 1024**3

#     print("\n===== 模型指标 =====")
#     print(f"总参数量: {total_params} ({total_params/1e6:.2f}M)")
#     print(f"可训练参数: {trainable_params} ({trainable_params/1e6:.2f}M)")
#     print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
#     print(f"最大显存占用: {mem_alloc:.2f} GB")


def trainer():

    model = SSANet(snr=0, cr=CR, bands=BANDS)   ## cr =[1, 5, 10, 15, 20] compression ratio
    model = torch.nn.DataParallel(model).to(device) ## GPU并行计算

    state_dict = torch.load('./ckpt/35_247.pth')
    model.load_state_dict(state_dict)



if __name__ == '__main__':
    trainer()


