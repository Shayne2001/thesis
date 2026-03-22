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

with open('data.pkl', 'rb') as file:
    c_a = pickle.load(file)

# 初始化一个列表来存储每个子列表的平均值
averages = []
merge = []
for sublist in c_a:
    # 沿着第0个维度合并子列表中的Tensor
    merged_tensor = torch.cat(sublist, dim=0)
    # 将合并后的Tensor添加到merged_tensors列表中
    merge.append(merged_tensor)

for tensor in merge:
    # 沿着维度 0 计算平均值
    average_tensor = torch.mean(tensor, dim=0)
    # 将平均值 Tensor 添加到 averages 列表中
    averages.append(average_tensor)


# 输出每个子列表的平均值
for i in averages:
    print(i)