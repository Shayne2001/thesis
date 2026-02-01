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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    ## Reading files #
    
    ## Load test files from specified text with SOURCE/TARGET name (you can replace it with other path u want)
    flist = loadTxt('./train_path/train_%s.txt' % SOURCE) ## 包含训练集的路径列表
    valfn = loadTxt('./test_path/val_%s.txt' % TARGET)   ## 包含验证集的路径列表
    tlen = len(flist)

    ## DataLoader返回一个迭代器，该迭代器将返回一个batch_size大小的数据
    train_loader = torch.utils.data.DataLoader(dataset_h5(flist, width=4, marginal=60, root=''), batch_size=batch_size, shuffle=True, num_workers=128)
    val_loader = torch.utils.data.DataLoader(dataset_h5(valfn,mode = 'Validation', root=''), batch_size=5, shuffle=False, pin_memory=False)

    model = SSANet(snr=0, cr=CR, bands=BANDS)   ## cr =[1, 5, 10, 15, 20] compression ratio
    model = torch.nn.DataParallel(model).to(device) ## GPU并行计算
    
    class SSALoss(nn.Module):
        def __init__(self, alaph=0.2, beta=0.1):
           super().__init__()
           self.mse = nn.MSELoss()
           self.alpa = alaph
           self.beta = beta
        def forward(self, pred, target, gates):
            loss_recon = self.mse(pred, target)
            loss_sparsity = sum([torch.mean(torch.abs(g - 0.5)) for g in gates])
            prune_gates = [1 - torch.mean(g) for g in gates]
            loss_prune = sum([torch.relu(pr - 0.6) for pr in prune_gates])
            return loss_recon + self.alpa * loss_sparsity + self.beta * loss_prune

    ## 微调
    ## state_dict = torch.load('./checkpoint/DCSN_130fig_130fig_cr_1_epoch_770_2.425.pth')
    ## model.load_state_dict(state_dict)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)  ## finetune lr=0.00003 学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') ## 学习率调整策略 最小化验证集上的损失
    L1Loss = torch.nn.L1Loss() ##采用L1Loss作为损失函数
    loss_fn = SSALoss()

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('Rec'):
        os.mkdir('Rec')    
    writer = SummaryWriter('log/%s_exp2_%s_%s_%s' % (prefix, SOURCE, TARGET, CR)) ##训练信息通过Tensorboard可视化
    
    resume_ind = 0
    step = resume_ind
    best_sam = 99

    for epoch in range(resume_ind, MAX_EP): 
        ep_loss = 0.   ##每个Epoch的总损失
        for batch_idx, (x,_) in enumerate(train_loader):
            running_loss=0.  ## 每个batch的损失
            
            optimizer.zero_grad()  ##每次迭代之前清除模型的梯度
            ## x.shape = (4, 60, 128, 4, 172)
            x = x.view(x.size()[0]*x.size()[1], x.size()[2], x.size()[3], x.size()[4])
            ## x.view改变张量的形状  这里将第一个维度大小和第二个维度合并,其余维度保持不变
            x = x.to(device).permute(0,3,1,2).float() ## 重新排列张量的维度  x.shape = (240, 172, 128, 4)
            # print("x: ", x.shape)
            decoded, _ = model(x)      ## 执行模型，返回解码后的结果
            # gates = [model.gate1.scores, model.gete2.scores]
            loss = L1Loss(decoded, x)  ## 计算L1Loss
            loss.backward()            ## 反向传播
            
            optimizer.step()           ## 更新参数
            running_loss  +=  loss.item()

        ## 每10个epoch进行一次验证    
        if epoch% 10 ==0 :
            with torch.no_grad():  ## 不进行梯度更新
                rmses, sams, fnames, psnrs = [], [], [], []
                start_time = time.time()
                for ind2, (vx, vfn) in enumerate(val_loader):
                    # print("vfn: ", len(vfn))
                    # print(vfn)
                    model.eval()   ## 模型评估
                    ## vx.shape = torch.Size([256, 172, 256, 4])
                    # print("vx: ", vx.shape)
                    vx = vx.view(vx.size()[0]*vx.size()[1], vx.size()[2], vx.size()[3], vx.size()[4])
                    ## vx.shape = torch.Size([256, 172, 256, 4])
                    vx= vx.to(device).permute(0,3,1,2).float()
                    if SIGMA>0:
                        val_dec = model(awgn(model(vx, mode=1), SNR), mode=2)
                    else:
                        val_dec,_ = model(vx)  ## shape = (256, 256, 4, 172)
    
                    
                    ## Recovery to image HSI
                    val_batch_size = len(vfn)
                    ## 
                    img = [np.zeros((VAL_HR, VAL_HR, BANDS)) for _ in range(val_batch_size)]
                    val_dec = val_dec.permute(0,2,3,1).cpu().numpy()
                    cnt = 0
                    
                    for bt in range(val_batch_size):
                        for z in range(0, VAL_HR, INTERVAL):
                            ## 将解码器的输出 val_dec 中的第 cnt 个图像赋值给 img 数组中对应批次 bt 的相应行
                            # print("img: ", img[bt].shape, "val: ", val_dec[cnt].shape)
                            img[bt][:,z:z+WIDTH,:] = val_dec[cnt]
                            cnt +=1
                        save_path = vfn[bt].split('/')
                        save_path = save_path[-2] + '-' + save_path[-1]
                        np.save('Rec/%s.npy' % (save_path), img[bt])
                        
                        ## lmat 函数用于加载真实的高分辨率图像，并将其转换为 float64 类型
                        GT = lmat(vfn[bt]).astype(np.float64)
                        ## 去除 GT 中的零值
                        for i in range(GT.shape[0]):
                            for j in range(GT.shape[1]):
                                    k = 0
                                    while(GT[i][j+k].sum()==0):
                                        k+=1
                                    GT[i][j] = GT[i][j+k]
                        ## 计算真实图像 GT 的最大值和最小值
                        maxv, minv=np.max(GT), np.min(GT)
                        ## 对模型输出的图像进行去归一化处理，使其值域与真实图像相同
                        img[bt] = img[bt]*(maxv-minv) + minv ## De-normalization
                        sams.append(sam(GT, img[bt]))
                        rmses.append(rmse(GT, img[bt]))
                        fnames.append(save_path)
                        psnrs.append(psnr(GT, img[bt]))
                
                ep = time.time()-start_time
                ## 计算处理单个验证样本所需的平均时间，len(sams) 是验证样本的数量。
                ep = ep / len(psnrs)
                plog('[epoch: %d, batch: %5d] loss: %.3f, , val-RMSE: %.3f, val-SAM: %.3f, val-PSNR: %.3f, AVG-Time: %.3f' %
                      (epoch, batch_idx+resume_ind, running_loss, np.mean(rmses), np.mean(sams), np.mean(psnrs), ep)
                      , prefix, SOURCE, TARGET, CR)
                ## 将验证集上的 RMSE 和 SAM 通过 Tensorboard 可视化
                writer.add_scalar('Validation RMSE', np.mean(rmses), step)
                writer.add_scalar('Validation SAM', np.mean(sams), step)
                ## 基于验证集上的平均 SAM 更新学习率
                scheduler.step(np.mean(sams))
                
                with open('log/validataion_%s_%s_%d.txt' % (SOURCE, TARGET, CR),'w') as fp:
                    ## 增加了对 PSNR 的记录
                    for r, s, f, p in zip(rmses, sams, fnames, psnrs):
                        fp.write("%s:\tRMSE:%.4f\tSAM:%.3f\tPSNR:%.3f\n" % (f, r, s, p))
                
            if best_sam > np.mean(sams):
                best_sam = np.mean(sams)
                torch.save(model.state_dict(), 'checkpoint/%s_%s_%s_cr_%d_epoch_%d_%.3f.pth' %(prefix, SOURCE, TARGET, CR, epoch, np.mean(sams)))
                
            ep_loss += running_loss
            writer.add_scalar('Running loss', running_loss, step)
                
            running_loss = 0.0
            # running_loss2 = 0.0
            model.train() 
        
        # if epoch < 600:
        #     model.gate1.requires_grad_(False)
        #     model.gate2.requires_grad_(False)
        # elif 600 <= epoch < 1600:
        #     model.gate1.requires_grad_(True)
        #     model.gate2.requires_grad_(True)
        #     model.gate1.temperature = max(0.1, 1.0 - 0.0005 * (epoch - 600))
        # else:
        #     model.gate1.eval()
        #     model.gate2.eval()
                
        if epoch% 50 ==0 and epoch > 1:
            torch.save(model.state_dict(), 'checkpoint/%s_%s_%s_cr_%d_epoch_%d.pth' %(prefix, SOURCE, TARGET, CR, epoch))
        
        step+=1
        
    # print("===== 训练后初始指标 =====")
    # model_metrics(model)


if __name__ == '__main__':
    trainer()


