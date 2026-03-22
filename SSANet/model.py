import torch
from torch import nn
import numpy as np, math
from torch.nn import functional as F
from torch.autograd import Variable
import functools
from module_util import *
from MMF import MMF


import pdb
from typing import Optional

from einops.layers.torch import Rearrange



class MRDF(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(MRDF, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        ## 64 --> 32
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        ## 64+32=96 --> 32
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        ## 64+32+32=128 --> 32
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        ## 64+32+32+32=160 --> 32
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        ## 64+32+32+32+32=192 --> 64
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        ## 第1层到第4层的通道数都为32，第5层的通道数为64
        # print("x: ", x.shape)
        x1 = self.lrelu(self.conv1(x))
        # print("x1: ", x1.shape)
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))         ## 在通道维度上进行拼接
        # print("x2: ", x2.shape)
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # print("x3: ", x3.shape)
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # print("x4: ", x4.shape)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # print("x5: ", x5.shape)

        return x5 * 0.2 + x                                        ## 输入特征图和最后一层的输出进行残差连接

class SSFE(nn.Module):

    ## nf：特征数量  gc：增长通道数
    def __init__(self, nf, gc=32):
        super(SSFE, self).__init__()
        self.MRDF1 = MRDF(nf, gc)
        self.MRDF2 = MRDF(nf, gc)
        self.MRDF3 = MRDF(nf, gc)
        ## MMF: 空间特征和光谱特征的自适应融合
        self.MMF = MMF(nf, nf)
        # self.MSCA = MultiSpectralAttentionLayer(128, 64, 2)
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)

    def forward(self, x):
        # print("before MRDF1: ", x.shape)
        out = self.MRDF1(x)
        # print("before MRDF2", out.shape)
        out = self.MRDF2(out)
        out = out * 0.2 + x    ## 输入特征图和MFB2的输出图进行残差连接
        # print("before MRDF3", out.shape)
        out = self.MRDF3(out)
        # print("before MMF", out.shape)
        out, _ = self.MMF(out, out, out)
        # print("before return", out.shape)
        alpha = torch.sigmoid(self.gate1)
        return out * alpha + x * (1 - alpha)   ## 输入特征图和MFB3的输出图进行残差连接

class DynamicGate(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        super().__init__()
        # 通道重要性评估网络
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels),
            nn.Sigmoid()  # 输出门控概率
        )
        self.temperature = 1.0  # Gumbel-Softmax温度参数

    def forward(self, x):
        b, c, h, w = x.shape
        # 计算通道重要性得分 [B, C]
        scores = self.fc(self.avg_pool(x).view(b, c))
        
        # 训练时：Gumbel-Softmax松弛
        if self.training:
            noise = torch.rand_like(scores)
            gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
            gates = torch.sigmoid((scores + gumbel_noise) / self.temperature)
        # 推理时：硬剪枝（保留Top-K通道）
        else:
            threshold = torch.quantile(scores, 0.6)  # 保留前40%通道
            gates = (scores >= threshold).float()
        
        return x * gates.view(b, c, 1, 1)

#Adaptive Global Channel Attention
class AGCA(nn.Module):
    def __init__(self, in_channel, ratio):
        super(AGCA, self).__init__()
        # 隐藏层通道数
        hide_channel = in_channel // ratio
        # 平均池化 获取通道的全局信息 Global
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)
        # Choose to deploy A0 on GPU or CPU according to your needs
        # A0是一个固定矩阵，用于保持初始通道关系
        self.A0 = torch.eye(hide_channel).to('cuda')
        # self.A0 = torch.eye(hide_channel)
        # A2 is initialized to 1e-6 通过训练调整通道间关系
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]、
        y = self.avg_pool(x)
        # y: [B, C, 1, 1]
        y = self.conv1(y)
        # y: [B, hide_channel, 1, 1]
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)
        # 将特征展平、转置
        # y: [B, 1, hide_channel]
        A1 = self.softmax(self.conv2(y))
        # 生成基础注意力权重
        A1 = A1.expand(B, C, C)
        # A1: [B, hide_channel, hide_channel]
        A = (self.A0 * A1) + self.A2
        y = torch.matmul(y, A)  # 调整通道间关系
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))

        return x * y

class LEncoder(nn.Module):
    def __init__(self, in_channels, last_ch, last_kernel_w, last_stride, last_padding_w, bit_num):
        super(LEncoder, self).__init__()
        # self.HWD1 = Down_wt(in_channels, 128)
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=[3, 2], stride=[2, 2], padding=[1, 0])
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        # self.HWD2 = Down_wt(128, last_ch)
        # self.MSCA1 = MultiSpectralAttentionLayer(128, 64, 2)
        self.MMF1 = MMF(128, 128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=[3, 1], stride=[1, 1], padding=[1, 0])
        self.gate2 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        # self.MSCA2 = MultiSpectralAttentionLayer(64, 64, 2)
        self.MMF2 = MMF(64, 64)
        # self.HWD3 = Down_wt(64, last_ch)
        self.conv3 = nn.Conv2d(64, last_ch, kernel_size=[3, last_kernel_w], stride=[last_stride, 2], padding=[1, last_padding_w])
        self.gate3 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        # self.MSCA3 = MultiSpectralAttentionLayer(last_ch, 32, 1)
        self.MMF3 = MMF(last_ch, last_ch)
        # self.AGCA = AGCA(64, 4).to('cuda')
        # self.Fca = torch.hub.load('cfzd/FcaNet', 'fca152' ,pretrained=True)
        self.relu = nn.LeakyReLU(True)

        # 动态剪枝模块
        # self.gate1 = DynamicGate(num_channels=128)
        # self.gate2 = DynamicGate(num_channels=64)


    def forward(self, x):
        # print("before conv1: ")
        # print(x.shape)
        # 第一层
        # x = self.HWD1(x)

        x = self.conv1(x)
        # print("before conv2: ")
        # print(x.shape)
        # x = self.gate1(x)
        x1 = self.relu(x)
        x2, c_a1 = self.MMF1(x1, x1, x1)
        alpha = torch.sigmoid(self.gate1)
        x = alpha * x1 + (1 - alpha) * x2
        # 第二层
        # x = self.HWD2(x)
        
        x = self.conv2(x)
        # x = self.gate2(x)
        # x3 [2160, 64, 64, 2]
        x3 = self.relu(x)
        # print("x3: ", x3.shape)
        x4, c_a2 = self.MMF2(x3, x3, x3)
        beta = torch.sigmoid(self.gate2)
        x = beta * x3 + (1 - beta) * x4
        # x = self.AGCA(x)
        # print("before conv3: ")
        # print(x.shape)
        # 第三层
        # x = self.HWD3(x)
        # x5 [2160, 27, 32, 1]
        x5 = self.relu(self.conv3(x))
        # print("x5: ", x5.shape)
        x6, c_a3 = self.MMF3(x5, x5, x5)
        gamma = torch.sigmoid(self.gate3)
        x = gamma * x5 + (1 - gamma) * x6
        # print("after encoder: ")
        # print(x.shape)
        return x, [c_a1, c_a2, c_a3]

class SSDecoder(nn.Module):
    ## 创建一个由多个相同类型的层组成的序列层
    def make_layer(block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)
    
    ## out_nc = 172 nf = 64 nb = 16
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, up_scale = 4):
        super(SSDecoder, self).__init__()
        SSFE_block = functools.partial(SSFE, nf=nf, gc=gc)
        self.up_scale = up_scale
        
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        ## nb:MFA块数量 这里堆叠了16个MFA块
        self.SSFE_trunk = make_layer(SSFE_block, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        ## upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.EUCB1 = EUCB(in_channels=64, out_channels=64)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.EUCB2 = EUCB(in_channels=64, out_channels=172)
        self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.EUCB3 = EUCB(in_channels=64, out_channels=172)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 1, 1, 0, bias=True)
        # self.FMB = FMB(dim = 172)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # print("before conv_first: ")
        # print(x.shape)
        # [960, 27, 32, 1]
        fea = self.conv_first(x)
        # print("before trunk_conv: ")
        # print(fea.shape)
        # [960, 64, 32, 1]
        trunk = self.trunk_conv(self.SSFE_trunk(fea))
        # print("trunk: ", trunk.shape)
        alpha = torch.sigmoid(self.gate1)
        fea = alpha * fea + (1 - alpha) * trunk

        ## fea 通过 upconv1 卷积层，并通过 F.interpolate 函数进行最近邻上采样，上采样比例为2倍
        # print("before upsample1: ")
        # print(fea.shape)
        # [960, 64, 32, 1]
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # fea = self.EUCB1(fea)
        ## 如果 up_scale 属性为4，说明需要进一步上采样
        if self.up_scale == 4:
            ## fea 再次通过 upconv2 卷积层，然后进行2倍上采样
            # print("before upsample2: ")
            # print(fea.shape)
            # [960, 64, 64, 2]
            fea = self.lrelu(self.upconv2(F.interpolate(fea,  scale_factor=2, mode='nearest')))
            # fea = self.EUCB2(fea)
        # print("before conv_last: ")
        # print(fea.shape)
        # [960, 64, 128, 4]
        fea = self.conv_last(self.lrelu(self.HRconv(fea)))
        # fea = self.FMB(fea)
        # print("after conv_last: ")
        # print(fea.shape)
        # [960, 172, 128, 4]
        # print("after decoder: ", fea.shape)
        return fea

##=========full module==========
class SSANet(nn.Module): 
    def __init__(self, snr=0, cr=1 , bands=172, bit_num = 8):
        super(SSANet, self).__init__()    ## 初始化DCSN类
        self.snr = snr
        self.bands = bands
        if cr == 1:
            last_stride = 2             ## 这里定义的都是最后一个卷积层的参数 步长为2
            last_ch = int(self.bands//6.25)                ## 通道数为27 满足压缩比为1%
            last_kernel_w = 1           ## 卷积核宽度为1
            last_padding_w = 0          ## 填充宽度为0 即不使用填充
        else:
            last_stride = 1 
            last_kernel_w = 2
            last_padding_w = 1
            
        up_scale = 4 if cr<5 else 2     ## 如果压缩比小于5，上采样比例为4，否则为2
        if cr==5:                       ## 根据压缩比确定最后一个卷积层的通道数 见论文的TABLE 2
            last_ch = int(self.bands//5)
        elif cr==10:
            last_ch = int(self.bands//2.5)
        elif cr==15:
            last_ch = int(self.bands//1.67)
        elif cr==20:
            last_ch = int(self.bands//1.25)
        
        if last_ch % 2 == 1:
            last_ch += 1
        self.encoder = LEncoder(self.bands, last_ch, last_kernel_w, last_stride, last_padding_w, bit_num = bit_num)
        # print(self.encoder)
        self.decoder = SSDecoder(last_ch, self.bands, 64, 16, up_scale=up_scale)
        ##  128*4*172=88064 --> 32*1*27 --> cr=1%
        ##  64*2*32 --> cr=4.65%
        ##  64*2*64 --> cr=9.30%
        ##  64*2*103 -->cr=14.97%
        ##  64*2*140 -->cr=20.3%

        ## 构建编码器
        # self.encoder = nn.Sequential(
        #     ## 第一个卷积层 输入通道数为172，输出通道数为128，卷积核大小为3*3，步长为2，水平方向填充为1
        #     nn.Conv2d(self.bands, 128, [3, 3], stride=[2,2], padding=[1,0]),  # b, 16, 10, 10
        #     nn.LeakyReLU(True),
        #     ## 第二个卷积层 输入通道数为128，输出通道数为64，卷积核大小为3*1，步长为1，水平方向填充为1
        #     nn.Conv2d(128, 64, [3,1], stride=[1,1], padding=[1,0]),  # b, 8, 3, 3
        #     nn.LeakyReLU(True),
        #     AGCA(64, 4).to('cuda'),
        #     ## 第三个卷积层 输入通道数为64，其余参数由压缩比决定
        #     nn.Conv2d(64, last_ch, [3,last_kernel_w], stride=[last_stride, 1], padding=[1, last_padding_w])
        # )
        
    
    ## 加入高斯噪声
    def awgn(self, x, snr):  
        snr = 10**(snr/10.0)  ## 将分贝单位的信噪比转化为线性单位
        xpower = torch.sum(x**2)/x.numel() ## 计算输入信号的平均功率
        npower = torch.sqrt(xpower / snr)  ## 计算噪声的平均功率
        return x + torch.randn(x.shape).cuda() * npower  ## 生成与 x 形状相同的高斯白噪声，并将其添加到原始信号上


    def forward(self, data, mode=0): ### Mode=0, default, mode=1: encode only, mode=2: decoded only
        
        if mode==0:
            x, c_a_l = self.encoder(data)
            if self.snr > 0:
                x = self.awgn(x, self.snr)
            y = self.decoder(x)
            return y, x, c_a_l
        elif mode==1:
            x, c_a_l = self.encoder(data)
            return x, c_a_l
        elif mode==2:
            return self.decoder(data)
        else:
            return self.decoder(self.encoder(data))
