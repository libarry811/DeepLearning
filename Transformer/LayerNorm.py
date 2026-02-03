import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-12):
        super(LayerNorm, self).__init__()
        # 可学习的缩放参数γ，初始化为全1，形状=(d_model,)，通过nn.Parameter设为可训练
        self.gamma = nn.Parameter(torch.ones(d_model))
        # 可学习的偏移参数β，初始化为全0，形状=(d_model,)，通过nn.Parameter设为可训练
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps# 赋值数值稳定项
    def forward(self, x):
        # 计算最后一维的均值，keepdim=True保持维度不变，避免广播维度不匹配
        # 如输入(b, t, d)，均值形状=(b, t, 1)，而非(b, t)
        mean = x.mean(-1, keepdim = True)
        # 计算最后一维的方差，unbiased=False使用有偏方差（样本方差，不除n-1），符合LayerNorm论文设计
        # keepdim=True保持维度与输入一致，方便后续计算
        var =x.var(-1, unbiased = False, keepdim = True)
        # 加eps避免方差为0时，平方根计算出现分母为0的错误
        out = (x - mean)/torch.sqrt(var + self.epd)
        # 缩放+偏移：通过可训练参数γ和β恢复特征表达能力，避免归一化后特征信息丢失
        # 逐元素相乘γ，再逐元素相加β，形状广播匹配
        out = self.gamma*out + self.beta
        return out
#official_layer_norm = nn.LayerNorm(d_model, eps=1e-12)