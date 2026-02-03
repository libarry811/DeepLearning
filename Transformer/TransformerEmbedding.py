import torch
from torch import nn
import torch.nn.functional as F
#torch.nn.functional 是 PyTorch 提供的函数式接口，包含了大量和神经网络相关的函数。
import math

from torch import Tensor
#将输入的词汇表索引转换为指定维度的Embedding
class TokenEmbedding(nn.Embedding):#将输入文本转为对应向量表示
    # 【修改】新增 padding_idx 参数，默认值设为 1 (保持兼容性)，但允许修改
    def __init__(self, vocab_size, d_model, padding_idx=1):
        # 【修改】将传进来的 padding_idx 传给父类，而不是写死 1
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=padding_idx)
        #输出矩阵(batch_size, seq_len, d_model)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEmbedding, self).__init__()  # max_len 是指模型能处理的最长输入序列长度

        # 【修改】先创建一个临时的 tensor (这里叫 encoding，不是 self.encoding)
        encoding = torch.zeros(max_len, d_model, device=device)
        encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)  # 在维度1的位置增加一个新维度从一维 (max_len,) 变成二维 (max_len, 1)
        # 增加一个维度后(max_len, 1)的张量可以和 d_model,)的频率张量进行广播运算，得到(max_len, d_model)的位置编码矩阵。
        # 偶数维度的索引可以表示为2i，奇数维度为2i+1，二者共享同一个i值计算频率；
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 选取所有行，0::2：从索引 0开始，步长 2 → 选取所有偶数维度（0、2、4、6...）
        # 【修改】这里操作的是临时的 encoding 变量
        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        # 选取所有行，1::2：从索引 1开始，步长 2 → 选取所有奇数维度（1、3、5、7...）
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        # 【修改】核心改动：注册 buffer
        # 这一步会自动把 encoding 注册为 self.encoding，并让它跟随模型进行 device 移动
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        batch_size, seq_len = x.size()  # 批量样本数，每个样本的序列长度，x形状(batch_size, seq_len)
        # 注册后，依然通过 self.encoding 访问
        return self.encoding[:seq_len, :]  # 返回形状为(seq_len, d_model)的位置编码张量


class TransformerEmbedding(nn.Module):
    # 【修改】新增 padding_idx 参数
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device, padding_idx=1):
        super(TransformerEmbedding, self).__init__()
        # 【修改】把 padding_idx 传给 TokenEmbedding
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)