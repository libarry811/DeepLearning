import torch
from torch import nn
import torch.nn.functional as F
#torch.nn.functional 是 PyTorch 提供的函数式接口，包含了大量和神经网络相关的函数。
import math

from torch import Tensor
#将输入的词汇表索引转换为指定维度的Embedding
class TokenEmbedding(nn.Embedding):#将输入文本转为对应向量表示
    def __init__(self, vocab_size, d_model): #接收词汇表的大小和embedding的维度
        # 继承父类nn.Embedding，初始化嵌入层（词汇表大小vocab_size，嵌入维度d_model，padding索引为1）
        #行业惯例是 0 给<unk>（未知词）、1 给<pad>（填充符）
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
        #输出矩阵(batch_size, seq_len, d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEmbedding, self).__init__()#max_len 是指模型能处理的最长输入序列长度
        self.encoding = torch.zeros(max_len, d_model, device = device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device = device)
        pos = pos.float().unsqueeze(dim = 1)#在维度1的位置增加一个新维度从一维 (max_len,) 变成二维 (max_len, 1)
        #增加一个维度后(max_len, 1)的张量可以和 d_model,)的频率张量进行广播运算，得到(max_len, d_model)的位置编码矩阵。
        #偶数维度的索引可以表示为2i，奇数维度为2i+1，二者共享同一个i值计算频率；
        _2i = torch.arange(0, d_model, step = 2, device = device).float()
        #选取所有行，0::2：从索引 0开始，步长 2 → 选取所有偶数维度（0、2、4、6...）
        self.encoding[:,0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        #选取所有行，1::2：从索引 1开始，步长 2 → 选取所有奇数维度（1、3、5、7...）
        self.encoding[:,1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    def forward(self, x):
        batch_size, seq_len = x.size() #批量样本数，每个样本的序列长度，x形状(batch_size, seq_len)
        return self.encoding[:seq_len,:]#返回形状为(seq_len, d_model)的位置编码张量

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, device, drop_prob):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p = drop_prob)
    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)