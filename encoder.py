import torch
from torch import nn
import torch.nn.functional as F

from MultHeadAttention import MultHeadAttention
from TransformerEmbedding import TransformerEmbedding

class PositionWiseFeedForward(nn.Module):#定义前馈神经网络
    def __init__(self, d_model, hidden, dropout = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):#ffn是PositionWiseFeedForward（位置前馈网络）的简写 / 别名
    def __init__(self, d_model, ffn_hidden, n_head, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask = None):
        _x = x
        x = self.attention(x, x, x, mask)
        #在注意力机制中，Dropout并不是丢弃某个 “节点”，而是对注意力子层输出的特征张量进行逐元素随机置0操作。
        #一层层搭建
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class Encoder(nn.Module):#enc_voc_size编码器处理的目标序列语言 / 领域的词表,n_layer编码器层数
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, dropout, device, padding_idx=1):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(enc_voc_size, d_model, max_len, dropout, device, padding_idx)
        #循环n_layer次，每次创建一个结构完全相同的EncoderLayer层，
        #把所有层装进nn.ModuleList容器，赋值给self.layers，让 PyTorch 管理这些层的参数。
        '''
        这是 PyTorch 中堆叠多个网络层的标准写法，核心是用nn.ModuleList容器，配合 Python列表推导式，
        快速创建并管理n_layer个结构完全相同的EncoderLayer层，
        是 Transformer 编码器 / 解码器堆叠多层的工业界通用写法，比手动逐个定义层简洁百倍。
        '''
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, n_head, dropout)
                for _ in range(n_layer)
            ]
        )
    def forward(self, x, src_mask):
        x = self.embedding(x)# 词嵌入+位置编码，初始化特征
        #每一个 layer，都是一个独立的 EncoderLayer 对象。
        for layer in self.layers:# 逐层经过所有EncoderLayer
            x = layer(x, src_mask)
        return x