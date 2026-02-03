import torch
from torch import nn
import torch.nn.functional as F

from MultHeadAttention import MultHeadAttention
from TransformerEmbedding import TransformerEmbedding
from encoder import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultHeadAttention(d_model, n_head)#掩码多头注意力
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attention = MultHeadAttention(d_model, n_head)#解码器的交叉注意力层
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    '''
    dec：解码器输入的目标序列,解码器之前所有时刻的输入
    enc：编码器对源序列的最终特征输出，供解码器交叉注意力层参考学习
    t_mask：目标序列掩码，用于解码器自注意力层，屏蔽未来 token 和 padding 无效 token
    s_mask：源序列掩码，复用自编码器，用于解码器交叉注意力层，屏蔽源序列的 padding 无效 token
    '''
    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.attention1(dec, dec, dec, t_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.cross_attention(x, enc, enc, s_mask)#q由掩码多头注意力产生
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x

class Decoder(nn.Module):#dec_voc_size解码器处理的目标序列语言 / 领域的词表
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, device, dropout):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, device, dropout)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, ffn_hidden, n_head, dropout)
                for _ in range(n_layer)
            ]
        )
        self.fc = nn.Linear(d_model, dec_voc_size)#全连接层
    def forward(self, dec,enc,  t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:# 逐层经过所有DecoderLayer
            dec = layer(dec, enc, t_mask, s_mask)
        dec = self.fc(dec)
        return dec
