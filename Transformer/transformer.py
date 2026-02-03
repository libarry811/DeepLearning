import torch
from torch import nn

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,#编码器输入中 padding 填充 token 的索引，用于构建 s_mask 屏蔽编码器自注意力、解码器交叉注意力中的无效 padding 信息
                 trg_pad_idx,#解码器输入中 padding 填充 token 的索引，用于构建 t_mask 屏蔽解码器自注意力中的无效 padding 信息
                 enc_voc_size,
                 dec_voc_size,
                 d_model,
                 max_len,
                 n_heads,
                 ffn_hidden,
                 n_layers,
                 device,
                 drop_prob
                 ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_heads, n_layers, device, drop_prob)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_heads, n_layers, device, drop_prob)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    '''
    q：需要生成掩码的查询序列（token ID 序列），用于确定掩码的查询维度长度
    k：需要生成掩码的键序列（token ID 序列），用于确定掩码的键维度长度
    pad_idx_q：查询序列 q 中 padding 填充 token 的索引，标记 q 里的无效填充位
    pad_idx_k：键序列 k 中 padding 填充 token 的索引，标记 k 里的无效填充位
    '''
    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):#填充掩码
        # 获取查询序列q、键序列k的序列长度（batch, seq_len）中第二个维度的尺寸
        len_q,len_k = q.size(1),k.size(1)#size(1)就是取序列长度维度
        # q.ne(pad_idx_q)：q中token不等于pad_idx_q则为True(有效)，等于则为False(填充位)，形状(batch, len_q)
        # unsqueeze(1).unsqueeze(3)：维度扩充为(batch, 1, len_q, 1)，为后续广播做维度匹配
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        # 按k的长度广播，形状变为(batch, 1, len_q, len_k)，标记q序列各位置是否为有效token
        q = q.repeat(1, 1, 1, len_k)
        # k.ne(pad_idx_k)：k中token不等于pad_idx_q则为True(有效)，等于则为False(填充位)，形状(batch, len_k)
        # unsqueeze(1).unsqueeze(2)：维度扩充为(batch, 1, 1, len_k)，为后续广播做维度匹配
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        # 按q的长度广播，形状变为(batch, 1, len_q, len_k)，标记k序列各位置是否为有效token
        k = k.repeat(1, 1, len_q, 1)
        # 按元素与操作：仅当q和k对应位置均为有效token时，掩码值为True，否则为False(需屏蔽)
        # 最终掩码形状(batch, 1, len_q, len_k)，适配多头注意力的掩码输入维度
        '''
        这个中间的 1 是为了适配 Transformer 多头注意力的维度设计，做的维度对齐占位，
        让掩码的形状能和多头注意力的 QKV 特征张量形状完美匹配，支持多头维度的广播计算，不会因维度不匹配报形状错误。
        输入的 Q/K/V 特征张量形状是 (batch, n_heads, seq_len, d_k),d_k 是单头特征维度
        '''
        mask = q&k
        return mask
    def make_casual_mask(self, q, k):#因果掩码，防止未来信息泄露
        len_q, len_k = q.size(1), k.size(1)  # size(1)就是取序列长度维度
        # torch.ones(len_q, len_k)：创建len_q行len_k列的全1矩阵，代表注意力矩阵的原始形状
        # torch.tril()：提取矩阵的下三角部分（含对角线），上三角置0，实现“只能关注当前/过去、屏蔽未来”的因果逻辑
        # .type(torch.BoolTensor)：转换为布尔类型，True表示有效位置，False表示需屏蔽的未来位置
        # .to(self.device)：将掩码移至模型所在设备（GPU/CPU），避免张量设备不匹配报错
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask# 返回因果掩码，形状为(len_q, len_k)，后续会自动广播适配batch和多头维度
    def forward(self, src, trg):
        # 构建编码器源序列的填充掩码，q/k均为源序列src，屏蔽src中的padding无效位
        # 最终形状(batch, 1, src_len, src_len)，仅保留源序列自身有效token的注意力交互
        src_mask = self.make_pad_mask(src, src,self.src_pad_idx, self.src_pad_idx)
        # 构建解码器目标序列的最终掩码：因果掩码 * 填充掩码，同时实现两大核心屏蔽逻辑
        # 因果掩码：屏蔽未来token；填充掩码：屏蔽padding无效位；相乘实现“双重过滤”
        '''相乘的逻辑效果：只有当两个掩码对应位置同时为 1（True） 时，相乘结果才为 1（有效）；只要其中一个掩码位置为 0（False），相乘结果就为 0（屏蔽）。
            比如：某位置是未来 token（因果掩码为 0） → 不管是否是 padding，相乘后为 0（屏蔽）；
            比如：某位置是padding（填充掩码为 0） → 不管是否是历史 token，相乘后为 0（屏蔽）；
            只有历史有效 token（因果掩码 1 + 填充掩码 1） → 相乘后为 1（允许注意力交互）。
            因果掩码的唯一作用是屏蔽未来 token，它只解决「解码器不能看未来」的问题，但完全不知道哪些位置是 padding 填充的无效位：
            因果掩码是基于序列长度生成的固定下三角矩阵，只会根据 token 的位置（前 / 后）做屏蔽，不会区分 token 是有效内容还是填充的 0（或其他 pad_idx）；
            比如目标序列是[我, 爱, 你, <PAD>, <PAD>]，因果掩码只会保证看第 1 个 token 时不看后面，看第 2 个时不看第 3/4/5 个，
            但会让模型去关注第 4、5 个<PAD>位，相当于让模型学习 “无意义的填充内容”，反而干扰有效特征的学习。
            理论上解码器确实是逐 token 自回归生成的（一次只输出一个有效 token，不会出 PAD），
            但训练阶段为了效率，用的是「批量并行训练」（Teacher Forcing），
            这时候解码器的输入是带 PAD 的完整目标序列，这就是因果掩码会 “看到” PAD 的根本原因。'''
        trg_mask = self.make_casual_mask(trg, trg)*self.make_pad_mask(trg, trg,self.trg_pad_idx, self.trg_pad_idx)
        # 将源序列src和源序列填充掩码src_mask传入编码器，得到编码器对源序列的最终特征表示enc
        # enc是解码器交叉注意力层的核心参考特征，形状为(batch, src_len, d_model)
        enc = self.encoder(src, src_mask)
        # 将目标序列trg、编码器输出enc、目标序列最终掩码trg_mask、源序列填充掩码src_mask传入解码器
        # 解码器先学习自身历史特征，再通过交叉注意力融合enc的源序列信息，最终输出词表维度的预测特征out
        out = self.decoder(trg, enc, trg_mask, src_mask)
        return out

'''
假设src_pad_idx=0（padding 填充的索引为 0）
源序列src是形状为(batch=2, src_len=4)的 token ID 张量（2 个样本，每个序列最大长度 4）：
# src：batch=2，src_len=4，src_pad_idx=0
# 样本1：有效token是[5,8,3]，最后1位补0；样本2：有效token是[2,7]，最后2位补0
src = torch.tensor([[5,8,3,0], 
                    [2,7,0,0]])
调用src_mask = self.make_pad_mask(src, src, 0, 0)生成源序列掩码，
最终src_mask形状为(2, 1, 4, 4)->（batch, 多头占位 1, src_len, src_len）。
src_mask 具体样子（逐样本拆解，注释版）
# src_mask：形状(2,1,4,4)，bool类型→True(1)=有效位（允许注意力），False(0)=屏蔽位（禁止注意力）
src_mask = torch.tensor([
    # 样本1的掩码：(1,4,4)，有效token是前3位[5,8,3]，第4位是padding(0)
    [[[True,  True,  True,  False],  # 第1个token(5)：可关注1/2/3位，屏蔽4位
      [True,  True,  True,  False],  # 第2个token(8)：可关注1/2/3位，屏蔽4位
      [True,  True,  True,  False],  # 第3个token(3)：可关注1/2/3位，屏蔽4位
      [False, False, False, False]]],# 第4个token(0)：自身是padding，所有位置都屏蔽
    
    # 样本2的掩码：(1,4,4)，有效token是前2位[2,7]，第3/4位是padding(0)
    [[[True,  True,  False, False],  # 第1个token(2)：可关注1/2位，屏蔽3/4位
      [True,  True,  False, False],  # 第2个token(7)：可关注1/2位，屏蔽3/4位
      [False, False, False, False],  # 第3个token(0)：自身是padding，所有位置都屏蔽
      [False, False, False, False]]] # 第4个token(0)：自身是padding，所有位置都屏蔽
])
'''