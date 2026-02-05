import torch
from torch import nn
import math

#x = torch.rand(128, 32, 512)#batch_size,seq_len,d_model
#d_model = 512
#n_head = 8

class MultHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        #我们需要把 d_model 维的输入映射到 n_head × d_k 维（其中 d_k = d_model // n_head）
        #所以输出特征数是 d_model（因为 n_head × d_k = d_model）
        #d_k 是多头注意力中每个注意力头的维度
        #nn.Linear(d_model, d_model) 保证输入输出维度一致，为后续切分多头做准备。
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        #把 8 个独立头合并后的特征做非线性融合，让模型学到多头特征之间的关联，而不是单纯的拼接堆叠
        self.w_combine = nn.Linear(d_model, d_model)
        #注意力分数张量的最后一维，刚好对应「当前查询 token 对所有键 token 的分数集合」
        #倒数第二维 “有多少个查询”，最后一维管“每个查询对应多少个分数”
        self.softmax = nn.Softmax(dim=-1)#工程固定写法
    def forward(self, q, k, v, mask = None):
        #这两行是为后续切分多头做准备
        batch, time, dimension = q.shape # 解包输入q的维度，分别赋值给批次、序列长度、嵌入维度,和这三个是同一个东西batch_size,seq_len,d_model
        n_d = self.d_model//self.n_head  #计算每个注意力头的维度和之前的d_k是同一个东西）
        #线性层权重矩阵W：形状dmodel×dmodel→ 对应理论中的WQ/WK/WV投影矩阵
        # 【核心修改点】分别获取 q, k, v 的序列长度
        # 在 Cross Attention 中，len_q (目标序列长) 和 len_k/len_v (源序列长) 是不一样的
        len_q = q.size(1)
        len_k = k.size(1)
        len_v = v.size(1)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        #多头注意力里切分多头的核心维度变换操作，用view方法把原本整体的 Q 张量
        #在嵌入维度上拆分成n_head个独立的子空间，为后续多注意力头并行计算做准备
        #dimension=n_head*n_d,q:batch,time,dimension->batch,time,n_head,n_d
        q = q.view(batch, len_q, self.n_head, n_d).permute(0, 2, 1, 3)#为多头并行计算和Q・K^T 矩阵乘法做维度适配
        k = k.view(batch, len_k, self.n_head, n_d).permute(0, 2, 1, 3)#qkv形状均为
        v = v.view(batch, len_v, self.n_head, n_d).permute(0, 2, 1, 3)#（batch,n_head,time,n_d）
        #score为（batch,n_head,time，time）
        score = q@k.transpose(2,3)/math.sqrt(n_d)#@矩阵乘法简写，对K最后两维转置，之后进行缩放
        '''
        掩码的生成代码：留在 Transformer 主类（make_pad_mask/make_casual_mask），负责 “造掩码”；填充0
        掩码的生效代码：留在多头注意力类（当前代码，masked_fill行），负责 “用掩码”；把0换为极小值
        '''
        if mask is not None:#若有掩码则将无效位置分数填充为极小值
            score = score.masked_fill(mask==0,-10000)
        score = self.softmax(score)@v#score(batch,n_head,time,n_d)
        #.contiguous()：让转置后的张量内存连续化,符合view要求
        #score(batch,len_q,dimension)
        score = score.permute(0,2,1,3).contiguous().view(batch,time,dimension)#合并多头操作
        out = self.w_combine(score)#out((batch,time,dimension))
        return out

#attention = MultHeadAttention(d_model, n_head)
#out = attention(x,x,x)
#print(out)
#print(out.shape)



