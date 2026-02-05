import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt  # 用于画 Loss 曲线
from d2l import torch as d2l
import os

# 导入你手搓的模型
from transformer import Transformer

# ==========================================
# 【新增修复】 解决 Matplotlib 中文乱码问题
# ==========================================
# Windows 系统通常用 'SimHei' (黑体)
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 0. 【补丁】修复 Windows 下 d2l 读取数据的编码报错
#    原因：Windows 中文版默认编码是 GBK，而数据集 fra.txt 是 UTF-8 编码。
#    如果不加这个补丁，d2l 库读取文件时会报 UnicodeDecodeError。
# ==========================================
def read_data_nmt_fixed():
    """载入“英语－法语”数据集（修复编码问题版）"""
    # 下载并解压数据，得到文件夹路径
    data_dir = d2l.download_extract('fra-eng')
    # 【核心修改点】 在 open 函数里加上 encoding='utf-8'，强制用 utf-8 读取
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()


# 用我们写好的修复函数，替换掉 d2l 库里原来的有 bug 的函数
# 这样后面调用 d2l.load_data_nmt 时，实际上运行的是我们的代码
d2l.read_data_nmt = read_data_nmt_fixed

# ==========================================
# 1. 超参数设置 (Hyperparameters)
#    这里不仅定义了参数，还解释了每个参数是干什么的
# ==========================================
# d_model: 词向量的维度，也是 Transformer 内部特征传递的维度
# 必须能被 num_heads 整除
num_hiddens = 32#num_hidden<=>d_model

# 编码器和解码器各自堆叠的层数
num_layers = 2

# Dropout 丢弃概率，防止过拟合 (通常在 0.1 ~ 0.5 之间)
dropout = 0.1

# 批量大小：一次训练喂给模型多少个句子
batch_size = 64

# 序列最大长度：
# 如果句子超过这个长度，会被截断；如果短于这个长度，会被填充(Pad)
num_steps = 10

# 学习率：决定梯度下降的步长
lr = 0.005

# 训练轮数：把所有数据反复训练多少遍
num_epochs = 200

# 前馈神经网络 (FFN) 的输入维度 (通常等于 d_model)
ffn_num_input = 32

# 前馈神经网络的隐藏层维度 (通常是 d_model 的 4 倍，这里设小点方便跑)
ffn_num_hiddens = 64

# 多头注意力的头数：决定把特征切分成几份并行关注
num_heads = 4

# 自动检测设备：如果有显卡(GPU)就用显卡，没有就用 CPU
device = d2l.try_gpu()
print(f"当前运行设备: {device}")

# ==========================================
# 2. 数据加载 (Data Loading)
# ==========================================
print("正在加载英语-法语数据集...")
# load_data_nmt 会自动帮我们做分词、构建词表、数字化等预处理
# train_iter 是数据迭代器，每次返回一个 batch 的数据
# src_vocab: 源语言(英文)词表
# tgt_vocab: 目标语言(法文)词表
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

# 获取 <pad> (填充符) 在词表中的索引 ID
# 这一点非常重要：计算 Loss 和 Attention Mask 时，都要忽略掉这些填充位
src_pad_idx = src_vocab['<pad>']#词是key，索引是value
tgt_pad_idx = tgt_vocab['<pad>']
print(f"源语言词表大小: {len(src_vocab)}, 目标语言词表大小: {len(tgt_vocab)}")
print(f"Padding ID - Src: {src_pad_idx}, Tgt: {tgt_pad_idx}")

# ==========================================
# 3. 初始化模型 (Model Initialization)
# ==========================================
# 实例化我们自己写的 Transformer 类
model = Transformer(
    src_pad_idx=src_pad_idx,  # 告诉模型源句子的 padding 是几，用于生成 Mask
    trg_pad_idx=tgt_pad_idx,  # 告诉模型目标句子的 padding 是几
    enc_voc_size=len(src_vocab),  # 编码器词表大小 (输入层 Embedding 用)
    dec_voc_size=len(tgt_vocab),  # 解码器词表大小 (输出层预测用)
    d_model=num_hiddens,  # 嵌入维度
    max_len=num_steps,  # 位置编码的最大长度
    n_heads=num_heads,  # 多头数量
    ffn_hidden=ffn_num_hiddens,  # FFN 隐藏层
    n_layers=num_layers,  # 层数
    drop_prob=dropout,  # Dropout
    device=device  # 设备
).to(device)  # .to(device) 把模型搬到 GPU 上去


# 权重初始化 (Xavier Initialization)就是给你的神经网络赋一个比较好的“初始运气”，让它训练起来更快、更稳，不容易死机。
# 这是一个训练 Trick：合理的初始化能让模型收敛得更快、更稳定
# 如果不加这一步，模型可能会一开始 Loss 很大，或者根本学不动
def xavier_init_weights(m):
    if type(m) == nn.Linear:
        # 动作：对这个层里的权重 (m.weight) 执行 Xavier 均匀初始化
        nn.init.xavier_uniform_(m.weight)

# 对整个模型 (model) 的每一个子层，都运行一遍上面这个函数
model.apply(xavier_init_weights)

# ==========================================
# 4. 定义损失函数和优化器 (Loss & Optimizer)
# ==========================================
# CrossEntropyLoss: 多分类任务的标准损失函数
# 关键参数 ignore_index=tgt_pad_idx：
# 意思是：如果真实标签是 <pad> (填充位)，就算预测错了也不扣分，不计算 Loss。
# 因为填充位只是为了凑长度，没有实际意义，模型不需要学习它。
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

# Adam 优化器：目前最流行的优化算法，能自适应调整学习率
optimizer = torch.optim.Adam(model.parameters(), lr=lr)#优化算法


# ==========================================
# 5. 定义单轮训练函数 (Training Step)
# ==========================================
def train_epoch(net, data_iter, lr, optimizer, criterion, device):
    net.train()
    epoch_loss = 0

    # 获取 <bos> 的索引 (通常 D2L 里的词表 bos 是 2)
    # 我们假设 tgt_vocab 是全局变量，或者你可以把它传进来
    # 为了保险，我们在函数内部获取一下
    bos_id = tgt_vocab['<bos>']#句子的“起始标志”

    for batch in data_iter:
        #d2l.load_data_nmt 生成的数据迭代器，每一轮 batch 都会吐出一个包含 4 个张量 (Tensor) 的列表
        X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]#对每一个张量执行 .to(device)

        # 【核心修改 1】构造错位的解码器输入 (Decoder Input)
        # 在 Y 的每一行最前面加一个 <bos>，并去掉最后一个词（保持长度不变）,最后一个一般为预处理加的eos
        '''Y = ["Je", "t'aime", "<eos>"]->["<bos>", "Je", "t'aime"]'''
        # 形状变化: [batch, seq_len] -> [batch, 1] + [batch, seq_len-1] -> [batch, seq_len]
        batch_size = Y.shape[0]
        # 这一步，是我们手动把 <bos> 加进去的！
        dec_input = torch.cat([#将两个张量拼接
            torch.full((batch_size, 1), bos_id, device=device),#创建指定形状、所有元素填充bos_id的张量
            Y[:, :-1]#截断目标序列最后一个 token，产生截断后张量
        ], dim=1)#沿第二个维度拼接

        optimizer.zero_grad()

        # 【核心修改 2】把错位后的 dec_input 喂给模型，而不是原始的 Y
        output = net(X, dec_input)

        # 计算 Loss (Target 依然是原始的 Y)
        # 这样就构成了：输入 <bos> -> 预测 Y[0]; 输入 Y[0] -> 预测 Y[1]
        # output 的原始形状是: [batch_size, seq_len, vocab_size]
        # 例如: [64, 10, 5000] -> 表示 64 个句子，每个句子 10 个词，每个词有 5000 种可能的概率
        #
        # CrossEntropyLoss 要求输入是二维的: [N, C] (N=样本总数, C=类别数/词表大小)
        # 所以我们需要把 batch_size 和 seq_len 两个维度合并成一个“总 token 数”维度
        # .reshape(-1, ...) 中的 -1 表示“自动计算该维度大小”
        # 变换后形状: [640, 5000] (假设 batch=64, len=10)
        output_reshape = output.reshape(-1, output.shape[-1])#output.shape[-1]取最后一个维度
        # Y (真实标签) 的原始形状是: [batch_size, seq_len]
        # 例如: [64, 10] -> 里面存的是每个位置正确的单词 ID
        #
        # CrossEntropyLoss 要求标签是一维的: [N]
        # 所以也要展平，变成一长串数字
        # 变换后形状: [640]
        Y_reshape = Y.reshape(-1)
        # criterion 是之前定义的 nn.CrossEntropyLoss
        # 它会拿 output_reshape 里的预测概率，去和 Y_reshape 里的真实答案做对比
        # 如果预测概率最高的词和真实答案不一样，Loss 就大；反之就小
        # 注意：我们在定义 criterion 时加了 ignore_index=pad_id，
        # 所以 Y_reshape 里那些等于 <pad> 的位置，不会产生 Loss，也不参与计算
        loss = criterion(output_reshape, Y_reshape)
        loss.backward()#反向传播
        # 4. 梯度裁剪 (Gradient Clipping)
        # 这行代码的作用是：检查所有参数的梯度，如果总长度超过 1.0，
        # 就按比例把它们缩小，强行限制在 1.0 以内。能让模型训练更稳定。
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()#更新参数
        epoch_loss += loss.item()#累加损失

    return epoch_loss / len(data_iter)# 最后返回这一轮的平均 Loss，data_iter 的长度就是这一轮有多少个 batch


# ==========================================
# 6. 主训练循环 (Main Training Loop)
# ==========================================
print("开始训练...")
loss_history = []  # 用一个列表把每一轮的 loss 存起来，后面画图用

total_start_time = time.time()

for epoch in range(num_epochs):
    start_time = time.time()

    # 跑一轮训练
    train_loss = train_epoch(model, train_iter, lr, optimizer, criterion, device)

    # 记录 Loss
    loss_history.append(train_loss)

    # 每 10 轮打印一次日志，看看进度
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Time: {time.time() - start_time:.2f}s')

print(f"训练结束！总耗时: {time.time() - total_start_time:.2f}s")

# ==========================================
# 7. 绘制 Loss 曲线 (Visualization)
# ==========================================
# 创建一个画布
plt.figure(figsize=(8, 5))
# 画折线图：x轴是轮数，y轴是 Loss
plt.plot(range(1, num_epochs + 1), loss_history, label='Training Loss', color='blue')
plt.xlabel('Epochs (训练轮数)')
plt.ylabel('Loss (损失值)')
plt.title('Transformer Training Loss Curve (训练曲线)')
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()  # 弹窗显示图像


# ==========================================
# 8. 预测/翻译测试 (Inference / Prediction)
#    这里我们手动实现一个简单的“贪心搜索” (Greedy Search)
# ==========================================
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    net.eval()  # 开启评估模式 (Dropout 关闭)

    # 1. 预处理源句子：分词 -> 查表转数字 -> 加 <eos> 结束符
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]

    # 2. 长度处理：短了就补 <pad>，长了就截断
    if len(src_tokens) > num_steps:
        src_tokens = src_tokens[:num_steps]
    else:
        src_tokens = src_tokens + [src_vocab['<pad>']] * (num_steps - len(src_tokens))

    # 转成 tensor 并放到 GPU 上，增加一个 batch 维度 -> [1, seq_len]
    src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)

    # 3. 准备解码器的初始输入
    # 翻译刚开始时，解码器只知道一个 <bos> (Beginning of Sequence) 开始符
    bos_token = tgt_vocab['<bos>']#获取索引，假设是2
    dec_input = torch.tensor([[bos_token]], dtype=torch.long, device=device)#tensor([[2]])变成了一个 2维张量(1, 1)
    '''
    [           # <--- 外层括号里只有 1 个元素 (那个内层列表) -> Batch Size = 1
        [ 2 ]     # <--- 内层列表里只有 1 个数字            -> Seq Len = 1
    ]
    '''
    output_seq = []

    # 4. 循环生成：一个词一个词地往外蹦
    for _ in range(num_steps):
        # 这里的 net(src, trg) 每次都会把当前的 dec_input 完整输进去
        # 虽然有点浪费计算力（没有用 KV Cache），但逻辑最简单
        with torch.no_grad():  # 预测时不需要算梯度，节省内存
            preds = net(src_tensor, dec_input)#源序列张量,解码器输入

        # preds 的形状是 [1, curr_seq_len, vocab_size]
        # 我们只关心最后一个时间步的输出 (也就是刚预测出来的那个词)
        next_token_logits = preds[:, -1, :]

        # 选概率最大的那个词 (Argmax) -> 这就是“贪心搜索”
        next_token = next_token_logits.argmax(dim=1).item()

        # 如果预测出了 <eos> (End of Sequence)，说明翻译结束了
        if next_token == tgt_vocab['<eos>']:
            break#跳出循环

        output_seq.append(next_token)#给人看的

        # 把预测出来的这个新词，拼接到输入序列的屁股后面
        # 作为下一轮预测的输入
        next_token_tensor = torch.tensor([[next_token]], device=device)#给机器看的
        dec_input = torch.cat([dec_input, next_token_tensor], dim=1)#拼接解码器之前时刻的输出

    # 把数字序列转回文字
    return ' '.join(tgt_vocab.to_tokens(output_seq))


# ==========================================
# 9. 实际测试看看效果
# ==========================================
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

print("\n=== 翻译效果测试 ===")
for eng, fra in zip(engs, fras):
    # 调用我们的预测函数
    translation = predict_seq2seq(model, eng, src_vocab, tgt_vocab, num_steps, device)

    # 计算 BLEU 分数 (机器翻译的常用评价指标)
    bleu_score = d2l.bleu(translation, fra, k=2)

    print(f"英文输入: {eng}")
    print(f"模型预测: {translation}")
    print(f"真实参考: {fra}")
    print(f"BLEU分数: {bleu_score:.3f}")
    print("-" * 20)