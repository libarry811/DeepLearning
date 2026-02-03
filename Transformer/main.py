import torch
from transformer import Transformer

def test_model():
    # 1. 定义设备 (如果有显卡就用显卡，没有就用CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前运行设备: {device}")

    # 2. 设定超参数 (随便设一些能跑通的参数)
    src_pad_idx = 1
    trg_pad_idx = 1
    enc_voc_size = 32000  # 假设源语言词表大小
    dec_voc_size = 32000  # 假设目标语言词表大小
    d_model = 512         # 嵌入维度
    max_len = 100         # 序列最大长度
    n_heads = 8           # 多头注意力的头数
    ffn_hidden = 2048     # 前馈层隐藏层维度
    n_layers = 6          # 编码器/解码器层数
    drop_prob = 0.1       # Dropout概率

    # 3. 初始化模型
    # 注意：这里的参数顺序必须严格对应 transformer.py 中 __init__ 的顺序
    model = Transformer(
        src_pad_idx,
        trg_pad_idx,
        enc_voc_size,
        dec_voc_size,
        d_model,
        max_len,
        n_heads,
        ffn_hidden,
        n_layers,
        drop_prob,  # 你的代码中 drop_prob 在 device 之前
        device
    ).to(device)

    print("模型初始化成功！")

    # 4. 构造假数据 (Batch Size = 2, 序列长度 = 50)
    batch_size = 2
    src_len = 50
    trg_len = 50

    # 生成随机整数作为输入 tokens (范围在 2 到 voc_size 之间，避开 padding)
    src = torch.randint(2, enc_voc_size, (batch_size, src_len)).to(device)
    trg = torch.randint(2, dec_voc_size, (batch_size, trg_len)).to(device)

    print(f"输入形状: src {src.shape}, trg {trg.shape}")

    # 5. 运行前向传播
    try:
        output = model(src, trg)
        print("-" * 30)
        print("🎉 恭喜！模型前向传播运行成功！")
        print(f"输出张量形状: {output.shape}")
        print(f"预期形状: ({batch_size}, {trg_len}, {dec_voc_size})")
        print("-" * 30)
    except Exception as e:
        print("-" * 30)
        print("❌ 运行出错，错误信息如下：")
        print(e)
        print("-" * 30)

if __name__ == '__main__':
    test_model()