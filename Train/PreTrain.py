import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from Model import TradeTransformer  # 请确保 Model.py 在同一目录

# ==================== 配置 ====================
CONFIG = {
    # 数据
    "L": 20,                 # 输入交易日长度（历史窗口）
    "K": 5,                  # 预测/输出未来交易日长度（MTP 步数）
    "n_stocks": 500,         # S&P 500 成分股数量
    "feat_per_stock": 20,    # 每只股票的特征数（收益率、波动率等）
    "n_macro": 5,            # 宏观特征数
    "d_in": 500 * 20 + 5,    # 总输入维度：10005

    # 模型
    "d_model": 512,
    "n_heads": 8,
    "d_c": 128,              # MLA KV 压缩维度
    "d_ff": 1024,
    "n_experts": 8,
    "top_k": 2,
    "n_enc_layers": 6,
    "n_dec_layers": 6,
    "max_out_len": 5,        # 等于 K
    "dropout": 0.1,
    "theta": 10000.0,

    # 训练
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "epochs": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "./checkpoints",
    "log_interval": 50,
    "num_train_samples": 10000,   # 模拟训练样本数
    "num_val_samples": 1000,      # 模拟验证样本数
}

# ==================== 模拟数据集（请替换为真实数据） ====================
class SP500PretrainDataset(Dataset):
    def __init__(self, num_samples, L, K, d_in, n_stocks):
        self.num_samples = num_samples
        self.L = L
        self.K = K
        self.d_in = d_in
        self.n_stocks = n_stocks

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # ----- 1. 市场特征序列（模拟） -----
        market_seq = torch.randn(self.L, self.d_in)

        # ----- 2. 时间戳（模拟，实际需用真实交易日日期） -----
        base_day = np.random.randint(18000, 20000)
        enc_days = [base_day]
        for _ in range(self.L - 1):
            gap = np.random.choice([1, 1, 3])  # 1天或3天（周末）
            enc_days.append(enc_days[-1] + gap)
        enc_timestamps = torch.tensor(enc_days, dtype=torch.float32)

        dec_days = [enc_days[-1] + gap]
        for _ in range(self.K - 1):
            gap = np.random.choice([1, 1, 3])
            dec_days.append(dec_days[-1] + gap)
        dec_timestamps = torch.tensor(dec_days, dtype=torch.float32)

        # ----- 3. 教师权重（模拟：等权。实际应为 S&P 500 市值权重） -----
        # 真实用法：从预先计算的特征中读取目标权重
        # teacher_weights = torch.from_numpy( np.load(f"path/to/weight_{idx}.npy") )
        teacher_weights = torch.ones(self.K, self.n_stocks) / self.n_stocks

        return {
            'market_seq': market_seq,
            'enc_timestamps': enc_timestamps,
            'dec_timestamps': dec_timestamps,
            'teacher_weights': teacher_weights,
        }

# ==================== 损失函数 ====================
def pretrain_loss_fn(logits, teacher_weights):
    """
    logits: (B, K, 500)  模型输出的 logits（未经 Softmax）
    teacher_weights: (B, K, 500)  目标权重，非负且 sum=1
    返回：标量损失（交叉熵）
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    ce = -torch.sum(teacher_weights * log_probs, dim=-1)  # (B, K)
    return ce.mean()

# ==================== 训练循环 ====================
def train_epoch(model, dataloader, optimizer, device, epoch, config):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        # 将数据移到设备
        market_seq = batch['market_seq'].to(device)
        enc_ts = batch['enc_timestamps'].to(device)
        dec_ts = batch['dec_timestamps'].to(device)
        teacher_w = batch['teacher_weights'].to(device)

        optimizer.zero_grad()

        # 前向传播：使用 teacher forcing
        logits, expert_counts_list = model(
            market_seq,
            enc_ts,
            dec_ts,
            teacher_weights=teacher_w
        )

        loss = pretrain_loss_fn(logits, teacher_w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ---------- 更新 MoE 专家偏置（无辅助损失） ----------
        # 我们需要将每层的 expert_counts 对应传入各层的 moe.update_bias
        # 这里提供一个正确索引的方式（假设 counts 顺序与层顺序一致）
        moe_layers = []
        for layer in model.encoder_layers:
            moe_layers.append(layer.moe)
        for layer in model.decoder_layers:
            moe_layers.append(layer.moe)

        # expert_counts_list 是各层返回的 counts 列表
        for moe, counts in zip(moe_layers, expert_counts_list):
            if counts is not None:
                moe.update_bias(counts)

        total_loss += loss.item()

        if batch_idx % config['log_interval'] == 0:
            print(f"Epoch {epoch:2d} | Batch {batch_idx:4d}/{len(dataloader)} | Loss: {loss.item():.6f}")

    return total_loss / len(dataloader)

# ==================== 验证函数 ====================
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        market_seq = batch['market_seq'].to(device)
        enc_ts = batch['enc_timestamps'].to(device)
        dec_ts = batch['dec_timestamps'].to(device)
        teacher_w = batch['teacher_weights'].to(device)

        logits, _ = model(market_seq, enc_ts, dec_ts, teacher_weights=teacher_w)
        loss = pretrain_loss_fn(logits, teacher_w)
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ==================== 主函数 ====================
def main():
    config = CONFIG
    os.makedirs(config['save_dir'], exist_ok=True)

    # 1. 数据
    train_dataset = SP500PretrainDataset(
        config['num_train_samples'], config['L'], config['K'],
        config['d_in'], config['n_stocks']
    )
    val_dataset = SP500PretrainDataset(
        config['num_val_samples'], config['L'], config['K'],
        config['d_in'], config['n_stocks']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    # 2. 模型
    model = TradeTransformer(
        d_in=config['d_in'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_c=config['d_c'],
        d_ff=config['d_ff'],
        n_experts=config['n_experts'],
        top_k=config['top_k'],
        n_enc_layers=config['n_enc_layers'],
        n_dec_layers=config['n_dec_layers'],
        max_out_len=config['K'],
        dropout=config['dropout'],
        theta=config['theta']
    ).to(config['device'])

    # 3. 优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    best_val_loss = float('inf')

    # 4. 训练循环
    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, config['device'], epoch, config)
        val_loss = evaluate(model, val_loader, config['device'])
        scheduler.step()

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_pretrain.pth'))
            print(f"  -> Best model saved (val_loss={best_val_loss:.6f})")

    print("Pre-training completed.")

if __name__ == "__main__":
    main()