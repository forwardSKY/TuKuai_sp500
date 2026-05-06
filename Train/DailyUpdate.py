import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from Model import TradeTransformer

# ==================== 配置 ====================
CONFIG = {
    # 数据
    "L": 20,                 # 输入交易日长度
    "K": 5,                  # 预测/评估步数
    "n_stocks": 500,
    "d_in": 10005,
    "data_dir": "./data/daily_samples",    # 存放每日新增样本的目录
    "recent_days": 60,       # 使用最近多少天的样本进行更新

    # 模型（与预训练/GRPO一致）
    "d_model": 512,
    "n_heads": 8,
    "d_c": 128,
    "d_ff": 1024,
    "n_experts": 8,
    "top_k": 2,
    "n_enc_layers": 6,
    "n_dec_layers": 6,
    "max_out_len": 5,
    "dropout": 0.1,
    "theta": 10000.0,

    # GRPO 每日更新参数
    "G": 8,                   # 更多采样轨迹以提高稳定性
    "beta": 0.2,              # 更保守的 KL 惩罚
    "epsilon": 0.2,
    "entropy_coef": 0.01,     # 保留少量熵奖励

    # 训练
    "batch_size": 4,          # 小批量（数据有限）
    "lr": 5e-5,               # 极低学习率
    "weight_decay": 0.01,
    "epochs": 1,              # 仅训练 1 个 epoch
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "./checkpoints/best_grpo.pth",  # 当前使用的模型（可能是 GRPO 后的）
    "save_path": "./checkpoints/model_daily.pth", # 更新后保存的模型
    "log_interval": 5,
}

# ==================== 数据集（从文件加载最近 N 天） ====================
class DailyDataset(Dataset):
    def __init__(self, data_dir, recent_days, L, K, d_in, n_stocks):
        # 假设数据文件为 data_dir/sample_YYYYMMDD.pt，每个文件是一个样本字典
        self.samples = []
        files = sorted(glob.glob(os.path.join(data_dir, 'sample_*.pt')))
        recent_files = files[-recent_days:]  # 取最近 N 天
        for f in recent_files:
            sample = torch.load(f)  # 应包含键：market_seq, enc_timestamps, dec_timestamps, future_returns, benchmark_returns
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ==================== 奖励计算（同之前） ====================
def compute_sequence_return(weights_seq, future_returns, benchmark_returns, initial_weights=None, cost_kappa=0.001):
    K = weights_seq.shape[0]
    if initial_weights is None:
        prev_weights = torch.ones(500) / 500
    else:
        prev_weights = initial_weights

    portfolio_returns = []
    for k in range(K):
        w = weights_seq[k]
        daily_ret = torch.dot(w, future_returns[k])
        turnover = 0.5 * torch.sum(torch.abs(w - prev_weights))
        cost = 2 * cost_kappa * turnover
        net_ret = daily_ret - cost
        portfolio_returns.append(net_ret)
        prev_weights = w
    portfolio_returns = torch.stack(portfolio_returns)
    excess_returns = portfolio_returns - benchmark_returns
    total_excess = excess_returns.sum()
    return total_excess, portfolio_returns, excess_returns

# ==================== 采样函数 ====================
def sample_actions(model, market_seq, enc_ts, dec_ts, G):
    B = market_seq.shape[0]
    K = dec_ts.shape[1]
    all_weights = []
    all_log_probs = []
    for g in range(G):
        logits, _ = model(market_seq, enc_ts, dec_ts, teacher_weights=None)
        alpha = torch.exp(logits) + 1e-8
        dist = torch.distributions.Dirichlet(alpha)
        weights = dist.rsample()  # (B, K, 500)
        log_prob = dist.log_prob(weights)  # (B, K)
        all_weights.append(weights)
        all_log_probs.append(log_prob)
    weight_stack = torch.stack(all_weights, dim=0)  # (G, B, K, 500)
    log_prob_stack = torch.stack(all_log_probs, dim=0)  # (G, B, K)
    return weight_stack, log_prob_stack

# ==================== GRPO 损失（同之前） ====================
def grpo_loss(current_log_probs, old_log_probs, advantages, epsilon=0.2, beta=0.1, kl_div=0.0):
    ratio_sum = torch.exp(current_log_probs.sum(-1) - old_log_probs.detach().sum(-1))  # (G, B)
    ratio_clipped = torch.clamp(ratio_sum, 1 - epsilon, 1 + epsilon)
    surr1 = ratio_sum * advantages
    surr2 = ratio_clipped * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    kl_loss = beta * (old_log_probs.sum(-1).detach() - current_log_probs.sum(-1)).mean()
    entropy_bonus = -CONFIG['entropy_coef'] * current_log_probs.sum(-1).mean()

    return policy_loss + kl_loss + entropy_bonus

# ==================== 每日更新主函数 ====================
def daily_update():
    config = CONFIG
    device = config['device']

    # 1. 加载数据
    dataset = DailyDataset(config['data_dir'], config['recent_days'],
                           config['L'], config['K'], config['d_in'], config['n_stocks'])
    if len(dataset) == 0:
        print("No recent samples found. Skipping update.")
        return
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    # 2. 加载当前模型
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
    ).to(device)

    if not os.path.exists(config['model_path']):
        raise FileNotFoundError(f"Model file not found: {config['model_path']}")
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.train()

    # 3. 参考模型（注意：对于每日更新，参考模型可以是本日更新前的模型快照）
    ref_model = TradeTransformer(
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
    ).to(device)
    ref_model.load_state_dict(torch.load(config['model_path'], map_location=device))
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # 4. 训练一个 epoch
    total_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        market_seq = batch['market_seq'].to(device)
        enc_ts = batch['enc_timestamps'].to(device)
        dec_ts = batch['dec_timestamps'].to(device)
        future_returns = batch['future_returns'].to(device)
        benchmark_returns = batch['benchmark_returns'].to(device)

        B = market_seq.shape[0]
        K = dec_ts.shape[1]
        G = config['G']

        # 采样
        with torch.no_grad():
            weights, old_log_probs = sample_actions(model, market_seq, enc_ts, dec_ts, G)

        # 奖励
        rewards = torch.zeros(G, B, device=device)
        for g in range(G):
            for b in range(B):
                reward, _, _ = compute_sequence_return(
                    weights[g, b], future_returns[b], benchmark_returns[b]
                )
                rewards[g, b] = reward

        # 优势（组内标准化）
        advantages = (rewards - rewards.mean(dim=0, keepdim=True)) / (rewards.std(dim=0, keepdim=True) + 1e-8)

        # 重新计算当前策略的对数概率
        logits, expert_counts_list = model(market_seq, enc_ts, dec_ts, teacher_weights=None)
        alpha = torch.exp(logits) + 1e-8
        dist = torch.distributions.Dirichlet(alpha)
        new_log_probs_list = []
        for g in range(G):
            new_log_probs_list.append(dist.log_prob(weights[g]))
        new_log_probs = torch.stack(new_log_probs_list, dim=0)  # (G, B, K)

        # 参考模型下的 log_prob（用于 KL）
        with torch.no_grad():
            ref_logits, _ = ref_model(market_seq, enc_ts, dec_ts, teacher_weights=None)
            ref_alpha = torch.exp(ref_logits) + 1e-8
            ref_dist = torch.distributions.Dirichlet(ref_alpha)
            ref_log_probs = []
            for g in range(G):
                ref_log_probs.append(ref_dist.log_prob(weights[g]))
            ref_log_probs = torch.stack(ref_log_probs, dim=0)

        kl_div = (ref_log_probs - new_log_probs).mean()

        # 损失
        loss = grpo_loss(
            current_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            epsilon=config['epsilon'],
            beta=config['beta'],
            kl_div=kl_div,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 更新 MoE 偏置
        moe_layers = [layer.moe for layer in model.encoder_layers] + [layer.moe for layer in model.decoder_layers]
        for moe, counts in zip(moe_layers, expert_counts_list):
            if counts is not None:
                moe.update_bias(counts)

        total_loss += loss.item()
        if batch_idx % config['log_interval'] == 0:
            print(f"Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f} | Avg Reward: {rewards.mean():.4f} | KL: {kl_div.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Daily update completed. Average loss: {avg_loss:.4f}")

    # 5. 保存更新后的模型
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    torch.save(model.state_dict(), config['save_path'])
    print(f"Model saved to {config['save_path']}")

if __name__ == "__main__":
    daily_update()