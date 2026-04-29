import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from Model import TradeTransformer

# ==================== 配置 ====================
CONFIG = {
    # 数据
    "L": 20,                 # 输入交易日长度
    "K": 5,                  # 预测/评估步数
    "n_stocks": 500,
    "d_in": 10005,           # 输入维度
    "num_train_samples": 5000,
    "num_val_samples": 500,

    # 模型（需与预训练一致）
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

    # GRPO 参数
    "G": 4,                  # 每个状态的采样轨迹数
    "beta": 0.1,             # KL 惩罚初始系数
    "beta_decay": 0.995,     # 每步衰减系数（可选）
    "epsilon": 0.2,          # PPO 裁剪参数
    "entropy_coef": 0.01,    # 熵奖励系数（可选，鼓励探索）

    # 训练
    "batch_size": 16,        # GRPO 采样昂贵，可减小批大小
    "lr": 1e-4,              # 微调学习率
    "weight_decay": 0.01,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "pretrain_path": "./checkpoints/best_pretrain.pth",
    "save_dir": "./checkpoints",
    "log_interval": 10,
}

# ==================== 数据集（需提供未来真实收益率） ====================
class GRPODataset(Dataset):
    def __init__(self, num_samples, L, K, d_in, n_stocks):
        self.num_samples = num_samples
        self.L = L
        self.K = K
        self.d_in = d_in
        self.n_stocks = n_stocks

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 模拟数据：实际需从预处理数据加载
        market_seq = torch.randn(self.L, self.d_in)

        # 时间戳（模拟）
        base_day = np.random.randint(18000, 20000)
        enc_days = [base_day]
        for _ in range(self.L - 1):
            enc_days.append(enc_days[-1] + np.random.choice([1, 3]))
        enc_timestamps = torch.tensor(enc_days, dtype=torch.float32)

        dec_days = [enc_days[-1] + 1]
        for _ in range(self.K - 1):
            dec_days.append(dec_days[-1] + np.random.choice([1, 3]))
        dec_timestamps = torch.tensor(dec_days, dtype=torch.float32)

        # 未来真实收益率 (K, 500) —— 用于奖励计算
        future_returns = torch.randn(self.K, self.n_stocks) * 0.02  # 日收益率 ~ N(0, 0.02)

        # 指数收益率 (K,)
        benchmark_returns = torch.randn(self.K) * 0.01

        return {
            'market_seq': market_seq,
            'enc_timestamps': enc_timestamps,
            'dec_timestamps': dec_timestamps,
            'future_returns': future_returns,
            'benchmark_returns': benchmark_returns,
        }

# ==================== 奖励计算 ====================
def compute_sequence_return(weights_seq, future_returns, benchmark_returns, initial_weights=None, cost_kappa=0.001):
    """
    weights_seq: (K, 500) 采样的未来权重序列
    future_returns: (K, 500) 真实日收益率
    benchmark_returns: (K,) 指数日收益率
    返回：净超额收益（简单求和）
    """
    K = weights_seq.shape[0]
    if initial_weights is None:
        prev_weights = torch.ones(500) / 500  # 假设从等权开始
    else:
        prev_weights = initial_weights

    portfolio_returns = []
    for k in range(K):
        w = weights_seq[k]
        # 组合收益率
        daily_ret = torch.dot(w, future_returns[k])
        # 交易成本（双边换手）
        turnover = 0.5 * torch.sum(torch.abs(w - prev_weights))
        cost = 2 * cost_kappa * turnover
        net_ret = daily_ret - cost
        portfolio_returns.append(net_ret)
        prev_weights = w

    portfolio_returns = torch.stack(portfolio_returns)         # (K,)
    excess_returns = portfolio_returns - benchmark_returns     # (K,)
    # 累计超额收益（简单求和，也可用对数累加）
    total_excess = excess_returns.sum()
    return total_excess, portfolio_returns, excess_returns

# ==================== 采样函数 ====================
def sample_actions(model, market_seq, enc_ts, dec_ts, G):
    """从模型采样 G 组动作序列，返回 weight_sequences (G, B, K, 500) 和 log_probs"""
    B = market_seq.shape[0]
    K = dec_ts.shape[1]
    all_weights = []
    all_log_probs = []
    for g in range(G):
        logits, _ = model(market_seq, enc_ts, dec_ts, teacher_weights=None)  # (B, K, 500)
        # 构建 Dirichlet 分布
        alpha = torch.exp(logits) + 1e-8  # 确保 >0
        dist = torch.distributions.Dirichlet(alpha)
        weights = dist.rsample()  # (B, K, 500)
        log_prob = dist.log_prob(weights)  # (B, K) 每个样本每个时间步的对数概率
        all_weights.append(weights)
        all_log_probs.append(log_prob)
    # 堆叠为 (G, B, K, 500) 和 (G, B, K)
    weight_stack = torch.stack(all_weights, dim=0)
    log_prob_stack = torch.stack(all_log_probs, dim=0)
    return weight_stack, log_prob_stack

# ==================== GRPO 损失 ====================
def grpo_loss(current_log_probs, old_log_probs, advantages, epsilon=0.2, beta=0.1, kl_div=0.0):
    """
    current_log_probs: (G, B, K) 当前策略的对数概率
    old_log_probs: (G, B, K) 旧策略（采样时）的对数概率，需要 detach
    advantages: (G, B) 组内标准化后的优势
    """
    # 概率比
    ratio = torch.exp(current_log_probs - old_log_probs.detach())  # (G, B, K)
    # 平均到每个轨迹（可以对时间步求和或平均，这里采用 sum over K）
    ratio_sum = ratio.sum(dim=-1)  # (G, B)
    current_log_prob_sum = current_log_probs.sum(dim=-1) # (G, B)
    old_log_prob_sum = old_log_probs.sum(dim=-1)

    # 裁剪损失
    ratio_clipped = torch.clamp(ratio_sum, 1 - epsilon, 1 + epsilon)
    surr1 = ratio_sum * advantages
    surr2 = ratio_clipped * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL 惩罚项（使用采样概率比近似？这里简化：直接加在损失中）
    # 也可单独计算 kl = (old_log_prob_sum - current_log_prob_sum).mean() * beta
    kl_loss = beta * (old_log_prob_sum - current_log_prob_sum).mean()

    # 熵奖励（增加探索）
    entropy_bonus = -current_log_prob_sum.mean() * CONFIG.get('entropy_coef', 0.0)

    total_loss = policy_loss + kl_loss + entropy_bonus
    return total_loss

# ==================== 训练循环 ====================
def train_grpo_epoch(model, ref_model, dataloader, optimizer, device, epoch, config):
    model.train()
    ref_model.eval()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        market_seq = batch['market_seq'].to(device)
        enc_ts = batch['enc_timestamps'].to(device)
        dec_ts = batch['dec_timestamps'].to(device)
        future_returns = batch['future_returns'].to(device)
        benchmark_returns = batch['benchmark_returns'].to(device)

        B = market_seq.shape[0]
        K = dec_ts.shape[1]

        # 1. 用当前策略采样 G 组动作（无梯度）
        with torch.no_grad():
            weights, old_log_probs = sample_actions(model, market_seq, enc_ts, dec_ts, config['G'])
            # weights: (G, B, K, 500), old_log_probs: (G, B, K)

        # 2. 计算每条轨迹的超额收益
        rewards = torch.zeros(config['G'], B, device=device)
        for g in range(config['G']):
            for b in range(B):
                reward, _, _ = compute_sequence_return(
                    weights[g, b], future_returns[b], benchmark_returns[b]
                )
                rewards[g, b] = reward

        # 3. 组内标准化优势 (G, B)
        advantages = (rewards - rewards.mean(dim=0, keepdim=True)) / (rewards.std(dim=0, keepdim=True) + 1e-8)

        # 4. 重新计算当前策略的对数概率（用于损失）
        logits, expert_counts_list = model(market_seq, enc_ts, dec_ts, teacher_weights=None)
        alpha = torch.exp(logits) + 1e-8
        dist = torch.distributions.Dirichlet(alpha)
        new_log_probs = dist.log_prob(weights)  # (G, B, K) 需要扩展维度？注意 weights 第一维是G，logits 是 (B, K, 500)
        # 需要将 logits 扩展为 (G, B, K, 500) 或者对每个 g 计算对应的 log_prob
        # 简便做法：logits 对每个 g 是一样的模型输出，但 weights 不同，所以可以直接用 dist.log_prob(weights[g]) 对于每个g
        # 我们重构：遍历 G
        new_log_probs_list = []
        for g in range(config['G']):
            lp = dist.log_prob(weights[g])  # (B, K)
            new_log_probs_list.append(lp)
        new_log_probs = torch.stack(new_log_probs_list, dim=0)  # (G, B, K)

        # 5. 计算 KL 散度近似：用 old_log_probs 和 new_log_probs 以及参考策略的 log_prob
        #   参考策略（ref_model）的输出，用于 KL 惩罚：计算在参考策略下的对数概率
        with torch.no_grad():
            ref_logits, _ = ref_model(market_seq, enc_ts, dec_ts, teacher_weights=None)  # (B, K, 500)
            ref_alpha = torch.exp(ref_logits) + 1e-8
            ref_dist = torch.distributions.Dirichlet(ref_alpha)
            ref_log_probs = []
            for g in range(config['G']):
                ref_log_probs.append(ref_dist.log_prob(weights[g]))  # (B, K)
            ref_log_probs = torch.stack(ref_log_probs, dim=0)  # (G, B, K)

        # 计算 KL 近似：当前策略 log_prob - 参考策略 log_prob (期望的KL)
        # 这里简化：KL ≈ mean(ref_log_prob - new_log_prob)
        kl_div = (ref_log_probs.detach() - new_log_probs).mean()

        # 6. 损失计算
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

        # MoE 偏置更新
        moe_layers = [layer.moe for layer in model.encoder_layers] + [layer.moe for layer in model.decoder_layers]
        for moe, counts in zip(moe_layers, expert_counts_list):
            if counts is not None:
                moe.update_bias(counts)

        total_loss += loss.item()

        if batch_idx % config['log_interval'] == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f} | "
                  f"Avg Reward: {rewards.mean():.4f} | KL: {kl_div.item():.4f}")

    return total_loss / len(dataloader)

# ==================== 验证函数 ====================
@torch.no_grad()
def evaluate(model, ref_model, dataloader, device, config):
    model.eval()
    ref_model.eval()
    total_reward = 0.0
    num_samples = 0
    for batch in dataloader:
        market_seq = batch['market_seq'].to(device)
        enc_ts = batch['enc_timestamps'].to(device)
        dec_ts = batch['dec_timestamps'].to(device)
        future_returns = batch['future_returns'].to(device)
        benchmark_returns = batch['benchmark_returns'].to(device)

        B = market_seq.shape[0]
        # 采样一次（G=1）用于评估
        weights, _ = sample_actions(model, market_seq, enc_ts, dec_ts, G=1)
        for b in range(B):
            reward, _, _ = compute_sequence_return(
                weights[0, b], future_returns[b], benchmark_returns[b]
            )
            total_reward += reward.item()
            num_samples += 1
    avg_reward = total_reward / num_samples
    return avg_reward

# ==================== 主函数 ====================
def main():
    config = CONFIG
    os.makedirs(config['save_dir'], exist_ok=True)

    # 数据集
    train_dataset = GRPODataset(config['num_train_samples'], config['L'], config['K'],
                                config['d_in'], config['n_stocks'])
    val_dataset = GRPODataset(config['num_val_samples'], config['L'], config['K'],
                              config['d_in'], config['n_stocks'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    # 模型
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

    # 加载预训练权重
    pretrain_path = config['pretrain_path']
    if os.path.exists(pretrain_path):
        model.load_state_dict(torch.load(pretrain_path, map_location=config['device']))
        print("Loaded pre-trained model.")
    else:
        raise FileNotFoundError(f"Pre-trained model not found at {pretrain_path}")

    # 参考模型（冻结）
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
    ).to(config['device'])
    ref_model.load_state_dict(torch.load(pretrain_path, map_location=config['device']))
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    best_eval_reward = -float('inf')
    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_grpo_epoch(model, ref_model, train_loader, optimizer, config['device'], epoch, config)
        avg_val_reward = evaluate(model, ref_model, val_loader, config['device'], config)
        scheduler.step()

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Avg Excess Return: {avg_val_reward:.4f}")

        if avg_val_reward > best_eval_reward:
            best_eval_reward = avg_val_reward
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_grpo.pth'))
            print(f"  -> Best GRPO model saved (excess return={best_eval_reward:.4f})")

    print("GRPO training completed.")

if __name__ == "__main__":
    main()