
## 1. 环境依赖与仓库结构
 
### 1.1 外部依赖
 
- Python ≥ 3.10
- PyTorch ≥ 2.0 (推荐 2.1+ with CUDA 12.1)
- transformers (用于可直接使用的 RMSNorm 等，可选)
- numpy, pandas, h5py (数据处理)
- wandb (可选，实验跟踪)
- deepspeed (可选，多卡专家并行)
- DeepEP (可选，MoE 通信库，需单独安装)
**安装命令：**
 
```bash
pip install torch numpy pandas h5py transformers wandb
# 若需 MoE 并行
pip install deepspeed
# DeepEP 安装参照官方仓库: https://github.com/deepseek-ai/DeepEP
```
 
### 1.2 项目文件结构
 
```plaintext
project/
├── Model.py              # 模型定义
├── PreTrain.py           # 预训练脚本
├── GRPOTrain.py          # GRPO 后训练脚本
├── DailyUpdate.py        # 每日在线更新脚本
├── configs/              # 配置文件（可选）
├── data/                 # 原始及预处理数据
│   ├── raw/              # 原始行情数据
│   └── processed/        # 预处理后的样本文件
├── checkpoints/          # 模型保存目录
└── daily_samples/        # 每日新增样本存放
```
 
---
 
## 2. 数据格式
 
所有样本采用 **PyTorch 张量字典** 格式，保存在 `.pt` 文件中，每个文件一个样本。
 
### 2.1 单个样本的字典结构
 
```python
{
    'market_seq': torch.Tensor,      # (L, D_in) float32
    'enc_timestamps': torch.Tensor,  # (L,)     float32
    'dec_timestamps': torch.Tensor,  # (K,)     float32
    'teacher_weights': torch.Tensor, # (K, 500) float32  (预训练必须，GRPO阶段可不含)
    'future_returns': torch.Tensor,  # (K, 500) float32  (GRPO阶段必须)
    'benchmark_returns': torch.Tensor, # (K,)  float32  (GRPO阶段必须)
}
```
 
**各字段说明：**
 
| 字段 | 维度 | 说明 |
|---|---|---|
| `market_seq` | `(L, D_in)` | 过去 L 个交易日的市场特征，D_in = 500 * feat_per_stock + n_macro。所有特征已按市值排序并稳健标准化，无未来信息。 |
| `enc_timestamps` | `(L,)` | 对应 L 个历史交易日的真实自然日编号（例如 `datetime.toordinal()` 或相对基准日期的天数）。跨周末、假日的时间间隔自动反映在数值差中。 |
| `dec_timestamps` | `(K,)` | 未来 K 个交易日的自然日编号，跳过非交易日。由交易日历预先生成。 |
| `teacher_weights` | `(K, 500)` | 仅预训练使用。未来 K 天每天的目标权重向量，非负，求和为 1。基于决策时刻 T 日已知的市值数据计算（无未来信息）。通常用 T 日收盘市值占指数总市值的比例，未来 5 天均用同一组权重（复制）。 |
| `future_returns` | `(K, 500)` | GRPO/每日更新使用。从 T 日收盘到 T+K 日收盘，每只股票的真实日收益率（简单收益率 `close / prev_close - 1`）。 |
| `benchmark_returns` | `(K,)` | GRPO/每日更新使用。同期 S&P 500 指数的日收益率。 |
 
### 2.2 预训练数据准备
 
- 预训练样本可以一次性生成，存储为 `.pt` 文件，按日期命名（如 `sample_20230401.pt`）。
- 训练时通过自定义 Dataset 加载。
**生成预训练样本的关键约束：**
 
1. **特征截面**：对每一天，按当天实际成分股市值降序排列，填充 500 个槽位（不足时补零并记录 mask）。
2. **标准化**：每个特征在截面上做稳健 Z-score（减去中位数，除以四分位距），或使用历史分位数归一化，避免使用未来信息。
3. **教师权重**：使用决策日 T 的已知市值计算 $w_i = \text{MarketCap}_T(i) / \sum_j \text{MarketCap}_T(j)$，复制到未来 5 天，即 `teacher_weights[k,:] = w` for all k。
### 2.3 GRPO 后训练数据准备
 
与预训练使用相同的历史样本，但额外需要 `future_returns` 和 `benchmark_returns`。这些字段必须与每个样本严格对齐，避免任何前视。
 
**奖励计算所需的数据范围**：对于每一个样本日期 T，需要在数据集中保存 T+1 到 T+5 的真实日收益率。因此，数据集的最晚日期要比回测结束日早至少 K 天。
 
### 2.4 每日在线更新数据
 
- 每日收盘后，生成一个新的 `.pt` 样本文件，放入 `./daily_samples/` 目录。
- 文件名建议包含日期，如 `sample_20250728.pt`。
- 该样本包含完整的 6 个字段（`teacher_weights` 可选，因为在线更新仅用 GRPO 奖励）。
- `DailyUpdate.py` 会自动加载该目录下最近 N 天的文件（由 `recent_days` 配置决定）。
---
 
## 3. 运行指南
 
### 3.1 预训练
 
- **脚本**：`PreTrain.py`
- **配置**：修改脚本开头的 `CONFIG` 字典，关键参数：
  - `L`, `K`, `n_stocks`, `d_in` 需与实际数据匹配。
  - `epochs`：训练轮数，建议 20~50。
  - `batch_size`：根据 GPU 显存调整，32 为常用值。
  - `device`：自动检测 GPU，若有多个 GPU 可手动设置为 `"cuda:0"`。
- **运行**：
```bash
python PreTrain.py
```
 
- **保存**：模型将保存在 `./checkpoints/best_pretrain.pth`。
**多卡训练（可选）：**
 
> 若使用 DataParallel 或 DDP，可在脚本开头初始化进程组，并将模型用 DistributedDataParallel 包装。由于模型包含 MoE，单卡通常足够，或使用 DeepSpeed ZeRO 优化。
 
### 3.2 GRPO 后训练
 
- **脚本**：`GRPOTrain.py`
- **前置条件**：确保 `./checkpoints/best_pretrain.pth` 存在。
- **配置**：同样修改 `CONFIG`，关键参数：
  - `pretrain_path`：预训练模型路径。
  - `G`：采样轨迹数，推荐 4~8（显存允许则更大）。
  - `beta`：KL 惩罚系数，0.1 左右。
  - `lr`：微调学习率，1e-4 左右。
  - `epochs`：10~20。
- **运行**：
```bash
python GRPOTrain.py
```
 
- **保存**：微调后的模型保存为 `./checkpoints/best_grpo.pth`。
### 3.3 每日在线更新
 
- **脚本**：`DailyUpdate.py`
- **前置条件**：
  - 已有微调模型 `./checkpoints/best_grpo.pth`（或使用前一日的 `model_daily.pth`）。
  - `./daily_samples/` 目录下有最近 N 天的样本文件。
- **配置**：调整 `CONFIG`：
  - `model_path`：起始模型路径。
  - `save_path`：更新后模型保存路径。
  - `data_dir`：每日样本存放目录。
  - `recent_days`：使用最近多少天的样本，建议 60。
  - `G`：8 或 16（增强稳定性）。
  - `beta`：0.2~0.5（更保守）。
  - `lr`：5e-5 或更低。
- **运行**（每天收盘后自动执行）：
```bash
python DailyUpdate.py
```
 
或加入 cron 任务。
 
---
 
## 4. 单卡与多卡配置
 
### 4.1 单卡训练（推荐起步）
 
- 默认的单卡配置已写在各训练脚本中，无需额外操作。模型通过 `.to(device)` 移动到单 GPU。
- 如果单卡显存不足，可减小 `batch_size`、`d_model`、`n_experts` 等。
### 4.2 多卡专家并行（MoE）
 
当模型规模大，MoE 专家数量多，单卡无法容纳全部专家参数时，需将不同专家放在不同 GPU 上，即**专家并行**。
 
- **工具**：DeepSpeed 或 DeepEP。
- **使用 DeepSpeed 的示例配置** (`ds_config.json`)：
```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 1
  },
  "moe": {
    "ep_size": 4,
    "moe_experts": 8,
    "top_k": 2,
    "min_capacity": 0,
    "noisy_gate_policy": "Jitter"
  }
}
```
 
> 注：`ep_size: 4` 表示专家并行度，即使用 4 张 GPU。
 
- **启动命令**：
```bash
deepspeed --num_gpus=4 PreTrain.py --deepspeed_config ds_config.json
```
 
> 需在代码中集成 DeepSpeed 初始化。复杂情况下推荐使用 DeepEP 通信库（DeepSeek 官方开源），具体参考其文档。
 
### 4.3 多卡数据并行
 
对于非 MoE 部分，普通的 DistributedDataParallel (DDP) 也可用于加速。需在每个脚本前添加初始化：
 
```python
import torch.distributed as dist
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
model = TradeTransformer(...).to(local_rank)
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```
 
然后使用 `torchrun` 启动：
 
```bash
torchrun --nproc_per_node=4 PreTrain.py
```
 
注意数据集的分发（`DistributedSampler`）和检查点保存。
 
---
 
## 5. 模型推理与执行
 
训练完成后，可以使用以下简单的推理脚本生成每日目标权重：
 
```python
import torch
from Model import TradeTransformer
 
model = TradeTransformer(d_in=10005, ...)  # 使用与训练相同的参数
model.load_state_dict(torch.load('checkpoints/best_grpo.pth'))
model.eval()
model.to('cuda')
 
# 读取今日样本
sample = torch.load('path/to/today_sample.pt')
market_seq = sample['market_seq'].unsqueeze(0).cuda()
enc_ts = sample['enc_timestamps'].unsqueeze(0).cuda()
dec_ts = sample['dec_timestamps'].unsqueeze(0).cuda()
 
with torch.no_grad():
    logits, _ = model(market_seq, enc_ts, dec_ts, teacher_weights=None)
    weights = torch.softmax(logits[:, 0, :], dim=-1)  # 取第一天的权重
    # weights: (1, 500)，可转为 numpy 执行交易
```
 
---
 
## 6. 常见问题
 
**Q：数据量不够大怎么办？**
 
> A：可以使用数据增强（如添加微小噪声）、滑动窗口构造更多样本，或使用更小的模型（减少 `d_model`、`n_layers`）。
 
**Q：训练不稳定，损失上下跳动**
 
> A：检查特征标准化是否正确、是否存在未来信息泄露；降低学习率；增大 KL 惩罚系数 `beta`。
 
**Q：推理时权重总和不为 1**
 
> A：Softmax 输出保证和为 1，若需使用 Dirichlet 采样，可取其均值。
 
**Q：MoE 专家不均衡**
 
> A：已实现无辅助损失的偏置更新，确认在每个 batch 后调用了 `update_bias`。若仍不均，可稍微增大 `bias_update_speed`。