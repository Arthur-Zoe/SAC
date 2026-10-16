# SAC：学习与实验

# Soft Actor-Critic Reproduction

A from-scratch PyTorch implementation of Soft Actor-Critic (SAC),
built for studying continuous-control reinforcement learning.

Implemented features:

- Gaussian stochastic actor
- Reparameterization trick
- Tanh-squashed actions
- Double-Q critics
- Target networks
- Automatic entropy temperature
- Replay buffer
- W&B experiment tracking
- Multi-seed evaluation

Experiments:

- Pendulum-v1
- HalfCheetah-v5
- Single-Q vs Double-Q ablation

See [RESULTS.md](RESULTS.md) for experimental results.

本项目从原有 `test_actor_critic.py` 拆分而来，保留现有 Pendulum SAC 的网络、公式和更新顺序。原始四个 Python 文件完整保存在 `backups/original_20261015T152225Z/`，附 SHA-256 清单。没有引入 SB3，也没有复制其他项目的 SAC。

## 目录与职责

```text
SAC/
├── configs/
│   ├── pendulum.yaml
│   └── halfcheetah.yaml
├── src/sac/
│   ├── __init__.py
│   ├── actor.py
│   ├── critic.py
│   ├── replay_buffer.py
│   ├── agent.py
│   ├── evaluate.py
│   ├── utils.py
│   └── train.py
├── tests/test_smoke.py
├── backups/original_20261015T152225Z/
├── runs/                       # 每次实验独立保存，本地生成
├── README.md
├── REFACTOR_REPORT.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

| 文件 | 职责 |
| --- | --- |
| `actor.py` | 两层 ReLU MLP，Normal、rsample、tanh、动作缩放与 log probability correction、确定性动作 |
| `critic.py` | 单个 Q(s,a)，输出 `[B,1]`；双 Q 由 Agent 组合 |
| `replay_buffer.py` | 复制 transition，环形覆盖，有放回抽样，numpy → float32 torch tensor |
| `agent.py` | Actor、双 Critic、Target、optimizer、温度；分开各项 update，最后显式串联 |
| `evaluate.py` | 确定性动作；固定 reset seeds；默认 20 episodes，返回 `(mean_return, std_return)` |
| `utils.py` | Python/numpy/torch seeds 与设备选择 |
| `train.py` | 配置读取和校验、环境交互、更新、评估、W&B 与本地日志、保存 Actor |
| `tests/test_smoke.py` | shape、backward、公式、TimeLimit、训练入口、新旧数值等价、评估可重复性 |

## 安装与启动

在项目根目录执行所有相对路径命令：

```bash
cd /home/dyj/CS285_Spring2026/SAC
python3 -m pip install -r requirements.txt
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m sac.train --config configs/pendulum.yaml
```

当前主机已经有依赖，无须重新安装。建议新机器使用自己的虚拟环境。若希望不设置 `PYTHONPATH`，可在选定的 Python 环境中运行 `python3 -m pip install -e .`；随后用 `python3 -m sac.train ...`。`pyproject.toml` 只提供 src 包安装及 pytest 的测试目录设置。

Pendulum 默认配置完整保留原训练参数：seed 42，100000 steps，buffer 100000，batch 256，gamma 0.99，tau 0.005，learning rate 0.0003，start steps 5000，update after 1000，updates per step 1，eval interval 5000，alpha 初值 0.2、自动学习，hidden dim 256。评估改为 20 个固定 seeds。`torch_threads=1` 是 CPU 执行设置，可通过配置覆盖；不改变更新公式。

特别注意：原代码在第 1000 step 就开始梯度更新，第 5000 step 之后才开始用 Actor 收集数据；此行为保留。

短训练：

```bash
PYTHONPATH=src python3 -m sac.train --config configs/pendulum.yaml \
  --total-steps 6000 --device cpu --wandb-mode disabled
```

所有 YAML 字段都可以用重复的 `--set KEY=VALUE` 覆盖；布尔值使用 `true/false`。常用参数另有 `--seed`、`--env-name`、`--total-steps`、`--device`、`--wandb-mode`。同一个字段同时出现在 `--set` 与常用参数时，常用参数优先。未知键和非法类型直接报错。保存的 `config.yaml` 是覆盖后的有效配置。

## 多 seed 与消融配置

```bash
for seed in 0 1 2; do
  PYTHONPATH=src python3 -m sac.train --config configs/pendulum.yaml \
    --seed "$seed" --wandb-mode online --set wandb_group=pendulum-baseline
done
```

每个运行目录含环境名、seed、UTC 时间戳，避免互相覆盖。所有训练 seed 共用相同的评估 seeds，方便比较。同一配置与 seed 在相同运行环境下可复查；跨硬件/PyTorch 版本的逐位一致性未承诺。

现有固定 alpha 分支可以通过配置选择，用于之后的实验：

```bash
PYTHONPATH=src python3 -m sac.train --config configs/pendulum.yaml \
  --seed 0 --set auto_alpha=false --set init_alpha=0.2 \
  --set wandb_group=pendulum-fixed-alpha
```

尚未执行上述正式多 seed 或固定 alpha 长训练。没有加入 REDQ、CQL。

`double_q: bool` 默认为 `true`，保持原有双 Q 计算和更新顺序。可用
`--set double_q=false` 切换 Single-Q：只创建、训练 Q1 和 target_Q1，
critic target 直接使用 target_Q1，Actor 直接使用 Q1；critic loss 只有一项 MSE。
其他超参数及日志字段不变。Single-Q 的 `critic/q2` 记录与 `critic/q1` 相同的值，
用于兼容原日志字段，并不表示存在 Q2 网络。

## SAC 数据流

`agent.update()` 明确执行：

```text
replay batch -> critic target/update -> actor update -> alpha update -> soft target update
```

Critic 用 buffer 中实际执行的动作；target 的两个 Q 使用同一个 next action：

```text
y = r + gamma * (1 - terminated) * [min(target_Q1, target_Q2)(s', a') - alpha * log_pi(a'|s')]
critic_loss = MSE(Q1(s,a_buffer), y) + MSE(Q2(s,a_buffer), y)
actor_loss = mean(alpha.detach() * log_pi(a_new|s) - min(Q1,Q2)(s,a_new))
alpha_loss = -mean(log_alpha * (log_pi + target_entropy).detach())
target_entropy = -action_dim
target = (1 - tau) * target + tau * critic
```

Actor 更新期间冻结 Critic 参数，但保留 Q 对动作的梯度。alpha 使用 Actor 更新中同一次采样的 detached log probability；Target 不接收梯度。

环境在 `terminated or truncated` 时 reset，buffer 的 `done` 只保存 `terminated`。TimeLimit 不切断 bootstrap。Actor 保留原有 `log_std.clamp(-20,2)` 和 `log(scale*(1-tanh(u)^2)+1e-6)` 写法，没有改 reward scaling、更新频率或目标熵规则。

Buffer 从 deque 改为 list 环形索引，避免随机访问的线性开销；仍然有放回均匀采样。容量未溢出时可以复现原始索引顺序；满容量覆盖后，物理索引顺序与原 deque 不同，固定随机数对应的 transition 序列可能不同，但采样分布相同。

## 评估与输出

每次评估都执行 `env.reset(seed=eval_seed_start + episode_idx)`，默认 seeds 1000–1019；不抽样动作。独立评估环境不消耗训练用 numpy/torch 随机数。标准差是这 20 个 episode 的总体标准差（`ddof=0`），不是不同训练 seed 之间的不确定性。

按 `eval_interval` 评估，并在最后一步补做评估（最后一步恰好为间隔整数倍时只评估一次）。每次运行保存：

- `config.yaml`：实际配置。
- `runtime.json`：Python、torch、gymnasium、numpy、device、CPU threads。
- `metrics.jsonl`：日志行，均带 `env_step`，即使禁用 W&B 也保存。
- `summary.json`：更新次数、最终评估、alpha、抽样日志 loss 范围和最终参数有限性。
- `actor.pt`：与原模型 key 兼容的 Actor state_dict，仅用于策略加载；不包含 buffer/optimizer，**不支持恢复训练**。
- `wandb/`：W&B offline/online 模式下的 SDK 文件。

环境 transition、落盘指标和最终参数会检查 NaN/Inf。smoke test 另检查 backward 梯度。日志只保存定期快照，因此不是每一次中间梯度的完整数值审计。

## W&B

默认 `wandb_mode=offline`，project 为 `sac-reproduction`；不需要登录即可本地记录。禁用 W&B 时使用 `--wandb-mode disabled`，甚至不会 import W&B。

正式上传前登录你的账户：

```bash
wandb login
PYTHONPATH=src python3 -m sac.train --config configs/pendulum.yaml --wandb-mode online
```

离线文件可之后通过 `wandb sync <offline-run目录>` 上传。当前验证没有向 W&B 云端上传。

所有 `train/*`、`eval/*`、`loss/*`、`critic/*`、`policy/*` 都通过 `define_metric(..., step_metric="env_step")` 使用 environment step 为横轴，同一步的指标合并为一次 `run.log(..., step=step)`。参见 [W&B Run 文档](https://docs.wandb.ai/ref/python/experiments/run/)。

记录的指标包括：

```text
train/episode_return       train/episode_length       train/replay_buffer_size
eval/return_mean           eval/return_std
loss/critic                loss/actor                 loss/alpha
critic/q1                  critic/q2                  critic/target_q
policy/alpha               policy/entropy             policy/log_prob
```

默认每 100 environment steps 记录最新一次梯度更新的诊断；episode 结束记录回报和长度，评估时记录 mean/std。诊断不是区间平均。`critic/target_q` 表示 Bellman target y 的均值；`policy/entropy` 是 squashed/scaled 动作分布的 `-mean(log_prob)` Monte Carlo 估计，可以为负。

受限沙箱可能禁止 W&B 后台服务的本机 socket，offline 模式也需要该服务。遇到这种环境限制，可在有本机 socket 权限的终端运行；或显式禁用 W&B，继续保存 JSONL。代码不会静默禁用日志。

## HalfCheetah / Hopper

安装可选 MuJoCo 依赖后，切换配置即可；网络维度与动作范围由环境推断：

```bash
python3 -m pip install 'gymnasium[mujoco]>=1.0,<2'
PYTHONPATH=src python3 -m sac.train --config configs/halfcheetah.yaml --seed 0
PYTHONPATH=src python3 -m sac.train --config configs/halfcheetah.yaml \
  --env-name Hopper-v5 --seed 0 --set wandb_group=hopper-baseline
```

环境版本依据 [HalfCheetah 文档](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) 和 [Hopper 文档](https://gymnasium.farama.org/environments/mujoco/hopper/)。`halfcheetah.yaml` 沿用 Pendulum 的训练超参数，明确只是可运行的起点，尚未调参或验证收敛。当前主机已实际完成两个环境各 32 transitions + 1 次 SAC update 的兼容性检查，尚未长训练。之后按实验计划单独决定训练预算、seed 数量；无需改 SAC 公式。

## 已执行验收

完整证据和限制见 `REFACTOR_REPORT.md`：CPU smoke test 12 passed / 1 CUDA skipped；Pendulum 6000 steps / 5001 updates；W&B offline 1200 steps / 201 updates。新旧实现连续 3 次更新的所有对应指标、网络参数与 `log_alpha` 逐位一致。
# SAC
