# Soft Actor-Critic Reproduction

A PyTorch implementation of Soft Actor-Critic (SAC) for learning and experimenting with continuous-control reinforcement learning.

The implementation was written from scratch while studying SAC and tested on Pendulum-v1 and HalfCheetah-v5.

## Features

- Gaussian stochastic actor
- Reparameterization trick
- Tanh-squashed continuous actions
- Replay buffer
- Twin Q critics and target networks
- Automatic entropy temperature tuning
- Single-Q / Double-Q ablation
- Multi-seed evaluation
- W&B experiment tracking

## Project Structure

```text
SAC/
├── configs/
│   ├── pendulum.yaml
│   └── halfcheetah.yaml
├── src/
│   └── sac/
│       ├── actor.py
│       ├── critic.py
│       ├── replay_buffer.py
│       ├── agent.py
│       ├── evaluate.py
│       ├── train.py
│       └── utils.py
├── tests/
├── RESULTS.md
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
pip install -r requirements.txt
```

For MuJoCo environments:

```bash
pip install "gymnasium[mujoco]"
```

Run tests:

```bash
PYTHONPATH=src python -m pytest -q
```

## Training

### Pendulum-v1

```bash
PYTHONPATH=src python -m sac.train \
  --config configs/pendulum.yaml \
  --device cuda
```

### HalfCheetah-v5

```bash
PYTHONPATH=src python -m sac.train \
  --config configs/halfcheetah.yaml \
  --device cuda
```

Enable W&B logging:

```bash
PYTHONPATH=src python -m sac.train \
  --config configs/halfcheetah.yaml \
  --seed 0 \
  --device cuda \
  --wandb-mode online
```

## Multi-seed Experiments

Example for HalfCheetah:

```bash
for seed in 0 1 2; do
  PYTHONPATH=src python -m sac.train \
    --config configs/halfcheetah.yaml \
    --seed "$seed" \
    --device cuda \
    --wandb-mode online \
    --set wandb_group=halfcheetah-baseline
done
```

## Single-Q vs Double-Q

Standard SAC uses two critics and the conservative estimate

```math
min(Q_1,Q_2)
```

for both the Bellman target and the actor objective.

The implementation can also run a Single-Q variant:

```bash
PYTHONPATH=src python -m sac.train \
  --config configs/halfcheetah.yaml \
  --seed 0 \
  --device cuda \
  --wandb-mode online \
  --set double_q=false \
  --set wandb_group=halfcheetah-single-q
```

With `double_q=false`, only `Q1` and `target_Q1` are used.

## SAC Update

The main training update is:

```text
sample replay batch
        ↓
update critic
        ↓
update actor
        ↓
update alpha
        ↓
soft-update target critic
```

For Double-Q SAC, the critic target is

```math
y =
r +
\gamma (1-d)
\left[
\min\left(Q'_1(s',a'), Q'_2(s',a')\right)
-
\alpha \log \pi(a'|s')
\right].
```

The actor objective is

```math
L_\pi =
\mathbb{E}
\left[
\alpha \log \pi(a|s)
-
\min\left(Q_1(s,a), Q_2(s,a)\right)
\right].
```

For Single-Q, the minimum over the two critics is replaced by $Q_1$.

## Experiments

Completed:

- Pendulum-v1 baseline
- HalfCheetah-v5 baseline
- 3-seed evaluation
- Single-Q vs Double-Q ablation
- Fixed alpha vs automatic alpha ablation

Planned:

- UTD experiments
- REDQ reproduction

Results and plots are summarized in [RESULTS.md](RESULTS/RESULTS.md).