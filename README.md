## Single-Q vs Double-Q

Standard SAC uses two critics and the conservative estimate

$$
\min(Q_1, Q_2)
$$

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

$$
y =
r +
\gamma (1-d)
\left[
\min\left(Q'_1(s',a'), Q'_2(s',a')\right)
-
\alpha \log \pi(a'|s')
\right].
$$

The actor objective is

$$
L_\pi =
\mathbb{E}
\left[
\alpha \log \pi(a|s)
-
\min\left(Q_1(s,a), Q_2(s,a)\right)
\right].
$$

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

## Notes

Evaluation uses fixed initial seeds so that different checkpoints and training runs are evaluated under the same initial conditions.

For environment termination:

- `terminated or truncated` ends the episode and triggers reset.
- Only `terminated` is used in the Bellman terminal mask.

This keeps bootstrapping across time-limit truncations.

W&B is used for experiment tracking, while each run also saves local metrics and configuration files.