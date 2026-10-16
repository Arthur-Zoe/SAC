# Results

Experiments were run on `HalfCheetah-v5` for 100k environment steps with three training seeds.

Unless otherwise stated, all hyperparameters were kept unchanged between compared variants.

---

## 1. Single-Q vs Double-Q

Standard SAC uses two critics and takes the smaller estimate when computing the critic target and actor objective:

```math
\min(Q_1, Q_2)
```

I compared the standard Double-Q implementation with a Single-Q variant.

### Evaluation return

![Single-Q vs Double-Q return](Q_eval_return_mean.png)

Double-Q performs better overall across the three seeds.

Single-Q can still learn, but its performance is generally lower and less consistent.

### Critic values

![Single-Q vs Double-Q critic](Q_critic_q1.png)

Single-Q tends to produce larger critic values, especially later in training.

However, the larger Q-values do not correspond to better evaluation return.

This is consistent with the motivation for clipped Double-Q learning: a single critic is more vulnerable to optimistic value estimates, while taking the minimum of two critics gives a more conservative estimate.

This experiment does not directly compare Q-values with Monte-Carlo returns, so it should not be treated as a direct measurement of overestimation.

### Conclusion

For this HalfCheetah experiment, Double-Q gives more reliable learning and better policy performance than Single-Q.

---

## 2. Fixed Alpha vs Automatic Alpha

SAC uses the temperature parameter $\alpha$ to control the weight of the entropy term.

I compared:

- fixed $\alpha = 0.2$
- automatic alpha, initialized at $0.2$

All other settings were unchanged.

HalfCheetah has a 6-dimensional action space, so the target entropy used by the implementation is

```math
\mathcal{H}_{\text{target}} = -6
```

### Evaluation return

![Fixed vs automatic alpha return](alpha_eval_return_mean.png)

Automatic alpha gives more consistent learning across the three seeds.

The fixed-alpha runs show much larger variation. Two seeds eventually reach reasonable performance, while one seed remains far below the automatic-alpha runs even after 100k environment steps.

The difference is therefore visible not only in final performance, but also in sample efficiency and robustness across seeds.

### Alpha

![Alpha](alpha.png)

For the fixed-alpha experiments,

```math
\alpha = 0.2
```

throughout training.

With automatic temperature tuning, alpha changes substantially during training.

It starts at $0.2$, quickly decreases during the early stage, reaches a very small value around 15k–20k steps, and then gradually increases again.

The final alpha value also differs between seeds.

This shows that the entropy weight used by SAC is not constant over the course of training.

### Policy entropy

![Policy entropy](entropy.png)

This plot shows the clearest difference between the two variants.

With automatic alpha, the policy entropy quickly approaches

```math
\mathcal{H}(\pi) \approx -6
```

and then remains close to the target entropy.

With fixed $\alpha = 0.2$, the policy entropy stays much higher and varies significantly between seeds.

In this experiment, keeping alpha fixed at $0.2$ therefore applies too much entropy regularization for much of training, while automatic temperature tuning adjusts the regularization strength to keep policy entropy near the target.

### Conclusion

Automatic temperature tuning is not simply searching for one better constant value of alpha.

Instead, alpha changes throughout training as part of a feedback process:

```text
policy entropy
      ↓
alpha update
      ↓
entropy regularization strength
      ↓
policy entropy
```

For HalfCheetah, this produces much more controlled policy entropy and more consistent learning than keeping $\alpha = 0.2$ fixed.

---

## Summary

The experiments so far support two important components of SAC.

### Double-Q

Using two critics and the conservative estimate

```math
\min(Q_1,Q_2)
```

produces better and more reliable learning than the Single-Q variant.

Single-Q also tends to produce larger critic values without achieving higher returns, which is consistent with more optimistic value estimation.

### Automatic temperature tuning

Automatic alpha keeps policy entropy close to the target

```math
\mathcal{H}_{\text{target}} = -6
```

while fixed $\alpha = 0.2$ produces substantially higher and less consistent entropy.

The automatic-alpha runs also show more reliable policy learning across seeds.

---

## Next

- UTD experiments
- REDQ reproduction