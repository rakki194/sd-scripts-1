# SPARKFREE Optimizer

SPARKFREE (**S**tochastic **P**arameter **A**djustment with **R**andomized **K**icks For **R**obust, **E**ffortless **E**volution) is a modern, schedule-free optimizer for deep learning, based on the Schedule-Free AdamW algorithm described in "The Road Less Scheduled" (Defazio et al., 2024).

## Overview

SPARKFREE eliminates the need for hand-crafted learning rate schedules by using a principled averaging and interpolation scheme. It achieves state-of-the-art performance across a wide range of tasks, matching or exceeding the best-tuned schedules, while requiring no additional hyperparameters over AdamW.

- **No learning rate schedule required**: No need to set a training horizon or schedule.
- **Momentum via interpolation**: Uses a novel form of momentum that is worst-case optimal for any value of the momentum parameter.
- **Averaged parameters**: Maintains both a base and an averaged parameter buffer, always returning the averaged parameter.
- **AdamW-style updates**: Incorporates adaptive moment estimation and decoupled weight decay.
- **BF16 and stochastic rounding support**: Efficient and robust for large models.

## Algorithm

At each step, for each parameter:

1. **Interpolate**: Compute $y_t = (1-\beta) z_t + \beta x_t$
2. **Gradient**: Compute gradient at $y_t$
3. **Second moment**: $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
4. **Learning rate**: $\gamma_t = \text{lr} \cdot \sqrt{1-\beta_2^t} \cdot \min(1, t/\text{warmup})$
5. **Weight decay**: Apply at $y_t$
6. **Base update**: $z_{t+1} = z_t - \gamma_t g_t / (\sqrt{v_t} + \epsilon)$
7. **Averaging weight**: $c_{t+1} = \gamma_t^2 / \sum_{i=1}^t \gamma_i^2$
8. **Averaged update**: $x_{t+1} = (1-c_{t+1}) x_t + c_{t+1} z_{t+1}$
9. **Return**: Always use $x_T$ as the model parameter.

## Usage

```python
import torch
from sparkfree import SPARKFREE

model = YourModel()
optimizer = SPARKFREE(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    warmup_steps=1000,
)

model = model.to(torch.bfloat16)  # Recommended for large models
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate |
| `betas` | (0.9, 0.999) | Interpolation/momentum and second moment decay |
| `eps` | 1e-8 | Numerical stability |
| `weight_decay` | 0.0 | Decoupled weight decay (applied at $y_t$) |
| `warmup_steps` | 0 | Linear warmup steps |
| `use_stochastic_rounding` | True | Use stochastic BF16 rounding |
| `stochastic_rounding_prob` | 0.5 | Probability for stochastic rounding (if not using bit manipulation) |
| `stochastic_rounding_magnitude` | None | Magnitude for stochastic rounding noise |
| `use_bit_manipulation` | True | Use bit-level stochastic rounding (recommended) |

## Best Practices

- **BF16 recommended**: For large models, use BF16 for parameters and activations for best efficiency and stability.
- **No schedule needed**: Do not use learning rate schedules; tune only the base learning rate and weight decay.
- **Momentum**: The default $\beta=0.9$ works well for most problems; try $0.98$ for very large models.
- **Warmup**: Use a short warmup (e.g., 1-5\% of total steps) for stability at the start of training.

## Theory

SPARKFREE is based on the "Schedule-Free" approach, which unifies learning rate schedules and iterate averaging. It achieves optimal convergence rates for convex and non-convex problems, and is robust to the choice of momentum parameter.

See: [The Road Less Scheduled (Defazio et al., 2024)](https://arxiv.org/abs/2402.09353)

## Acknowledgments

- **Schedule-Free**: This optimizer is a direct implementation of the Schedule-Free AdamW algorithm from Defazio et al. (2024).
- **SPARKLES**: Some implementation details (BF16 support, stochastic rounding) are inspired by the SPARKLES optimizer.

## License

MIT
