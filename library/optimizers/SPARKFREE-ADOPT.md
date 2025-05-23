# SPARKFREE-ADOPT Optimizer

SPARKFREE-ADOPT is a schedule-free, adaptive optimizer that implements the ADOPT algorithm ("ADaptive gradient method with the OPTimal convergence rate") with support for stochastic BF16 rounding, optional elementwise gradient clipping, and optional Adam-style bias correction. It is designed for robust, efficient deep learning training with optimal theoretical guarantees.

## Mathematical Formulation

The ADOPT algorithm is a variant of Adam/AdaGrad with a crucial difference in the update order and denominator:

- **Momentum update order:** The parameter is updated using the *previous* momentum buffer, and the momentum buffer is updated *after* the parameter update.
- **Denominator:** The adaptive step is normalized by $\max(\sqrt{v}, \epsilon)$ (elementwise), not $\sqrt{v} + \epsilon$.
- **Optional elementwise gradient clipping:** The normalized gradient can be clipped elementwise to $[-c_t, c_t]$ before the momentum update.
- **Optional bias correction:** If enabled, the momentum and second moment estimates are bias-corrected as in Adam (see below).

### Update Equations

Let $g_t$ be the gradient at step $t$, $m_t$ the momentum buffer, $v_t$ the second moment, $\beta_1, \beta_2$ the decay rates, $\alpha$ the learning rate, and $\epsilon$ a small constant.

1. **Second moment update:**
   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$
2. **Bias correction (optional):**
   $$
   \hat{m}_t = m_t / (1 - \beta_1^t) \\
   \hat{v}_t = v_t / (1 - \beta_2^t)
   $$
   If bias correction is disabled, use $m_t$ and $v_t$ directly.
3. **Denominator:**
   $$
   d_t = \max(\sqrt{\hat{v}_t}, \epsilon)
   $$
4. **Normalized gradient:**
   $$
   \hat{g}_t = g_t / d_t
   $$
5. **Elementwise clipping (optional):**
   $$
   \hat{g}_t = \mathrm{clip}(\hat{g}_t, -c_t, c_t)
   $$
6. **Weight decay:**
   $$
   \hat{g}_t = \hat{g}_t + \lambda \theta_{t-1}
   $$
7. **Parameter update (using previous momentum):**
   $$
   \theta_t = \theta_{t-1} - \alpha \hat{m}_{t-1}
   $$
8. **Momentum update:**
   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) \hat{g}_t
   $$

- $c_t$ is the elementwise clip value (can be constant or scheduled).
- $\lambda$ is the weight decay coefficient.
- $\hat{m}_t, \hat{v}_t$ are bias-corrected if `bias_correction=True`, otherwise use $m_t, v_t$ directly.

## Why is this optimal?

ADOPT achieves the optimal $\mathcal{O}(1/\sqrt{T})$ convergence rate for smooth nonconvex optimization, matching the best known rates for adaptive methods. The use of $\max(\sqrt{v}, \epsilon)$ prevents instability from near-zero denominators, and the update order ensures robust, stable training even with large learning rates or poor initialization.

See the original ADOPT paper for theoretical details:

- [ADOPT: Modified Adam Can Converge with Any $\beta_2$ with the Optimal Rate (Taniguchi et al., 2024)](https://arxiv.org/abs/2402.09353)

## Usage Example

```python
import torch
from sparkfree import SPARKFREE_ADOPT

model = YourModel()
optimizer = SPARKFREE_ADOPT(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    clip_lambda=lambda step: min(10.0, 0.1 * step**0.25),  # Optional
    bias_correction=True,  # Optional, set to True for Adam-style bias correction
)

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
| `betas` | (0.9, 0.999) | Momentum and second moment decay |
| `eps` | 1e-8 | Small constant for denominator |
| `weight_decay` | 0.0 | L2 penalty |
| `clip_lambda` | None | Function of step -> clip value (elementwise clipping) |
| `use_stochastic_rounding` | True | Use stochastic BF16 rounding |
| `use_bit_manipulation` | True | Use bit-level stochastic rounding |
| `bias_correction` | False | Use Adam-style bias correction |

## Practical Tips

- **BF16 recommended:** For large models, use BF16 for parameters and activations for best efficiency and stability.
- **Clipping:** Use elementwise clipping if you observe instability or large spikes in gradients, especially at the start of training.
- **No learning rate schedule needed:** The optimizer is robust to the choice of learning rate and does not require a schedule.
- **Momentum:** The default $\beta_1=0.9$ works well for most problems.
- **Bias correction:** Enable `bias_correction=True` if you want Adam-like behavior, especially for small batch sizes or short training runs. For pure ADOPT theory, leave it disabled (default).

## Differences from Adam/AdamW and SPARKFREE

- **Denominator:** Uses $\max(\sqrt{v}, \epsilon)$ instead of $\sqrt{v} + \epsilon$.
- **Update order:** Parameter is updated with previous momentum, then momentum is updated.
- **Clipping:** Supports elementwise gradient clipping.
- **Schedule-free:** No learning rate schedule or bias correction needed by default.
- **Stochastic rounding:** Supports efficient BF16 training as in SPARKFREE.
- **Bias correction:** Optional, for Adam-like behavior.

## References

- [ADOPT: Modified Adam Can Converge with Any $\beta_2$ with the Optimal Rate (Taniguchi et al., 2024)](https://arxiv.org/abs/2402.09353)
- [SPARKFREE Optimizer](./SPARKFREE.md)

## License

MIT
