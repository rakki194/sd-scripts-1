import torch
from torch.optim import Optimizer
from typing import Callable, Optional, Tuple


# @torch.compile
def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """
    Apply stochastic rounding by manipulating bits in the mantissa.
    Based on Lode's implementation.

    Args:
        target: Target tensor to store result in BF16 format
        source: Source tensor in FP32 format
    """
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))


class SPARKFREE(Optimizer):
    r"""
    SPARKFREE: Stochastic Parameter Adjustment with Randomized Kicks For Robust, Effortless Evolution

    Implements the Schedule-Free AdamW optimizer as described in the paper "The Road Less Scheduled"
    - Maintains two parameter buffers per parameter: z (base) and x (averaged)
    - At each step:
        1. y_t = (1-beta) * z_t + beta * x_t
        2. Compute gradient at y_t
        3. AdamW-style update for z_{t+1} (with bias correction, warmup, and weight decay at y_t)
        4. Update x_{t+1} as a weighted average of x_t and z_{t+1} with weight c_{t+1}
        5. c_{t+1} = gamma_t^2 / sum_{i=1}^t gamma_i^2
    - Returns x_T as the final parameter

    Supports BF16 and stochastic rounding for memory efficiency and stability.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        use_stochastic_rounding: bool = True,
        stochastic_rounding_prob: float = 0.5,
        stochastic_rounding_magnitude: Optional[float] = None,
        use_bit_manipulation: bool = True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            use_stochastic_rounding=use_stochastic_rounding,
            stochastic_rounding_prob=stochastic_rounding_prob,
            stochastic_rounding_magnitude=stochastic_rounding_magnitude,
            use_bit_manipulation=use_bit_manipulation,
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            warmup_steps = group["warmup_steps"]
            use_stochastic_rounding = group["use_stochastic_rounding"]
            stochastic_rounding_prob = group["stochastic_rounding_prob"]
            stochastic_rounding_magnitude = group["stochastic_rounding_magnitude"]
            use_bit_manipulation = group["use_bit_manipulation"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["z"] = p.data.clone().detach()
                    state["x"] = p.data.clone().detach()
                    state["v"] = torch.zeros_like(p.data)
                    state["sum_gamma2"] = 0.0

                z = state["z"]
                x = state["x"]
                v = state["v"]
                step = state["step"] + 1
                state["step"] = step

                # Compute effective learning rate with warmup and bias correction
                bias_correction2 = 1 - beta2**step
                if warmup_steps > 0:
                    warmup = min(1.0, step / warmup_steps)
                else:
                    warmup = 1.0
                gamma_t = lr * (bias_correction2**0.5) * warmup

                # y_t = (1-beta1) * z + beta1 * x
                y = (1 - beta1) * z + beta1 * x

                # Compute gradient at y (already provided by autograd)
                g = grad

                # AdamW-style second moment update
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Weight decay at y
                if weight_decay != 0:
                    g = g + weight_decay * y

                # z_{t+1} update
                denom = v.sqrt().add_(eps)
                z_new = z - gamma_t * g / denom

                # c_{t+1} update
                if isinstance(state["sum_gamma2"], float):
                    sum_gamma2 = torch.tensor(
                        state["sum_gamma2"], device=p.device, dtype=torch.float32
                    )
                else:
                    sum_gamma2 = state["sum_gamma2"]
                sum_gamma2 = sum_gamma2 + gamma_t**2
                c = gamma_t**2 / sum_gamma2
                state["sum_gamma2"] = sum_gamma2

                # x_{t+1} update
                x_new = (1 - c) * x + c * z_new

                # Write back
                state["z"] = z_new.detach()
                state["x"] = x_new.detach()

                # Write x_new to p.data (so model always holds the averaged parameter)
                if p.dtype == torch.bfloat16:
                    if use_stochastic_rounding:
                        if use_bit_manipulation:
                            copy_stochastic_(p.data, x_new)
                        else:
                            # fallback: just convert
                            p.data.copy_(x_new.to(torch.bfloat16))
                    else:
                        p.data.copy_(x_new.to(torch.bfloat16))
                else:
                    p.data.copy_(x_new)

        return loss
