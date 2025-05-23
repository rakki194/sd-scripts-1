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


class SPARKFREE_ADOPT(Optimizer):
    r"""
    SPARKFREE-ADOPT: Schedule-Free ADOPT optimizer with stochastic rounding and BF16 support.

    Implements the ADOPT algorithm ("ADaptive gradient method with the OPTimal convergence rate") with:
    - max(sqrt(v), eps) denominator (elementwise)
    - Parameter update with previous momentum buffer
    - Momentum buffer update after parameter update
    - Optional elementwise gradient clipping
    - Stochastic rounding for BF16
    - Optional bias correction (as in Adam) [default: False]

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 1e-3)
        betas: Tuple of (beta1, beta2) for momentum and second moment (default: (0.9, 0.999))
        eps: Small constant for denominator (default: 1e-8)
        weight_decay: L2 penalty (default: 0.0)
        clip_lambda: Optional function of step -> clip value (default: None, disables clipping)
        use_stochastic_rounding: Whether to use stochastic BF16 rounding (default: True)
        use_bit_manipulation: Use bit-level stochastic rounding (default: True)
        bias_correction: Whether to use Adam-style bias correction (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_lambda: Optional[Callable[[int], float]] = None,
        use_stochastic_rounding: bool = True,
        use_bit_manipulation: bool = True,
        bias_correction: bool = False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clip_lambda=clip_lambda,
            use_stochastic_rounding=use_stochastic_rounding,
            use_bit_manipulation=use_bit_manipulation,
            bias_correction=bias_correction,
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
            clip_lambda = group["clip_lambda"]
            use_stochastic_rounding = group["use_stochastic_rounding"]
            use_bit_manipulation = group["use_bit_manipulation"]
            bias_correction = group.get("bias_correction", False)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = grad.clone().detach() ** 2

                m = state["m"]
                v = state["v"]
                step = state["step"] + 1
                state["step"] = step

                # Update v (second moment)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute denominator (elementwise max)
                if bias_correction:
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    m_hat = m / bias_correction1
                    v_hat = v / bias_correction2
                    denom = torch.max(
                        v_hat.sqrt(), torch.tensor(eps, device=v.device, dtype=v.dtype)
                    )
                else:
                    m_hat = m
                    denom = torch.max(
                        v.sqrt(), torch.tensor(eps, device=v.device, dtype=v.dtype)
                    )

                # Normalize gradient
                g_norm = grad / denom

                # Optional elementwise clipping
                if clip_lambda is not None:
                    c = clip_lambda(step)
                    g_norm = torch.clamp(g_norm, -c, c)

                # Weight decay
                if weight_decay != 0:
                    g_norm = g_norm + weight_decay * p.data

                # Parameter update with previous momentum
                p_new = p.data - lr * m_hat

                # Stochastic rounding for BF16
                if p.dtype == torch.bfloat16:
                    if use_stochastic_rounding:
                        if use_bit_manipulation:
                            copy_stochastic_(p.data, p_new)
                        else:
                            p.data.copy_(p_new.to(torch.bfloat16))
                    else:
                        p.data.copy_(p_new.to(torch.bfloat16))
                else:
                    p.data.copy_(p_new)

                # Momentum update (after param update)
                m.mul_(beta1).add_(g_norm, alpha=1 - beta1)

        return loss
