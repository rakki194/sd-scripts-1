import torch
from torch.optim import Optimizer
from typing import Callable, Optional, Tuple

from .copy_stochastic_cuda_wrapper import copy_stochastic_bf16_ as cuda_copy_stochastic_bf16_, fused_optimizer as cuda_fused_optimizer
from .copy_stochastic_cuda_wrapper import stochastic_bf16_rounding_ as cuda_stochastic_bf16_rounding_

class SPARKLES(Optimizer):
    r"""
    Implements the SPARKLES optimization algorithm: Stochastic Parameter Adjustment with Randomized Kick for Learning Enhancement Strategy.

    Based on SAVEUS, but incorporates the stochastic strategy from SADAM research to help escape local minima and saddle points.

    The optimizer combines several advanced optimization techniques:
    1. Gradient Centralization: Removes the mean of gradients for each layer:
       g_t = g_t - mean(g_t)

    2. Adaptive Gradient Normalization: Normalizes gradients using their standard deviation:
       g_t = (1 - α) * g_t + α * (g_t / std(g_t))
       where α is the normalization parameter

    3. Momentum with Amplification:
       - First moment: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
       - Amplified gradient: g_t = g_t + amp_fac * m_t
       where β₁ is the first moment decay rate

    4. Adaptive Step Sizes:
       - Second moment: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
       - Bias correction: m̂_t = m_t / (1 - β₁ᵗ)
                         v̂_t = v_t / (1 - β₂ᵗ)
       - Step size: η_t = lr / (1 - β₁ᵗ)
       where β₂ is the second moment decay rate

    5. Stochastic Update Strategy:
       - When gradients become small between iterations (||g_t - g_{t-1}|| < threshold),
         apply stochastic operator R to the updates to escape potential local minima
       - The permutation strategy can be set to either 'global' (randomly permutes all elements) or 'none' (no permutation)

    6. Stochastic BF16 Rounding (Enabled by default):
       - Applies controlled stochastic noise during bfloat16 conversion
       - Helps improve training dynamics by adding beneficial noise at the precision level
       - Uses bit-manipulation for true stochastic rounding

    Complete Update Rule:
    1. If decouple_weight_decay:
       θ_t = θ_{t-1} * (1 - η_t * λ) - η_t * g_t / √(v̂_t + ε)
    2. Otherwise:
       θ_t = θ_{t-1} - η_t * (g_t + λ * θ_{t-1}) / √(v̂_t + ε)

    Where:
    - θ_t: Parameters at step t
    - η_t: Learning rate with bias correction
    - g_t: Gradient (after centralization, normalization, and amplification)
    - v̂_t: Bias-corrected second moment estimate
    - λ: Weight decay coefficient
    - ε: Small constant for numerical stability

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional):
            Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional):
            Coefficients for computing running averages of gradient (β₁) and its square (β₂) (default: (0.9, 0.999)).
        eps (float, optional):
            Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional):
            Weight decay (L2 penalty) (default: 0).
        centralization (float, optional):
            Strength of gradient centralization (default: 0.5).
        normalization (float, optional):
            Interpolation factor for normalized gradients (default: 0.5).
        normalize_channels (bool, optional):
            Whether to normalize gradients channel-wise (default: True).
        amp_fac (float, optional):
            Amplification factor for the momentum term (default: 2.0).
        clip_lambda (Optional[Callable[[int], float]], optional):
            Function computing gradient clipping threshold from step number (default: step**0.25).
        decouple_weight_decay (bool, optional):
            Whether to apply weight decay directly to weights (default: False).
        clip_gradients (bool, optional):
            Whether to enable gradient clipping (default: False).
        stochastic_threshold (float, optional):
            Threshold for applying stochastic updates (default: 1e-6).
        use_stochastic_rounding (bool, optional):
            Whether to apply stochastic rounding when converting to BF16 (default: True).
        stochastic_rounding_prob (float, optional):
            Probability of applying noise in stochastic rounding (default: 0.5).
        stochastic_rounding_magnitude (float, optional):
            Magnitude of noise to apply in stochastic rounding (default: None, uses epsilon of BF16).
        use_bit_manipulation (bool, optional):
            Whether to use bit manipulation for stochastic rounding (default: True).
        permutation_strategy (str, optional):
            Which permutation strategy to use ('global' or 'none') (default: 'global').
        deterministic_seed (Optional[int], optional):
            Seed for deterministic behavior in CUDA kernels (default: None).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        centralization: float = 0.5,
        normalization: float = 0.5,
        normalize_channels: bool = True,
        amp_fac: float = 2.0,
        clip_lambda: Optional[Callable[[int], float]] = lambda step: step**0.25,
        decouple_weight_decay: bool = False,
        clip_gradients: bool = False,
        stochastic_threshold: float = 1e-6,
        use_stochastic_rounding: bool = True,
        stochastic_rounding_prob: float = 0.5,
        stochastic_rounding_magnitude: Optional[float] = None,
        use_bit_manipulation: bool = True,
        permutation_strategy: str = "global",
        deterministic_seed: Optional[int] = None,
    ):
        if deterministic_seed is not None:
            print(f"[SPARKLES] deterministic_seed set to: {deterministic_seed}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
            normalization=normalization,
            normalize_channels=normalize_channels,
            amp_fac=amp_fac,
            clip_lambda=clip_lambda,
            decouple_weight_decay=decouple_weight_decay,
            clip_gradients=clip_gradients,
            stochastic_threshold=stochastic_threshold,
            use_stochastic_rounding=use_stochastic_rounding,
            stochastic_rounding_prob=stochastic_rounding_prob,
            stochastic_rounding_magnitude=stochastic_rounding_magnitude,
            use_bit_manipulation=use_bit_manipulation,
            permutation_strategy=permutation_strategy,
            deterministic_seed=deterministic_seed,
        )
        super(SPARKLES, self).__init__(params, defaults)

    def normalize_gradient(
        self,
        x: torch.Tensor,
        use_channels: bool = False,
        alpha: float = 1.0,
        epsilon: float = 1e-8,
    ) -> None:
        """
        Normalizes the input gradient tensor using its standard deviation, optionally channel-wise.

        This function interpolates between the original gradient and its normalized version (divided by its 
        standard deviation), controlled by the parameter `alpha`. If `use_channels` is True and the tensor has
        more than one dimension, normalization is performed across all dimensions except the first (typically
        the batch dimension), provided all reduction dimensions have more than one element.
        Otherwise, normalization is performed globally if the tensor has more than two elements.

        Args:
            x (torch.Tensor): The gradient tensor to normalize (in-place).
            use_channels (bool, optional): If True, normalize per-channel (across all but the first dimension). 
            Default: False.
            alpha (float, optional): Interpolation weight between the original and normalized gradient.
            Default: 1.0.
            epsilon (float, optional): Small value added to the standard deviation for numerical stability.
            Default: 1e-8.
        """
        size: int = x.dim()
        if size > 1 and use_channels:
            reduce_dims = tuple(range(1, size))
            # Only normalize if all reduction dims have more than 1 element
            if all(x.size(d) > 1 for d in reduce_dims):
                s = x.std(dim=reduce_dims, keepdim=True).add_(epsilon)
                x.lerp_(x.div_(s), weight=alpha)
        elif torch.numel(x) > 2:
            s = x.std().add_(epsilon)
            x.lerp_(x.div_(s), weight=alpha)

    def apply_global_permutation(self, x: torch.Tensor) -> None:
        """Apply global permutation - shuffles all elements randomly.

        :param x: torch.Tensor. Tensor to permute.
        """
        if x.dim() > 1:
            # For ND tensors (N>1), shuffle along first dimension
            perm_idx = torch.randperm(x.size(0), device=x.device)
            x.copy_(x[perm_idx])
        else:
            # For 1D tensors, permute all elements
            perm_idx = torch.randperm(x.numel(), device=x.device)
            x.copy_(x[perm_idx])

    def apply_stochastic_operator(
        self,
        x: torch.Tensor,
        strategy: str = "global",
    ) -> None:
        r"""Apply stochastic operator R to tensor x.
        Only 'global' and 'none' strategies are supported.

        Args:
            x (torch.Tensor): Tensor to apply stochastic operator to.
            strategy (str): Either 'global' (randomly permute all elements) or 'none' (no permutation).
        """
        if strategy == "global":
            self.apply_global_permutation(x)
        elif strategy == "none":
            pass
        else:
            raise ValueError(
                f"Unknown permutation strategy: {strategy}. "
                f"Must be one of 'global', 'none'."
            )

    def apply_stochastic_bf16_rounding(
        self,
        x: torch.Tensor,
        probability: float = 0.5,
        magnitude: Optional[float] = None,
        use_bit_manipulation: bool = True,
    ) -> torch.Tensor:
        """Apply stochastic rounding during conversion to bfloat16.

        This adds controlled noise to the conversion process to improve training dynamics.
        The noise is applied with probability and magnitude controlled by parameters.

        Args:
            x: Input tensor to convert
            probability: Probability of applying noise to each element
            magnitude: Magnitude of noise to apply (defaults to BF16 epsilon)
            use_bit_manipulation: Whether to use bit manipulation for stochastic rounding

        Returns:
            Tensor converted to BF16 with stochastic rounding applied
        """
        if use_bit_manipulation:
            # Use bit manipulation method (Lode's approach)
            if x.dtype == torch.bfloat16:
                # Already BF16, convert to float32 temporarily for bit manipulation
                x_fp32 = x.to(torch.float32)
                result = torch.empty_like(x)
                cuda_copy_stochastic_bf16_(result, x_fp32)
                return result
            else:
                # Create new tensor in BF16 format
                result = torch.empty_like(x, dtype=torch.bfloat16)
                cuda_copy_stochastic_bf16_(result, x)
                return result
        else:
            # Use the original noise-based method
            # Convert to BF16 first
            if not x.is_cuda:
                x_bf16 = x.to(torch.bfloat16)
                if magnitude is None:
                    magnitude = torch.finfo(torch.bfloat16).eps
                rand_mask = torch.rand_like(x, device=x.device) < probability
                sign_mask = torch.rand_like(x, device=x.device) < 0.5
                noise = torch.ones_like(x, device=x.device)
                noise = torch.where(sign_mask, noise, -noise)
                noise = noise * magnitude
                noise = torch.where(rand_mask, noise, torch.zeros_like(x, device=x.device))
                return x_bf16 + noise.to(torch.bfloat16)
            else:
                # Use CUDA kernel for stochastic BF16 rounding
                if magnitude is None:
                    magnitude = float(torch.finfo(torch.bfloat16).eps)
                result = torch.empty_like(x, dtype=torch.bfloat16, device=x.device)
                # Use a random seed for non-deterministic mode
                seed = torch.randint(0, 2**31, ()).item()
                cuda_stochastic_bf16_rounding_(result, x, float(probability), float(magnitude), int(seed))
                return result

    def _is_foreach_compatible(self, group):
        # Check if we've already cached the result
        if "foreach_compat" in group:
            return group["foreach_compat"]
        # Find the first param with a dense grad
        for p in group["params"]:
            if p.grad is not None and not p.grad.is_sparse:
                device = p.device
                dtype = p.dtype
                break
        else:
            group["foreach_compat"] = False
            return False  # No compatible params
        # Check the rest for compatibility
        for p in group["params"]:
            if p.grad is not None and not p.grad.is_sparse:
                if p.device != device or p.dtype != dtype:
                    group["foreach_compat"] = False
                    return False
        group["foreach_compat"] = True
        return True

    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step, using foreach if possible."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            normalization = group["normalization"]
            normalize_channels = group["normalize_channels"]
            amp_fac = group["amp_fac"]
            clip_lambda = group["clip_lambda"]
            decouple_weight_decay = group["decouple_weight_decay"]
            clip_gradients = group["clip_gradients"]
            stochastic_threshold = group["stochastic_threshold"]
            use_stochastic_rounding = group["use_stochastic_rounding"]
            stochastic_rounding_prob = group["stochastic_rounding_prob"]
            stochastic_rounding_magnitude = group["stochastic_rounding_magnitude"]
            use_bit_manipulation = group["use_bit_manipulation"]
            permutation_strategy = group["permutation_strategy"]
            deterministic_seed = group["deterministic_seed"]

            # Determine foreach compatibility
            foreach = group.get("foreach", None)
            if foreach is None:
                foreach = self._is_foreach_compatible(group)
                group["foreach"] = foreach

            # Collect compatible tensors for foreach
            active_p, grads, emas, ema_sqs, prev_grads, states = [], [], [], [], [], []
            for p in group["params"]:
                if p.grad is None or p.grad.is_sparse:
                    continue
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(p.data)
                    state["ema_squared"] = torch.zeros_like(p.data)
                    state["prev_grad"] = torch.zeros_like(p.data)
                    state["stochastic_applied"] = False
                active_p.append(p)
                grads.append(p.grad.data)
                emas.append(state["ema"])
                ema_sqs.append(state["ema_squared"])
                prev_grads.append(state["prev_grad"])
                states.append(state)

            if foreach and len(active_p) > 0:
                # Only support foreach for float32 or float16 on CUDA/CPU for now
                # (BF16, stochastic, and CUDA fused logic must remain per-param)
                # Center and normalize gradients
                if centralization != 0:
                    means = [g.mean(dim=tuple(range(1, g.dim())), keepdim=True) * centralization for g in grads]
                    torch._foreach_sub_(grads, means)
                if normalization != 0:
                    # No foreach for normalize_gradient, fallback to per-tensor
                    for g in grads:
                        self.normalize_gradient(g, use_channels=normalize_channels, alpha=normalization)
                # Bias correction
                steps = [s["step"] + 1 for s in states]
                beta1, beta2 = betas
                bias_correction = [1 - beta1 ** step for step in steps]
                bias_correction_sqrt = [(1 - beta2 ** step) ** 0.5 for step in steps]
                step_sizes = [lr / bc for bc in bias_correction]
                # Update EMA and EMA2
                torch._foreach_mul_(emas, beta1)
                torch._foreach_add_(emas, grads, alpha=1 - beta1)
                # Amplify gradient with EMA
                torch._foreach_add_(grads, emas, alpha=amp_fac)
                torch._foreach_mul_(ema_sqs, beta2)
                torch._foreach_addcmul_(ema_sqs, grads, grads, value=1 - beta2)
                # Compute denom
                denoms = [es.sqrt().div(bcs).add(eps) for es, bcs in zip(ema_sqs, bias_correction_sqrt)]
                # Prepare update
                updates = []
                for i, (g, d, p) in enumerate(zip(grads, denoms, active_p)):
                    if decouple_weight_decay and weight_decay != 0:
                        p.data.mul_(1 - step_sizes[i] * weight_decay)
                        update = g.div(d).mul(step_sizes[i])
                    else:
                        if weight_decay != 0:
                            g.add_(p.data, alpha=weight_decay)
                        update = g.div(d).mul(step_sizes[i])
                    # Stochastic operator (per-param)
                    if steps[i] > 1:
                        grad_diff_norm = torch.norm(g - prev_grads[i])
                        if grad_diff_norm < stochastic_threshold:
                            self.apply_stochastic_operator(update, strategy=permutation_strategy)
                            states[i]["stochastic_applied"] = True
                        else:
                            states[i]["stochastic_applied"] = False
                    # Gradient clipping
                    if clip_gradients and clip_lambda is not None:
                        clip = clip_lambda(steps[i])
                        update.clamp_(-clip, clip)
                    updates.append(update)
                # Update parameters
                for i, p in enumerate(active_p):
                    p.data.sub_(updates[i])
                    # Update prev_grad
                    prev_grads[i].copy_(grads[i])
                    # Update state step
                    states[i]["step"] = steps[i]
                # Update state tensors
                for i, s in enumerate(states):
                    s["ema"] = emas[i]
                    s["ema_squared"] = ema_sqs[i]
                    s["prev_grad"] = prev_grads[i]
            else:
                # Fallback to per-parameter logic (original code)
                for i, p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError("SPARKLES does not support sparse gradients")
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["ema"] = torch.zeros_like(p.data)
                        state["ema_squared"] = torch.zeros_like(p.data)
                        state["prev_grad"] = torch.zeros_like(p.data)
                        state["stochastic_applied"] = False
                    ema, ema_squared = state["ema"], state["ema_squared"]
                    prev_grad = state["prev_grad"]
                    beta1, beta2 = betas
                    state["step"] += 1
                    grad_fp32 = grad.to(torch.float32)
                    p_fp32 = (
                        p.data.to(torch.float32)
                        if p.data.dtype == torch.bfloat16
                        else p.data.clone()
                    )
                    ema_fp32 = (
                        ema.to(torch.float32)
                        if ema.dtype == torch.bfloat16
                        else ema.clone()
                    )
                    ema_squared_fp32 = (
                        ema_squared.to(torch.float32)
                        if ema_squared.dtype == torch.bfloat16
                        else ema_squared.clone()
                    )
                    prev_grad_fp32 = (
                        prev_grad.to(torch.float32)
                        if prev_grad.dtype == torch.bfloat16
                        else prev_grad.clone()
                    )
                    if centralization != 0:
                        grad_fp32.sub_(
                            grad_fp32.mean(
                                dim=tuple(range(1, grad_fp32.dim())), keepdim=True
                            ).mul_(centralization)
                        )
                    if normalization != 0:
                        self.normalize_gradient(
                            grad_fp32, use_channels=normalize_channels, alpha=normalization
                        )
                    bias_correction = 1 - beta1 ** state["step"]
                    bias_correction_sqrt = (1 - beta2 ** state["step"]) ** 0.5
                    step_size = lr / bias_correction
                    ema_fp32.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)
                    grad_fp32.add_(ema_fp32, alpha=amp_fac)
                    ema_squared_fp32.mul_(beta2).addcmul_(
                        grad_fp32, grad_fp32, value=1 - beta2
                    )
                    denom = ema_squared_fp32.sqrt().div_(bias_correction_sqrt).add_(eps)
                    if decouple_weight_decay and weight_decay != 0:
                        p_fp32.mul_(1 - step_size * weight_decay)
                        update = grad_fp32.div(denom).mul(step_size)
                    else:
                        if weight_decay != 0:
                            grad_fp32.add_(p_fp32, alpha=weight_decay)
                        update = grad_fp32.div(denom).mul(step_size)
                    if state["step"] > 1:
                        grad_diff_norm = torch.norm(grad_fp32 - prev_grad_fp32)
                        if grad_diff_norm < stochastic_threshold:
                            self.apply_stochastic_operator(
                                update,
                                strategy=permutation_strategy,
                            )
                            state["stochastic_applied"] = True
                        else:
                            state["stochastic_applied"] = False
                    if clip_gradients and clip_lambda is not None:
                        clip = clip_lambda(state["step"])
                        update.clamp_(-clip, clip)
                    if deterministic_seed is not None:
                        param_seed = int(deterministic_seed) + state["step"] + hash(p) % 100000
                    else:
                        param_seed = torch.randint(0, 2**31, ()).item()
                    if (
                        p.dtype == torch.bfloat16 and p.is_cuda and
                        ema.dtype == torch.bfloat16 and ema.is_cuda and
                        ema_squared.dtype == torch.bfloat16 and ema_squared.is_cuda and
                        grad.dtype == torch.float32 and grad.is_cuda
                    ):
                        cuda_fused_optimizer(p.data, ema, ema_squared, grad, lr, beta1, beta2, param_seed)
                        if prev_grad.dtype == torch.bfloat16 and prev_grad.is_cuda:
                            if use_stochastic_rounding and use_bit_manipulation:
                                cuda_copy_stochastic_bf16_(prev_grad, grad_fp32, param_seed)
                            else:
                                prev_grad.copy_(self.apply_stochastic_bf16_rounding(
                                    grad_fp32,
                                    probability=stochastic_rounding_prob,
                                    magnitude=stochastic_rounding_magnitude,
                                    use_bit_manipulation=False,
                                ))
                        else:
                            prev_grad.copy_(grad_fp32)
                        continue
                    else:
                        p_fp32.sub_(update)
                    if p.dtype == torch.bfloat16:
                        if use_stochastic_rounding:
                            if use_bit_manipulation:
                                cuda_copy_stochastic_bf16_(p.data, p_fp32, param_seed)
                            else:
                                p.data.copy_(
                                    self.apply_stochastic_bf16_rounding(
                                        p_fp32,
                                        probability=stochastic_rounding_prob,
                                        magnitude=stochastic_rounding_magnitude,
                                        use_bit_manipulation=False,
                                    )
                                )
                        else:
                            p.data.copy_(p_fp32.to(torch.bfloat16))
                    else:
                        p.data.copy_(p_fp32)
                    if prev_grad.dtype == torch.bfloat16:
                        if use_stochastic_rounding:
                            if use_bit_manipulation:
                                cuda_copy_stochastic_bf16_(prev_grad, grad_fp32, param_seed)
                            else:
                                prev_grad.copy_(
                                    self.apply_stochastic_bf16_rounding(
                                        grad_fp32,
                                        probability=stochastic_rounding_prob,
                                        magnitude=stochastic_rounding_magnitude,
                                        use_bit_manipulation=False,
                                    )
                                )
                        else:
                            prev_grad.copy_(grad_fp32.to(torch.bfloat16))
                    else:
                        prev_grad.copy_(grad_fp32)
                    if ema.dtype == torch.bfloat16:
                        if use_stochastic_rounding:
                            if use_bit_manipulation:
                                cuda_copy_stochastic_bf16_(ema, ema_fp32, param_seed)
                            else:
                                ema.copy_(
                                    self.apply_stochastic_bf16_rounding(
                                        ema_fp32,
                                        probability=stochastic_rounding_prob,
                                        magnitude=stochastic_rounding_magnitude,
                                        use_bit_manipulation=False,
                                    )
                                )
                        else:
                            ema.copy_(ema_fp32.to(torch.bfloat16))
                    else:
                        ema.copy_(ema_fp32)
                    if ema_squared.dtype == torch.bfloat16:
                        if use_stochastic_rounding:
                            if use_bit_manipulation:
                                cuda_copy_stochastic_bf16_(ema_squared, ema_squared_fp32, param_seed)
                            else:
                                ema_squared.copy_(
                                    self.apply_stochastic_bf16_rounding(
                                        ema_squared_fp32,
                                        probability=stochastic_rounding_prob,
                                        magnitude=stochastic_rounding_magnitude,
                                        use_bit_manipulation=False,
                                    )
                                )
                        else:
                            ema_squared.copy_(ema_squared_fp32.to(torch.bfloat16))
                    else:
                        ema_squared.copy_(ema_squared_fp32)
        return loss
