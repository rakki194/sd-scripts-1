import torch
import importlib
import sys
import pathlib

try:
    copy_stochastic_cuda = importlib.import_module('library.optimizers.copy_stochastic_cuda')
except ImportError:
    ext_path = pathlib.Path(__file__).parent
    if not (ext_path / 'copy_stochastic_cuda.so').exists():
        import subprocess
        subprocess.check_call(
            [sys.executable, 'setup.py', 'build_ext', '--inplace'],
            cwd=ext_path
        )
    copy_stochastic_cuda = importlib.import_module('library.optimizers.copy_stochastic_cuda')

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor, seed: int = None):
    """
    Perform stochastic copy of float32 CUDA tensors with optional random seed.

    Args:
        target (torch.Tensor): Output tensor (float32, CUDA).
        source (torch.Tensor): Input tensor (float32, CUDA).
        seed (int, optional): Random seed for noise generation. If None, a random seed is used.
    """
    #if not target.is_cuda or not source.is_cuda:
    #    raise ValueError("Both tensors must be CUDA tensors")
    #if target.dtype != torch.float32 or source.dtype != torch.float32:
    #    raise ValueError("Both tensors must be float32")
    #print(f"[copy_stochastic_] Called with seed: {seed}")
    copy_stochastic_cuda.copy_stochastic_cuda(target, source, int(seed))

# New: bfloat16 wrapper
def copy_stochastic_bf16_(target: torch.Tensor, source: torch.Tensor, seed: int):
    """
    Perform stochastic copy from float32 to bfloat16 CUDA tensors with required random seed.

    Args:
        target (torch.Tensor): Output tensor (bfloat16, CUDA).
        source (torch.Tensor): Input tensor (float32, CUDA).
        seed (int): Random seed for noise generation.
    """
    #if not target.is_cuda or not source.is_cuda:
    #    raise ValueError("Both tensors must be CUDA tensors")
    #if target.dtype != torch.bfloat16 or source.dtype != torch.float32:
    #    raise ValueError("Target must be bfloat16 and source must be float32")

    #print(f"[copy_stochastic_bf16_] Called with seed: {seed}")
    copy_stochastic_cuda.copy_stochastic_bf16_cuda(target, source, int(seed))

def fused_optimizer(param, ema, ema2, grad, lr, ema_beta, ema2_beta, seed: int):
    """
    Fused CUDA kernel for parameter, EMA, and EMA2 update with stochastic bfloat16 rounding.

    Args:
        param (torch.Tensor): Parameter tensor (bfloat16, CUDA).
        ema (torch.Tensor): Exponential moving average tensor (bfloat16, CUDA).
        ema2 (torch.Tensor): Exponential moving average of squared gradients (bfloat16, CUDA).
        grad (torch.Tensor): Gradient tensor (float32, CUDA).
        lr (float): Learning rate.
        ema_beta (float): Decay rate for EMA.
        ema2_beta (float): Decay rate for EMA2.
        seed (int): Random seed for stochastic rounding.
    """
    #if not (param.is_cuda and ema.is_cuda and ema2.is_cuda and grad.is_cuda):
    #    raise ValueError("All tensors must be CUDA tensors")
    #if param.dtype != torch.bfloat16 or ema.dtype != torch.bfloat16 or ema2.dtype != torch.bfloat16:
    #    raise ValueError("param, ema, ema2 must be bfloat16")
    #if grad.dtype != torch.float32:
    #    raise ValueError("grad must be float32")

    #print(f"[fused_optimizer] Called with seed: {seed}")
    copy_stochastic_cuda.fused_optimizer(param, ema, ema2, grad, float(lr), float(ema_beta), float(ema2_beta), int(seed))

def stochastic_bf16_rounding_(target: torch.Tensor, source: torch.Tensor, probability: float, magnitude: float, seed: int):
    """
    Perform stochastic BF16 rounding on CUDA tensors with configurable probability and magnitude.

    Args:
        target (torch.Tensor): Output tensor (bfloat16, CUDA).
        source (torch.Tensor): Input tensor (float32, CUDA).
        probability (float): Probability of applying noise to each element.
        magnitude (float): Magnitude of noise to apply.
        seed (int): Random seed for noise generation.
    """
    #if not target.is_cuda or not source.is_cuda:
    #    raise ValueError("Both tensors must be CUDA tensors")
    #if target.dtype != torch.bfloat16 or source.dtype != torch.float32:
    #    raise ValueError("Target must be bfloat16 and source must be float32")
    copy_stochastic_cuda.stochastic_bf16_rounding_cuda(target, source, float(probability), float(magnitude), int(seed))