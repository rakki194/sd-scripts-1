import torch
import importlib
import sys
import pathlib

try:
    copy_stochastic_cuda = importlib.import_module('copy_stochastic_cuda')
except ImportError:
    ext_path = pathlib.Path(__file__).parent
    if not (ext_path / 'copy_stochastic_cuda.so').exists():
        import subprocess
        subprocess.check_call(
            [sys.executable, 'setup.py', 'build_ext', '--inplace'],
            cwd=ext_path
        )
    copy_stochastic_cuda = importlib.import_module('copy_stochastic_cuda')

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """
    CUDA version of copy_stochastic_.
    """
    if not target.is_cuda or not source.is_cuda:
        raise ValueError("Both tensors must be CUDA tensors")
    if target.dtype != torch.float32 or source.dtype != torch.float32:
        raise ValueError("Both tensors must be float32")
    copy_stochastic_cuda.copy_stochastic_cuda(target, source)

# New: bfloat16 wrapper
def copy_stochastic_bf16_(target: torch.Tensor, source: torch.Tensor):
    """
    CUDA version of copy_stochastic_ for bfloat16 target and float32 source.
    """
    if not target.is_cuda or not source.is_cuda:
        raise ValueError("Both tensors must be CUDA tensors")
    if target.dtype != torch.bfloat16 or source.dtype != torch.float32:
        raise ValueError("Target must be bfloat16 and source must be float32")
    copy_stochastic_cuda.copy_stochastic_bf16_cuda(target, source)

def fused_optimizer(param, ema, ema2, grad, lr, ema_beta, ema2_beta):
    """
    Fused CUDA kernel for param, ema, ema2 update with stochastic bfloat16 rounding.
    All tensors must be CUDA, param/ema/ema2 bfloat16, grad float32.
    """
    if not (param.is_cuda and ema.is_cuda and ema2.is_cuda and grad.is_cuda):
        raise ValueError("All tensors must be CUDA tensors")
    if param.dtype != torch.bfloat16 or ema.dtype != torch.bfloat16 or ema2.dtype != torch.bfloat16:
        raise ValueError("param, ema, ema2 must be bfloat16")
    if grad.dtype != torch.float32:
        raise ValueError("grad must be float32")
    copy_stochastic_cuda.fused_optimizer(param, ema, ema2, grad, float(lr), float(ema_beta), float(ema2_beta)) 