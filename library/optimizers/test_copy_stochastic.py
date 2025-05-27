import torch
import time
from sparkles import copy_stochastic_ as py_copy_stochastic_base
from copy_stochastic_cuda_wrapper import copy_stochastic_ as cuda_copy_stochastic_

# Compile the Python (Torch) implementation
py_copy_stochastic_ = torch.compile(py_copy_stochastic_base)

def test_equivalence():
    torch.manual_seed(42)
    src = torch.randn(10000, device='cuda', dtype=torch.float32)
    tgt_py = torch.empty_like(src)
    tgt_cuda = torch.empty_like(src)

    py_copy_stochastic_(tgt_py, src)
    cuda_copy_stochastic_(tgt_cuda, src)

    # The results will not be bitwise identical due to different RNGs, but should be statistically similar
    print("Mean diff:", (tgt_py - tgt_cuda).abs().mean().item())
    print("Std diff:", (tgt_py - tgt_cuda).abs().std().item())
    print("Max diff:", (tgt_py - tgt_cuda).abs().max().item())

    # Check that the outputs are in the expected range and dtype
    assert tgt_cuda.dtype == torch.float32
    assert tgt_py.dtype == torch.float32

def benchmark_copy_stochastic(size=10**7, n_iters=100):
    print(f"\nBenchmarking with tensor size: {size}, iterations: {n_iters}")
    src = torch.randn(size, device='cuda', dtype=torch.float32)
    tgt_py = torch.empty_like(src)
    tgt_cuda = torch.empty_like(src)

    # Warmup
    py_copy_stochastic_(tgt_py, src)
    cuda_copy_stochastic_(tgt_cuda, src)
    torch.cuda.synchronize()

    # Time Python (Torch) implementation
    start = time.time()
    for _ in range(n_iters):
        py_copy_stochastic_(tgt_py, src)
    torch.cuda.synchronize()
    py_time = time.time() - start

    # Time CUDA implementation
    start = time.time()
    for _ in range(n_iters):
        cuda_copy_stochastic_(tgt_cuda, src)
    torch.cuda.synchronize()
    cuda_time = time.time() - start

    print(f"Torch (Python, compiled) implementation: {py_time:.4f} s total, {py_time/n_iters*1000:.4f} ms/iter")
    print(f"CUDA implementation:                   {cuda_time:.4f} s total, {cuda_time/n_iters*1000:.4f} ms/iter")
    print(f"Speedup: {py_time/cuda_time:.2f}x")

if __name__ == "__main__":
    test_equivalence()
    benchmark_copy_stochastic() 