import torch
import time
from copy_stochastic_cuda_wrapper import copy_stochastic_bf16_ as cuda_copy_stochastic_bf16_

def test_bfloat16_vs_pytorch():
    torch.manual_seed(42)
    src = torch.randn(10000, device='cuda', dtype=torch.float32)
    tgt_pt = src.to(torch.bfloat16)
    tgt_cuda = torch.empty_like(src, dtype=torch.bfloat16)

    # CUDA bfloat16 kernel
    cuda_copy_stochastic_bf16_(tgt_cuda, src)

    print("[BF16] Mean diff (CUDA vs PyTorch):", (tgt_pt.float() - tgt_cuda.float()).abs().mean().item())
    print("[BF16] Std diff (CUDA vs PyTorch):", (tgt_pt.float() - tgt_cuda.float()).abs().std().item())
    print("[BF16] Max diff (CUDA vs PyTorch):", (tgt_pt.float() - tgt_cuda.float()).abs().max().item())
    assert tgt_cuda.dtype == torch.bfloat16
    assert tgt_pt.dtype == torch.bfloat16

def benchmark_copy_stochastic_bf16(size=10**7, n_iters=100):
    print(f"\nBenchmarking BF16 with tensor size: {size}, iterations: {n_iters}")
    src = torch.randn(size, device='cuda', dtype=torch.float32)
    tgt_cuda = torch.empty_like(src, dtype=torch.bfloat16)

    # Warmup
    cuda_copy_stochastic_bf16_(tgt_cuda, src)
    torch.cuda.synchronize()

    # Time CUDA bfloat16 implementation
    start = time.time()
    for _ in range(n_iters):
        cuda_copy_stochastic_bf16_(tgt_cuda, src)
    torch.cuda.synchronize()
    cuda_time = time.time() - start

    print(f"CUDA bfloat16 implementation: {cuda_time:.4f} s total, {cuda_time/n_iters*1000:.4f} ms/iter")

def benchmark_pytorch_bfloat16(size=10**7, n_iters=100):
    print(f"\nBenchmarking PyTorch bfloat16 with tensor size: {size}, iterations: {n_iters}")
    src = torch.randn(size, device='cuda', dtype=torch.float32)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        tgt_pt = src.to(torch.bfloat16)
    torch.cuda.synchronize()
    pt_time = time.time() - start
    print(f"PyTorch bfloat16 implementation: {pt_time:.4f} s total, {pt_time/n_iters*1000:.4f} ms/iter")

if __name__ == "__main__":
    test_bfloat16_vs_pytorch()
    benchmark_copy_stochastic_bf16()
    benchmark_pytorch_bfloat16() 