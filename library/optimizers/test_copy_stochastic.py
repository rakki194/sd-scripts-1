import torch
import time
from copy_stochastic_cuda_wrapper import copy_stochastic_bf16_ as cuda_copy_stochastic_bf16_
from sparkles import copy_stochastic_ as py_copy_stochastic_

def check_no_unexpected_nan_inf(input_tensor, output_tensor, name):
    # Only allow NaN in output where input was NaN
    input_nan = torch.isnan(input_tensor)
    output_nan = torch.isnan(output_tensor)
    if not torch.equal(input_nan, output_nan):
        raise AssertionError(f"{name} contains unexpected NaN")
    # Allow Inf in output if input was Inf, or if input was very large finite
    input_inf = torch.isinf(input_tensor)
    output_inf = torch.isinf(output_tensor)
    unexpected_inf = output_inf & ~input_inf
    if unexpected_inf.any():
        bfloat16_max = float(torch.finfo(torch.bfloat16).max)
        large_input = (input_tensor.abs() > (bfloat16_max * 0.99))
        if not torch.all(large_input[unexpected_inf]):
            raise AssertionError(f"{name} contains unexpected Inf (not due to large input)")

def compare_bitwise(a, b, name=""):
    a_bits = a.view(torch.int32)
    b_bits = b.view(torch.int32)
    diff = (a_bits != b_bits).sum().item()
    print(f"{name} Bitwise diff count: {diff} / {a.numel()}")

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

def test_edge_cases():
    print("Testing edge cases...")
    edge_cases = [
        0.0, -0.0, 1.0, -1.0, 1e-30, -1e-30, 1e30, -1e30,
        float('inf'), float('-inf'), float('nan'),
        torch.finfo(torch.float32).max, torch.finfo(torch.float32).min,
        torch.finfo(torch.float32).tiny, -torch.finfo(torch.float32).tiny,
    ]
    src = torch.tensor(edge_cases, device='cuda', dtype=torch.float32)
    tgt_py = torch.empty_like(src, dtype=torch.bfloat16)
    tgt_cuda = torch.empty_like(src, dtype=torch.bfloat16)

    py_copy_stochastic_(tgt_py, src)
    cuda_copy_stochastic_bf16_(tgt_cuda, src)

    print("Python:", tgt_py)
    print("CUDA:  ", tgt_cuda)
    check_no_unexpected_nan_inf(src, tgt_cuda, "CUDA")
    compare_bitwise(tgt_py.float(), tgt_cuda.float(), "Edge cases")

def test_random_large():
    print("Testing large random tensor...")
    src = torch.randn(1000000, device='cuda', dtype=torch.float32) * 1e3
    tgt_py = torch.empty_like(src, dtype=torch.bfloat16)
    tgt_cuda = torch.empty_like(src, dtype=torch.bfloat16)

    py_copy_stochastic_(tgt_py, src)
    cuda_copy_stochastic_bf16_(tgt_cuda, src)

    # Only check for unexpected NaN/Inf
    check_no_unexpected_nan_inf(src, tgt_cuda, "CUDA")
    print("Mean diff:", (tgt_py.float() - tgt_cuda.float()).abs().mean().item())
    print("Std diff:", (tgt_py.float() - tgt_cuda.float()).abs().std().item())
    print("Max diff:", (tgt_py.float() - tgt_cuda.float()).abs().max().item())
    compare_bitwise(tgt_py.float(), tgt_cuda.float(), "Random large")

def test_noncontiguous():
    print("Testing non-contiguous tensors...")
    src = torch.randn(100, 100, device='cuda', dtype=torch.float32).t()
    tgt_py = torch.empty_like(src, dtype=torch.bfloat16)
    tgt_cuda = torch.empty_like(src, dtype=torch.bfloat16)

    py_copy_stochastic_(tgt_py, src)
    cuda_copy_stochastic_bf16_(tgt_cuda, src)

    check_no_unexpected_nan_inf(src, tgt_cuda, "CUDA")
    print("Mean diff:", (tgt_py.float() - tgt_cuda.float()).abs().mean().item())
    compare_bitwise(tgt_py.float(), tgt_cuda.float(), "Noncontiguous")

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
    test_edge_cases()
    test_random_large()
    test_noncontiguous()
    benchmark_copy_stochastic_bf16()
    benchmark_pytorch_bfloat16() 