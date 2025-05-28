import torch
from library.optimizers import copy_stochastic_cuda_wrapper

def python_fused_update(param, ema, ema2, grad, lr, beta1, beta2, seed):
    # Clone to avoid in-place modification
    param = param.clone()
    ema = ema.clone()
    ema2 = ema2.clone()
    grad = grad.clone()
    torch.manual_seed(seed)
    # Convert to float32 for math
    p = param.to(torch.float32)
    e = ema.to(torch.float32)
    e2 = ema2.to(torch.float32)
    g = grad.to(torch.float32)
    # Update param
    new_p = p - lr * g
    # Stochastic rounding (bit manipulation, as in your reference)
    rand16 = torch.randint(0, 1 << 16, new_p.shape, dtype=torch.int32, device=new_p.device)
    new_p_bits = new_p.view(torch.int32) + rand16
    new_p_bits = new_p_bits & -65536
    rounded_p = new_p_bits.view(torch.float32)
    param_out = rounded_p.to(torch.bfloat16)
    # EMA update
    new_e = beta1 * e + (1.0 - beta1) * rounded_p
    ema_out = new_e.to(torch.bfloat16)
    # EMA2 update
    new_e2 = beta2 * e2 + (1.0 - beta2) * (g * g)
    ema2_out = new_e2.to(torch.bfloat16)
    return param_out, ema_out, ema2_out

def test_fused_optimizer_vec4_vs_python():
    torch.manual_seed(42)
    N = 1024
    param = torch.randn(N, device='cuda', dtype=torch.bfloat16)
    ema = torch.randn(N, device='cuda', dtype=torch.bfloat16)
    ema2 = torch.randn(N, device='cuda', dtype=torch.bfloat16)
    grad = torch.randn(N, device='cuda', dtype=torch.float32)
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    seed = 12345

    # CUDA kernel (in-place)
    param_cuda = param.clone()
    ema_cuda = ema.clone()
    ema2_cuda = ema2.clone()
    copy_stochastic_cuda_wrapper.fused_optimizer_vec4(
        param_cuda, ema_cuda, ema2_cuda, grad, lr, beta1, beta2, seed
    )

    # Python reference
    param_py, ema_py, ema2_py = python_fused_update(param, ema, ema2, grad, lr, beta1, beta2, seed)

    # Compare (allowing for stochasticity, but should be very close if seed is fixed)
    print("Max abs diff param:", (param_cuda - param_py).abs().max().item())
    print("Max abs diff ema:", (ema_cuda - ema_py).abs().max().item())
    print("Max abs diff ema2:", (ema2_cuda - ema2_py).abs().max().item())
    assert (param_cuda - param_py).abs().max() <= 2 * torch.finfo(torch.bfloat16).eps
    assert torch.allclose(ema_cuda, ema_py, atol=1e-2)
    assert torch.allclose(ema2_cuda, ema2_py, atol=1e-2)
    print("Test passed!")

def test_normalize_gradient_cuda():
    print("Testing normalize_gradient_cuda_ (global)...")
    x = torch.randn(1024, device='cuda', dtype=torch.float32)
    x_py = x.clone()
    x_cuda = x.clone()
    # Python version
    s = x_py.std().add(1e-8)
    x_py.lerp_(x_py / s, weight=0.5)
    # CUDA version
    copy_stochastic_cuda_wrapper.normalize_gradient_cuda_(x_cuda, use_channels=False, alpha=0.5, epsilon=1e-8)
    print("Max abs diff (global):", (x_py - x_cuda).abs().max().item())
    assert torch.allclose(x_py, x_cuda, atol=1e-5)
    print("Passed global normalization test.")

    print("Testing normalize_gradient_cuda_ (channel-wise)...")
    x2 = torch.randn(32, 16, device='cuda', dtype=torch.float32)
    x2_py = x2.clone()
    x2_cuda = x2.clone()
    s2 = x2_py.std(dim=0, keepdim=True).add(1e-8)
    x2_py.lerp_(x2_py / s2, weight=0.5)
    copy_stochastic_cuda_wrapper.normalize_gradient_cuda_(x2_cuda, use_channels=True, alpha=0.5, epsilon=1e-8)
    print("Max abs diff (channel):", (x2_py - x2_cuda).abs().max().item())
    assert torch.allclose(x2_py, x2_cuda, atol=1e-5)
    print("Passed channel-wise normalization test.")

def test_global_permutation_cuda():
    print("Testing global_permutation_cuda_...")
    x = torch.arange(100, device='cuda', dtype=torch.float32)
    x_py = x.clone()
    x_cuda = x.clone()
    seed = 123
    perm = torch.randperm(x_py.numel(), device=x_py.device, generator=torch.Generator().manual_seed(seed))
    x_py.copy_(x_py[perm])
    copy_stochastic_cuda_wrapper.global_permutation_cuda_(x_cuda, seed=seed)
    print("Max abs diff (perm):", (x_py - x_cuda).abs().max().item())
    assert torch.allclose(x_py, x_cuda)
    print("Passed global permutation test.")

if __name__ == '__main__':
    test_fused_optimizer_vec4_vs_python()
    test_normalize_gradient_cuda()
    test_global_permutation_cuda()