import torch
import time
from sparkles import SPARKLES

NUM_STEPS = 100
BATCH_SIZE = 32
D_IN, D_OUT = 2048, 2048

def run_benchmark(model, optimizer, device, desc):
    data = torch.randn(BATCH_SIZE, D_IN, device=device, dtype=model.weight.dtype)
    target = torch.randn(BATCH_SIZE, D_OUT, device=device, dtype=model.weight.dtype)
    step_fn = optimizer.step
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        out = model(data)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        step_fn()
    torch.cuda.synchronize()
    start = time.time()
    last_loss = None
    nan_step = None
    for i in range(NUM_STEPS):
        optimizer.zero_grad()
        out = model(data)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        step_fn()
        last_loss = loss.item()
        # NaN/Inf check
        if torch.isnan(model.weight).any() or torch.isinf(model.weight).any():
            print(f"WARNING: NaN or Inf detected in weights after step {i+1} in {desc}")
            nan_step = i+1
            break
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"{desc:40s} | {elapsed:.4f} sec | Final loss: {last_loss:.6f}")
    if nan_step is not None:
        print(f"{desc:40s} | NaN/Inf detected at step {nan_step}")
    return elapsed, last_loss

def main():
    print(f"Benchmarking SPARKLES fused kernel vs PyTorch-only (no CUDA kernel) on CUDA+bfloat16")
    print(f"{'Description':40s} | Time (s)   | Final Loss")
    print("-"*70)

    # 1. SPARKLES fused kernel (CUDA, bfloat16, use_bit_manipulation=True)
    model_fused = torch.nn.Linear(D_IN, D_OUT, device='cuda', dtype=torch.bfloat16)
    optimizer_fused = SPARKLES(model_fused.parameters(), lr=1e-2, use_bit_manipulation=True)
    run_benchmark(model_fused, optimizer_fused, 'cuda', 'SPARKLES fused kernel (CUDA)')

    # 2. SPARKLES PyTorch-only (CUDA, bfloat16, use_bit_manipulation=False)
    model_py = torch.nn.Linear(D_IN, D_OUT, device='cuda', dtype=torch.bfloat16)
    optimizer_py = SPARKLES(model_py.parameters(), lr=1e-2, use_bit_manipulation=False)
    run_benchmark(model_py, optimizer_py, 'cuda', 'SPARKLES PyTorch-only (no CUDA kernel)')

if __name__ == '__main__':
    main() 