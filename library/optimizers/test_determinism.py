import torch
from library.optimizers.copy_stochastic_cuda_wrapper import copy_stochastic_bf16_

def test_copy_stochastic_bf16(seed):
    torch.manual_seed(42)
    src = torch.randn(1000, device='cuda', dtype=torch.float32)
    tgt1 = torch.empty(1000, device='cuda', dtype=torch.bfloat16)
    tgt2 = torch.empty(1000, device='cuda', dtype=torch.bfloat16)

    copy_stochastic_bf16_(tgt1, src, seed)
    copy_stochastic_bf16_(tgt2, src, seed)

    # Compare
    all_close = torch.allclose(tgt1, tgt2)
    max_diff = (tgt1 - tgt2).abs().max().item()
    print(f"All close: {all_close}")
    print(f"Max diff: {max_diff}")
    if not all_close:
        print("Non-deterministic output detected!")
    else:
        print("Deterministic output confirmed.")

if __name__ == "__main__":
    test_copy_stochastic_bf16(seed=12345) 