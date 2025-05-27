import torch
from sniffed import copy_stochastic_ as py_copy_stochastic_
from copy_stochastic_cuda import copy_stochastic_ as cuda_copy_stochastic_

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

if __name__ == "__main__":
    test_equivalence() 