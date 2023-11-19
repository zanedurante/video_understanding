import torch

def test_cuda_availability():
    assert torch.cuda.is_available(), "CUDA is not available"
