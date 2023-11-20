import torch


def test_cuda_availability():
    assert torch.cuda.is_available(), "CUDA is not available"


def test_current_device():
    assert torch.cuda.current_device() == 0, "CUDA device is not 0"


def test_clip_imports():
    try:
        import ftfy
    except ImportError:
        assert False, "ftfy is not installed"
    try:
        import regex
    except ImportError:
        assert False, "regex is not installed"
