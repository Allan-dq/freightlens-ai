"""Resolve the compute device for training and inference."""


def resolve_device(name: str) -> str:
    """Return an explicit device; "auto" picks CUDA, then Apple MPS, then CPU."""
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
