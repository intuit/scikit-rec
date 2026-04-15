from typing import Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


def select_torch_device(device: Optional[str] = None) -> "torch.device":
    """Resolve and validate a PyTorch device string.

    When ``device`` is ``None``, uses CUDA if available, otherwise CPU. Apple
    MPS is not selected automatically; pass ``device="mps"`` explicitly when
    you want it. When a string is provided, validates it eagerly so that typos
    like ``"gpu"`` fail fast with a clear message rather than a cryptic error
    at forward-pass time.

    Args:
        device: Device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``,
            etc.) or ``None`` for CUDA-if-available else CPU.

    Returns:
        A ``torch.device`` instance.

    Raises:
        ValueError: If the device string is not recognised by PyTorch.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    try:
        return torch.device(device)
    except RuntimeError:
        raise ValueError(f"Invalid device '{device}'. Expected 'cpu', 'cuda', 'cuda:<N>', or 'mps'.") from None
