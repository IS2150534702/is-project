import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def round_up(a: int, b: int) -> int:
    return (a + b - 1) // b
