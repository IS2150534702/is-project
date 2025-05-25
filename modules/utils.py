import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def device_to_normalized_form(device: torch.device) -> str:
    return f'{device.type}:{device.index}' if device.index is not None else device.type

def str_to_dtype(dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    elif dtype == "fp16":
        return torch.float16
    elif dtype == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
