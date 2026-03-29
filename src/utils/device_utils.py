from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import torch


def resolve_device_ids(device_arg: str | None) -> list[int] | None:
    """Validate a user-provided device spec and return a list of GPU ids.

    Returns ``None`` when *device_arg* is ``None`` (meaning "use framework
    default"), an empty list for CPU, or a list of validated GPU indices.
    """
    if device_arg is None:
        return None

    device_ids = parse_device_ids(device_arg)

    import torch as _torch

    if len(device_ids) > 0 and not _torch.cuda.is_available():
        raise ValueError("CUDA is not available, but GPU device ids were provided.")
    if len(device_ids) == 0:
        return []

    device_count = _torch.cuda.device_count()
    invalid_ids = [idx for idx in device_ids if idx < 0 or idx >= device_count]
    if invalid_ids:
        raise ValueError(
            f"Invalid GPU ids {invalid_ids}. Available GPU ids: 0..{device_count - 1}."
        )
    return device_ids


def parse_device_ids(device_arg: str) -> List[int]:
    value = device_arg.strip().lower()
    if not value:
        raise ValueError("Device argument is empty.")
    if value in {"cpu", "-1"}:
        return []

    device_ids = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            raise ValueError(f"Invalid device list: {device_arg!r}")
        device_ids.append(int(token))
    return device_ids


def build_accelerate_max_memory_map(
    device_ids: Iterable[int],
    free_bytes_by_device: Mapping[int, int],
    reserve_bytes: int = 2 * 1024**3,
) -> Dict[int, str]:
    max_memory: Dict[int, str] = {}
    for device_id in device_ids:
        if device_id not in free_bytes_by_device:
            raise ValueError(f"Missing free memory info for device {device_id}.")
        free_bytes = free_bytes_by_device[device_id]
        usable_gib = max(int((free_bytes - reserve_bytes) / (1024**3)), 4)
        max_memory[device_id] = f"{usable_gib}GiB"
    return max_memory


def maybe_set_cuda_device_from_tensor(tensor) -> None:
    if tensor is None:
        return
    if not torch.cuda.is_available():
        return
    if not getattr(tensor, "is_cuda", False):
        return

    device = getattr(tensor, "device", None)
    device_index = getattr(device, "index", None)
    if device_index is None:
        return
    if torch.cuda.current_device() == device_index:
        return
    torch.cuda.set_device(device_index)
