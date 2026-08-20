from __future__ import annotations

import os
from pathlib import Path


_torch_xla = None
_runtime = None
_xla_model = None
_device = None
_cache_dir = None


def initialize(cache_dir=None):
    global _torch_xla, _runtime, _xla_model, _device, _cache_dir

    requested_cache = _resolve_cache_dir(cache_dir)
    if _device is not None:
        if requested_cache != _cache_dir:
            raise RuntimeError(f"TPU is already initialized with compilation cache {_cache_dir}; cannot change it to {requested_cache}")
        return _device

    try:
        import torch_xla
        import torch_xla.runtime as xr
        import torch_xla.core.xla_model as xm
    except ImportError as e:
        raise RuntimeError("TPU support requires torch-xla. Install the PyTorch/XLA TPU wheel before using --tpu.") from e

    _cache_dir = requested_cache
    _cache_dir.mkdir(parents=True, exist_ok=True)

    # The cache must be configured before the first XLA device is created.
    xr.initialize_cache(str(_cache_dir), readonly=False)
    _torch_xla = torch_xla
    _runtime = xr
    _xla_model = xm
    _device = torch_xla.device()
    return _device


def _resolve_cache_dir(cache_dir):
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "tpu_xla"
    return Path(cache_dir).expanduser().resolve()


def device():
    if _device is None:
        raise RuntimeError("TPU runtime has not been initialized")
    return _device


def cache_dir():
    return _cache_dir


def device_type():
    return _runtime.device_type() if _runtime is not None else "TPU"


def memory_info():
    return parse_memory_info(_xla_model.get_memory_info(device()))


def parse_memory_info(info):
    if not isinstance(info, dict):
        raise RuntimeError(f"PyTorch/XLA returned unsupported memory information: {info!r}")
    if "bytes_limit" in info and "bytes_used" in info:
        total = info["bytes_limit"]
        used = info["bytes_used"]
        unit = 1
    elif "kb_total" in info and "kb_free" in info:
        total = info["kb_total"]
        used = info["kb_total"] - info["kb_free"]
        unit = 1024
    else:
        raise RuntimeError(f"PyTorch/XLA memory information has unsupported fields: {sorted(info)}")
    if not isinstance(total, (int, float)) or not isinstance(used, (int, float)):
        raise RuntimeError(f"PyTorch/XLA returned non-numeric memory information: {info!r}")
    if total < 0 or used < 0 or used > total:
        raise RuntimeError(f"PyTorch/XLA returned invalid memory information: {info!r}")
    total = int(total * unit)
    return total, total - int(used * unit)


def sync():
    _torch_xla.sync(wait=True)


def is_oom(error):
    message = str(error).lower()
    if any(pattern in message for pattern in ("hbm oom", "out of memory", "error allocating device buffer")):
        return True
    return "resource_exhausted" in message and any(pattern in message for pattern in ("allocat", "memory", "hbm"))


def environment():
    return {
        "PJRT_DEVICE": os.environ.get("PJRT_DEVICE", ""),
        "TPU_TYPE": os.environ.get("TPU_TYPE", ""),
        "TPU_ACCELERATOR_TYPE": os.environ.get("TPU_ACCELERATOR_TYPE", ""),
        "TPU_NAME": os.environ.get("TPU_NAME", ""),
    }
