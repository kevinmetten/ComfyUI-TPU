from __future__ import annotations

import os
from pathlib import Path


_torch_xla = None
_runtime = None
_device = None
_cache_dir = None


def initialize(cache_dir=None):
    global _torch_xla, _runtime, _device, _cache_dir

    try:
        import torch_xla
        import torch_xla.runtime as xr
    except ImportError as e:
        raise RuntimeError("TPU support requires torch-xla. Install the PyTorch/XLA TPU wheel before using --tpu.") from e

    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "tpu_xla"
    _cache_dir = Path(cache_dir).expanduser().resolve()
    _cache_dir.mkdir(parents=True, exist_ok=True)

    # The cache must be configured before the first XLA device is created.
    xr.initialize_cache(str(_cache_dir), readonly=False)
    _torch_xla = torch_xla
    _runtime = xr
    _device = torch_xla.device()
    return _device


def device():
    if _device is None:
        raise RuntimeError("TPU runtime has not been initialized")
    return _device


def cache_dir():
    return _cache_dir


def device_type():
    return _runtime.device_type() if _runtime is not None else "TPU"


def memory_info():
    info = _torch_xla._XLAC._xla_memory_info(str(device()))
    total = int(info["kb_total"]) * 1024
    free = int(info["kb_free"]) * 1024
    return total, free


def sync():
    _torch_xla.sync()


def environment():
    return {
        "PJRT_DEVICE": os.environ.get("PJRT_DEVICE", ""),
        "TPU_TYPE": os.environ.get("TPU_TYPE", ""),
    }
