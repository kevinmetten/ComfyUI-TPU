#!/usr/bin/env python3
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from comfy import tpu


def timed(name, fn):
    started = time.perf_counter()
    try:
        value = fn()
        tpu.sync()
        return {"name": name, "seconds": time.perf_counter() - started, "finite": bool(torch.isfinite(value).all().cpu())}
    except Exception as e:
        return {"name": name, "seconds": time.perf_counter() - started, "error": f"{type(e).__name__}: {e}"}


def main():
    parser = argparse.ArgumentParser(description="Execute representative ComfyUI inference operations on a TPU")
    parser.add_argument("--cache-dir")
    parser.add_argument("--large", action="store_true", help="Also run video-transformer-sized matrix operations")
    args = parser.parse_args()
    device = tpu.initialize(args.cache_dir)
    total, free = tpu.memory_info()
    report = {"device": str(device), "device_type": tpu.device_type(), "memory_total": total, "memory_free": free, "environment": tpu.environment(), "operations": []}

    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        report["operations"].append(timed(f"matmul_{dtype}", lambda dtype=dtype: torch.randn(256, 256, device=device, dtype=dtype) @ torch.randn(256, 256, device=device, dtype=dtype)))

    x = torch.randn(2, 32, 64, device=device, dtype=torch.bfloat16)
    weight = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
    report["operations"].append(timed("linear_bf16", lambda: F.linear(x, weight)))
    image = torch.randn(1, 8, 16, 16, device=device, dtype=torch.bfloat16)
    kernel = torch.randn(16, 8, 3, 3, device=device, dtype=torch.bfloat16)
    report["operations"].append(timed("conv2d_bf16", lambda: F.conv2d(image, kernel, padding=1)))
    report["operations"].append(timed("layer_norm", lambda: F.layer_norm(x, (64,))))
    q = torch.randn(1, 4, 128, 64, device=device, dtype=torch.bfloat16)
    report["operations"].append(timed("sdpa", lambda: F.scaled_dot_product_attention(q, q, q)))
    angles = torch.arange(32, device=device, dtype=torch.float32) * (math.pi / 32)
    report["operations"].append(timed("rope_layout", lambda: torch.stack((x[..., :32].float() * angles.cos(), x[..., 32:].float() * angles.sin()), -1).flatten(-2)))
    qi = torch.randint(-127, 128, (256, 256), device=device, dtype=torch.int8)
    report["operations"].append(timed("int8_matmul", lambda: torch.matmul(qi, qi)))
    report["operations"].append(timed("non_contiguous", lambda: x.transpose(1, 2).contiguous().reshape(2, 64, 32)))
    host = torch.randn(64, 64, dtype=torch.bfloat16)
    report["operations"].append(timed("host_to_tpu", lambda: host.to(device)))
    report["operations"].append(timed("tpu_to_host", lambda: x.cpu()))
    if args.large:
        report["operations"].append(timed("video_transformer_matmul", lambda: torch.randn(2048, 4096, device=device, dtype=torch.bfloat16) @ torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)))

    try:
        import comfy_kitchen as ck
        qx, scale = ck.quantize_int8_rowwise(x.reshape(-1, 64))
        report["operations"].append(timed("comfy_kitchen_quantize_int8_rowwise", lambda: qx.float() * scale))
        report["comfy_kitchen_backends"] = ck.list_backends()
    except (ImportError, RuntimeError, NotImplementedError) as e:
        report["comfy_kitchen_error"] = str(e)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
