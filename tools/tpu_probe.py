#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from comfy import tpu
from tools.tpu_metadata import environment_info


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    return []


def describe(value):
    return [{"shape": list(item.shape), "dtype": str(item.dtype), "device": str(item.device)} for item in tensors(value)]


def verify(value, reference=None):
    values = tensors(value)
    result = {"finite": all(bool(torch.isfinite(item).all().cpu()) for item in values), "outputs": describe(value)}
    if reference is not None and values:
        actual = values[0].float()
        expected = reference.float()
        error = (actual - expected).abs()
        result.update({
            "max_abs_error": float(error.max().cpu()),
            "mean_abs_error": float(error.mean().cpu()),
            "relative_l2_error": float((torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected).clamp(min=1e-12)).cpu()),
        })
    return result


def run_operation(name, fn, reference=None):
    result = {"name": name}
    outputs = []
    durations = []
    try:
        for _ in range(2):
            started = time.perf_counter()
            value = fn()
            tpu.sync()
            durations.append(time.perf_counter() - started)
            outputs.append(value)
        result["first_execution_seconds"] = durations[0]
        result["second_execution_seconds"] = durations[1]
        result.update(verify(outputs[-1], reference() if callable(reference) else reference))
        result["completed_on_tpu"] = all(item.device.type == "xla" for item in tensors(outputs[-1]))
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def kitchen_probes(device, large):
    import comfy_kitchen as ck
    from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE, _build_hadamard

    rows, features, outputs = (32, 256, 384) if not large else (256, 4096, 4096)
    group_size = 256
    x = torch.randn(rows, features, device=device, dtype=torch.bfloat16)
    weight = torch.randn(outputs, features, device=device, dtype=torch.bfloat16)
    bias = torch.randn(outputs, device=device, dtype=torch.bfloat16)
    qweight, weight_scale = ck.quantize_int8_tensorwise(weight)
    convrot_quantize = ck.registry.get_implementation("quantize_int8_convrot_weight", kwargs={"weight": weight, "group_size": group_size, "stochastic_rounding": 0})
    qconvrot, convrot_scale = convrot_quantize(weight, group_size, stochastic_rounding=0)
    rotate_quantize = ck.registry.get_implementation("quantize_and_rotate_rowwise", kwargs={"x": x, "h": torch.empty(0, device=device), "group_size": group_size, "stochastic_rounding": 0})

    h = _build_hadamard(group_size, device=device, dtype=torch.bfloat16)
    operations = [
        run_operation("comfy_kitchen.quantize_int8_rowwise", lambda: ck.quantize_int8_rowwise(x)),
        run_operation("comfy_kitchen.quantize_int8_tensorwise", lambda: ck.quantize_int8_tensorwise(weight)),
        run_operation("comfy_kitchen.quantize_and_rotate_rowwise", lambda: rotate_quantize(x, h, group_size, stochastic_rounding=0)),
        run_operation("comfy_kitchen.quantize_int8_convrot_weight", lambda: convrot_quantize(weight, group_size, stochastic_rounding=0)),
        run_operation("comfy_kitchen.dequantize_int8_simple_dtype", lambda: torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(qweight, weight_scale, DTYPE_TO_CODE[torch.bfloat16])),
        run_operation("comfy_kitchen.dequantize_int8_convrot_weight_dtype", lambda: torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(qconvrot, convrot_scale, group_size, DTYPE_TO_CODE[torch.bfloat16])),
        run_operation("comfy_kitchen.int8_linear", lambda: ck.int8_linear(x, qweight, weight_scale, bias=bias, out_dtype=torch.bfloat16), lambda: F.linear(x, qweight.float().mul(weight_scale).to(torch.bfloat16), bias)),
        run_operation("comfy_kitchen.int8_linear_convrot", lambda: ck.int8_linear(x, qconvrot, convrot_scale, bias=bias, out_dtype=torch.bfloat16, convrot=True, convrot_groupsize=group_size), lambda: F.linear(x, weight, bias)),
    ]
    return {"backends": ck.list_backends(), "shape": {"rows": rows, "in_features": features, "out_features": outputs, "convrot_group_size": group_size}, "operations": operations}


def metrics_report():
    try:
        import torch_xla.debug.metrics as metrics
        return metrics.metrics_report()
    except (ImportError, RuntimeError) as e:
        return f"Unavailable: {e}"


def main():
    parser = argparse.ArgumentParser(description="Execute representative ComfyUI inference operations on a TPU")
    parser.add_argument("--cache-dir")
    parser.add_argument("--large", action="store_true", help="Use transformer-sized Comfy Kitchen linear operations")
    parser.add_argument("--output", type=Path, default=Path("tpu_probe.json"))
    parser.add_argument("--diagnostics", type=Path, help="Write XLA compiler and metrics diagnostics into this directory")
    args = parser.parse_args()
    if args.diagnostics:
        args.diagnostics.mkdir(parents=True, exist_ok=True)
        os.environ["PT_XLA_DEBUG_LEVEL"] = "2"
        os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_dump_to={args.diagnostics.resolve()}".strip()

    device = tpu.initialize(args.cache_dir)
    total, free = tpu.memory_info()
    report = {
        "runtime": {"backend": "PyTorch/XLA", "device_type": tpu.device_type(), "cache_dir": str(tpu.cache_dir())},
        "device": {"name": str(device), "type": device.type},
        "memory": {"total_bytes": total, "free_bytes": free, "used_bytes": total - free},
        "versions": environment_info(),
        "operations": [],
        "comfy_kitchen": {},
        "compilation": {},
        "fallbacks": [],
        "warnings": [],
    }
    x = torch.randn(2, 32, 64, device=device, dtype=torch.bfloat16)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        report["operations"].append(run_operation(f"torch.matmul.{dtype}", lambda dtype=dtype: torch.randn(256, 256, device=device, dtype=dtype) @ torch.randn(256, 256, device=device, dtype=dtype)))
    report["operations"].extend([
        run_operation("torch.linear.bfloat16", lambda: F.linear(x, torch.randn(128, 64, device=device, dtype=torch.bfloat16))),
        run_operation("torch.conv2d.bfloat16", lambda: F.conv2d(torch.randn(1, 8, 16, 16, device=device, dtype=torch.bfloat16), torch.randn(16, 8, 3, 3, device=device, dtype=torch.bfloat16), padding=1)),
        run_operation("torch.layer_norm", lambda: F.layer_norm(x, (64,))),
        run_operation("torch.sdpa", lambda: F.scaled_dot_product_attention(*(torch.randn(1, 4, 128, 64, device=device, dtype=torch.bfloat16) for _ in range(3)))),
        run_operation("torch.rope_layout", lambda: torch.stack((x[..., :32].float() * torch.arange(32, device=device).mul(math.pi / 32).cos(), x[..., 32:].float() * torch.arange(32, device=device).mul(math.pi / 32).sin()), -1).flatten(-2)),
        run_operation("torch.int8_matmul_generic", lambda: torch.randint(-127, 128, (256, 256), device=device, dtype=torch.int8) @ torch.randint(-127, 128, (256, 256), device=device, dtype=torch.int8)),
        run_operation("torch.non_contiguous", lambda: x.transpose(1, 2).contiguous().reshape(2, 64, 32)),
        run_operation("transfer.host_to_tpu", lambda: torch.randn(64, 64, dtype=torch.bfloat16).to(device)),
        run_operation("transfer.tpu_to_host", lambda: x.cpu()),
    ])
    try:
        report["comfy_kitchen"] = kitchen_probes(device, args.large)
    except Exception as e:
        report["comfy_kitchen"] = {"error": f"{type(e).__name__}: {e}"}
    report["compilation"]["xla_metrics"] = metrics_report()
    report["fallbacks"] = [line.strip() for line in str(report["compilation"]["xla_metrics"]).splitlines() if "fallback" in line.lower() or "aten::" in line]
    failures = [item for item in report["operations"] + report.get("comfy_kitchen", {}).get("operations", []) if "error" in item]
    report["warnings"].extend(f"{item['name']}: {item['error']}" for item in failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    if args.diagnostics:
        (args.diagnostics / "xla_metrics.txt").write_text(str(report["compilation"]["xla_metrics"]), encoding="utf-8")
    print(f"TPU probe completed: {len(report['operations']) + len(report.get('comfy_kitchen', {}).get('operations', []))} operations, {len(failures)} failures")
    print("TPU validation artifacts:")
    print(f"- {args.output.resolve()}")
    if args.diagnostics:
        print(f"- {args.diagnostics.resolve()}")
    print("- ComfyUI startup log from: python main.py --tpu --listen 0.0.0.0")


if __name__ == "__main__":
    main()
