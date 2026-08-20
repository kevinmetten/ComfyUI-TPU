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


def operation_result(name, status="blocked", blocked_by=None):
    return {
        "name": name,
        "status": status,
        "first_execution_seconds": None,
        "second_execution_seconds": None,
        "dtype": None,
        "shape": None,
        "device": None,
        "finite": None,
        "max_abs_error": None,
        "mean_abs_error": None,
        "relative_l2_error": None,
        "error": None,
        "blocked_by": blocked_by,
    }


def execute_operation(name, fn, reference=None):
    result = operation_result(name)
    durations = []
    value = None
    try:
        for _ in range(2):
            tpu.sync()
            started = time.perf_counter()
            value = fn()
            tpu.sync()
            durations.append(time.perf_counter() - started)
        result["status"] = "passed"
        result["first_execution_seconds"] = durations[0]
        result["second_execution_seconds"] = durations[1]
        result.update(verify(value, reference() if callable(reference) else reference))
        described = describe(value)
        if described:
            result["dtype"] = described[0]["dtype"]
            result["shape"] = described[0]["shape"]
            result["device"] = described[0]["device"]
        result["completed_on_tpu"] = all(item.device.type == "xla" for item in tensors(value))
    except Exception as e:
        result["status"] = "failed"
        if durations:
            result["first_execution_seconds"] = durations[0]
        result["error"] = f"{type(e).__name__}: {e}"
    return result, value


def run_operation(name, fn, reference=None):
    return execute_operation(name, fn, reference)[0]


def kitchen_probes(device, large):
    import comfy_kitchen as ck
    from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE, _build_hadamard

    rows, features, outputs = (32, 256, 384) if not large else (256, 4096, 4096)
    group_size = 256
    x = torch.randn(rows, features, device=device, dtype=torch.bfloat16)
    weight = torch.randn(outputs, features, device=device, dtype=torch.bfloat16)
    bias = torch.randn(outputs, device=device, dtype=torch.bfloat16)
    tpu.sync()
    convrot_quantize = ck.registry.get_implementation("quantize_int8_convrot_weight", kwargs={"weight": weight, "group_size": group_size, "stochastic_rounding": 0})
    rotate_quantize = ck.registry.get_implementation("quantize_and_rotate_rowwise", kwargs={"x": x, "h": torch.empty(0, device=device), "group_size": group_size, "stochastic_rounding": 0})
    operations = []
    rowwise_result, _ = execute_operation("comfy_kitchen.quantize_int8_rowwise", lambda: ck.quantize_int8_rowwise(x))
    operations.append(rowwise_result)

    tensorwise_name = "comfy_kitchen.quantize_int8_tensorwise"
    tensorwise_result, tensorwise = execute_operation(tensorwise_name, lambda: ck.quantize_int8_tensorwise(weight))
    operations.append(tensorwise_result)
    if tensorwise_result["status"] == "passed":
        qweight, weight_scale = tensorwise
        operations.append(run_operation("comfy_kitchen.dequantize_int8_simple_dtype", lambda: torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(qweight, weight_scale, DTYPE_TO_CODE[torch.bfloat16])))
        operations.append(run_operation("comfy_kitchen.int8_linear", lambda: ck.int8_linear(x, qweight, weight_scale, bias=bias, out_dtype=torch.bfloat16), lambda: F.linear(x, qweight.float().mul(weight_scale).to(torch.bfloat16), bias)))
    else:
        operations.append(operation_result("comfy_kitchen.dequantize_int8_simple_dtype", blocked_by=tensorwise_name))
        operations.append(operation_result("comfy_kitchen.int8_linear", blocked_by=tensorwise_name))

    convrot_name = "comfy_kitchen.quantize_int8_convrot_weight"
    convrot_result, convrot = execute_operation(convrot_name, lambda: convrot_quantize(weight, group_size, stochastic_rounding=0))
    operations.append(convrot_result)
    try:
        h = _build_hadamard(group_size, device=device, dtype=torch.bfloat16)
        tpu.sync()
        operations.append(run_operation("comfy_kitchen.quantize_and_rotate_rowwise", lambda: rotate_quantize(x, h, group_size, stochastic_rounding=0)))
    except Exception as e:
        failed = operation_result("comfy_kitchen.quantize_and_rotate_rowwise", status="failed")
        failed["error"] = f"{type(e).__name__}: {e}"
        operations.append(failed)
    if convrot_result["status"] == "passed":
        qconvrot, convrot_scale = convrot
        operations.append(run_operation("comfy_kitchen.dequantize_int8_convrot_weight_dtype", lambda: torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(qconvrot, convrot_scale, group_size, DTYPE_TO_CODE[torch.bfloat16])))
        operations.append(run_operation("comfy_kitchen.int8_linear_convrot", lambda: ck.int8_linear(x, qconvrot, convrot_scale, bias=bias, out_dtype=torch.bfloat16, convrot=True, convrot_groupsize=group_size), lambda: F.linear(x, weight, bias)))
    else:
        operations.append(operation_result("comfy_kitchen.dequantize_int8_convrot_weight_dtype", blocked_by=convrot_name))
        operations.append(operation_result("comfy_kitchen.int8_linear_convrot", blocked_by=convrot_name))
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
    matmul_inputs = {dtype: (torch.randn(256, 256, device=device, dtype=dtype), torch.randn(256, 256, device=device, dtype=dtype)) for dtype in (torch.float32, torch.bfloat16, torch.float16)}
    linear_weight = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
    conv_input = torch.randn(1, 8, 16, 16, device=device, dtype=torch.bfloat16)
    conv_weight = torch.randn(16, 8, 3, 3, device=device, dtype=torch.bfloat16)
    sdpa_inputs = tuple(torch.randn(1, 4, 128, 64, device=device, dtype=torch.bfloat16) for _ in range(3))
    int8_inputs = (torch.randint(-127, 128, (256, 256), device=device, dtype=torch.int8), torch.randint(-127, 128, (256, 256), device=device, dtype=torch.int8))
    rope_angles = torch.arange(32, device=device).mul(math.pi / 32)
    host_input = torch.randn(64, 64, dtype=torch.bfloat16)
    tpu.sync()
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        a, b = matmul_inputs[dtype]
        report["operations"].append(run_operation(f"torch.matmul.{dtype}", lambda a=a, b=b: a @ b))
    report["operations"].extend([
        run_operation("torch.linear.bfloat16", lambda: F.linear(x, linear_weight)),
        run_operation("torch.conv2d.bfloat16", lambda: F.conv2d(conv_input, conv_weight, padding=1)),
        run_operation("torch.layer_norm", lambda: F.layer_norm(x, (64,))),
        run_operation("torch.sdpa", lambda: F.scaled_dot_product_attention(*sdpa_inputs)),
        run_operation("torch.rope_layout", lambda: torch.stack((x[..., :32].float() * rope_angles.cos(), x[..., 32:].float() * rope_angles.sin()), -1).flatten(-2)),
        run_operation("torch.int8_matmul_generic", lambda: int8_inputs[0] @ int8_inputs[1]),
        run_operation("torch.non_contiguous", lambda: x.transpose(1, 2).contiguous().reshape(2, 64, 32)),
        run_operation("transfer.host_to_tpu", lambda: host_input.to(device)),
        run_operation("transfer.tpu_to_host", lambda: x.cpu()),
    ])
    report["compilation"]["after_core_operations"] = metrics_report()
    try:
        report["compilation"]["before_comfy_kitchen"] = metrics_report()
        report["comfy_kitchen"] = kitchen_probes(device, args.large)
        report["compilation"]["after_comfy_kitchen"] = metrics_report()
    except Exception as e:
        report["comfy_kitchen"] = {"error": f"{type(e).__name__}: {e}"}
    report["compilation"]["final"] = metrics_report()
    report["fallbacks"] = [line.strip() for phase in report["compilation"].values() for line in str(phase).splitlines() if "fallback" in line.lower() or "aten::" in line]
    failures = [item for item in report["operations"] + report.get("comfy_kitchen", {}).get("operations", []) if item.get("status") in ("failed", "blocked")]
    report["warnings"].extend(f"{item['name']}: {item.get('error') or 'blocked by ' + item['blocked_by']}" for item in failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    if args.diagnostics:
        for phase, metrics in report["compilation"].items():
            (args.diagnostics / f"xla_metrics_{phase}.txt").write_text(str(metrics), encoding="utf-8")
    print(f"TPU probe completed: {len(report['operations']) + len(report.get('comfy_kitchen', {}).get('operations', []))} operations, {len(failures)} failures")
    print("TPU validation artifacts:")
    print(f"- {args.output.resolve()}")
    if args.diagnostics:
        print(f"- {args.diagnostics.resolve()}")
    print("- ComfyUI startup log from: python main.py --tpu --listen 0.0.0.0")


if __name__ == "__main__":
    main()
