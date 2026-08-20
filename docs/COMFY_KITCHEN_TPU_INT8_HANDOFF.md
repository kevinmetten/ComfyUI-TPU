# Comfy-Kitchen TPU INT8 handoff

## Validated problem

Comfy-Kitchen 0.2.31 eager `_int8_matmul_accumulate` selects `torch._int_mm`, whose contract is INT8 inputs and INT32 accumulation. PyTorch/XLA 2.9 does not lower it: the v5e metrics delta was `aten::_int_mm = 4` for two normal and two ConvRot linear executions.

Plain `a @ b` is not a replacement. For `[[127, 127]] @ [[127], [127]]`, generic INT8 matmul returns INT8 value `2`, while `_int_mm` returns INT32 value `32258`.

## Required Kitchen implementation

Add a TPU backend through the existing backend registry rather than changing eager behavior. It should advertise `int8_linear` only for XLA inputs after the following are demonstrated on the stable 2.9 Colab stack:

1. s8 activation by s8 weight dot with s32 accumulation;
2. scalar or per-output-channel weight scaling and per-row activation scaling;
3. BF16 output and optional bias;
4. ConvRot's existing activation rotation before the same GEMM;
5. no `aten::` metrics delta, output on `xla:0`, and numerical agreement with CPU `_int_mm`;
6. repeat execution faster than the eager CPU-fallback route at 32×4096×4096 and 256×4096×4096.

If no public PyTorch/XLA operation exposes the accumulator type, the next implementation candidate is a Pallas TPU or registered XLA lowering for only the s8×s8→s32 dot. Quantization, rotation, scaling, bias, tensor layouts, and non-TPU backends should remain unchanged.

## Validation harness

Run:

```bash
python -m tools.tpu_probe --diagnostics diagnostics --output tpu_probe.json --int8-experiments --large
```

`diagnostics/int8_experiments.json` compares `_int_mm`, generic INT8 matmul, and INT32 matmul candidates to a CPU `_int_mm` reference and records numerical error, output placement, per-operation fallback deltas, and first/repeated timing. Enable TPU Kitchen dispatch only after a candidate satisfies all correctness and performance checks.
