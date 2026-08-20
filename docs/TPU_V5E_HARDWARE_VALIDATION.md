# First v5e-1 hardware validation

## Environment

- Google Colab v5e-1, one XLA device (`xla:0`)
- Python 3.12.13
- Torch 2.9.0+cpu, PyTorch/XLA 2.9.0, libtpu 0.0.21
- Comfy-Kitchen 0.2.31
- XLA memory limit: 16,909,336,576 bytes (16,126 MiB reported by ComfyUI)

## Validated behavior

ComfyUI selected the TPU, reported HBM, chose BF16 and PyTorch attention, initialized its persistent cache, started its server, and was reachable through a Cloudflare tunnel. FP32/BF16/FP16 matmul, BF16 linear and convolution, layer norm, SDPA, RoPE-style operations, generic INT8 matmul, non-contiguous operations, and host/TPU transfers completed on v5e-1. Repeated executions were materially faster than first compile/execution; for example BF16 matmul was approximately 28.7 ms then 0.241 ms, and SDPA was approximately 47.0 ms then 0.524 ms.

All eight eager Comfy-Kitchen capability probes completed functionally. Normal INT8 linear had approximately 0.00746 relative L2 error and ConvRot INT8 linear approximately 0.01058.

## INT8 fallback finding

XLA metrics gained four `aten::_int_mm` fallback calls, matching the two executions each of normal and ConvRot `int8_linear`. Generic `int8 @ int8` lowered to an XLA integer dot but returned INT8, while `torch._int_mm` returns INT32 accumulators. Substituting generic matmul would overflow and is not correct.

PyTorch/XLA 2.9 therefore has no demonstrated normal PyTorch primitive for an s8-by-s8 dot with required s32 accumulation. The correctness fallback remains in place. The probe now reports this honestly and includes candidates for `_int_mm`, generic INT8 matmul, and INT32 matmul against a CPU INT32 reference. A production TPU-native solution requires a Comfy-Kitchen TPU backend using a supported s8×s8→s32 XLA/Pallas/custom lowering, followed by v5e correctness and performance validation.

## Lifecycle finding

The original notebook cell ended after printing the tunnel URL and did not monitor its children. A persistent heartbeat loop was proven manually. The notebook now polls ComfyUI before starting cloudflared, keeps both processes alive, reports health every 60 seconds, and prints logs on failure.

## Remaining work

- Implement and validate a fully TPU-native wide-accumulation Comfy-Kitchen INT8 linear.
- Run a real checkpoint workflow, then an INT8/ConvRot checkpoint and MiniMax H3.
- Validate large-model staging/offload, video inference, real-model HBM peaks, cache reuse in a fresh runtime, and performance against T4.
