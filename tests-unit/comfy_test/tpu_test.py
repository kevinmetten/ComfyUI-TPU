import sys
import types
import zipfile
import json
from pathlib import Path

import pytest
import torch

from comfy import tpu
from tools import tpu_cache
from tools import tpu_colab_setup
from tools import tpu_probe


@pytest.fixture(autouse=True)
def reset_tpu():
    tpu._torch_xla = None
    tpu._runtime = None
    tpu._xla_model = None
    tpu._device = None
    tpu._cache_dir = None


def test_initialize_configures_writable_cache_before_device(monkeypatch, tmp_path):
    calls = []
    runtime = types.ModuleType("torch_xla.runtime")
    runtime.initialize_cache = lambda path, readonly: calls.append(("cache", Path(path), readonly))
    runtime.device_type = lambda: "TPU"
    core = types.ModuleType("torch_xla.core")
    core.__path__ = []
    xla_model = types.ModuleType("torch_xla.core.xla_model")
    xla_model.get_memory_info = lambda device: {"bytes_limit": 1024 * 1024, "bytes_used": 768 * 1024, "peak_bytes_used": 800 * 1024}
    module = types.ModuleType("torch_xla")
    module.__path__ = []
    module.runtime = runtime
    module.device = lambda: calls.append(("device",)) or "xla:0"
    module.sync = lambda **kwargs: calls.append(("sync", kwargs))
    monkeypatch.setitem(sys.modules, "torch_xla", module)
    monkeypatch.setitem(sys.modules, "torch_xla.runtime", runtime)
    monkeypatch.setitem(sys.modules, "torch_xla.core", core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", xla_model)

    assert tpu.initialize(tmp_path / "cache") == "xla:0"
    assert calls[:2] == [("cache", (tmp_path / "cache").resolve(), False), ("device",)]
    assert tpu.memory_info() == (1024 * 1024, 256 * 1024)
    tpu.sync()
    assert calls[-1] == ("sync", {"wait": True})


def test_initialize_is_idempotent(monkeypatch, tmp_path):
    test_initialize_configures_writable_cache_before_device(monkeypatch, tmp_path)
    assert tpu.initialize(tmp_path / "cache") == "xla:0"
    with pytest.raises(RuntimeError, match="already initialized"):
        tpu.initialize(tmp_path / "other")


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ({"bytes_limit": 1000, "bytes_used": 250}, (1000, 750)),
        ({"bytes_limit": 1000, "bytes_used": 250, "peak_bytes_used": 900}, (1000, 750)),
        ({"kb_total": 1000, "kb_free": 750}, (1000 * 1024, 750 * 1024)),
    ],
)
def test_parse_memory_info(info, expected):
    assert tpu.parse_memory_info(info) == expected


@pytest.mark.parametrize("info", [None, {}, {"bytes_limit": 100}, {"bytes_limit": "100", "bytes_used": 1}, {"bytes_limit": 100, "bytes_used": 101}])
def test_parse_memory_info_rejects_malformed_data(info):
    with pytest.raises(RuntimeError, match="memory information"):
        tpu.parse_memory_info(info)


@pytest.mark.parametrize("message", ["RESOURCE_EXHAUSTED: allocation failed", "HBM OOM", "XLA out of memory", "Error allocating device buffer"])
def test_tpu_oom_messages(message):
    assert tpu.is_oom(RuntimeError(message))


def test_tpu_oom_does_not_match_unrelated_runtime_error():
    assert not tpu.is_oom(RuntimeError("XLA compilation failed"))
    assert not tpu.is_oom(RuntimeError("RESOURCE_EXHAUSTED: worker thread quota"))


def test_missing_torch_xla_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_xla", None)
    with pytest.raises(RuntimeError, match="requires torch-xla"):
        tpu.initialize()


def test_cache_import_rejects_parent_path(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../outside", "bad")
    with pytest.raises(ValueError, match="unsafe path"):
        tpu_cache.import_cache(tmp_path / "cache", archive)


def test_cache_import_warns_on_version_mismatch(monkeypatch, tmp_path):
    archive = tmp_path / "cache.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("manifest.json", '{"torch_version": "1.0", "tpu_backend_version": "1.0"}')
    monkeypatch.setattr(tpu_cache, "environment_info", lambda: {
        "python": "3.12", "torch": {"version": "2.0"}, "torch_xla": {"version": "2.0"},
        "libtpu": {"version": None}, "comfy_kitchen": {"version": None},
    })
    warnings = tpu_cache.import_cache(tmp_path / "cache", archive)
    assert len(warnings) == 2
    assert all("may miss and recompile" in warning for warning in warnings)


def test_probe_runs_first_and_second_execution(monkeypatch):
    calls = []
    monkeypatch.setattr(tpu_probe.tpu, "sync", lambda: calls.append("sync"))
    def operation():
        calls.append("operation")
        return torch.ones(2)
    result = tpu_probe.run_operation("test", operation)
    assert "first_execution_seconds" in result
    assert "second_execution_seconds" in result
    assert result["finite"] is True
    assert calls == ["sync", "operation", "sync", "sync", "operation", "sync"]


def test_comfy_kitchen_probe_uses_current_int8_api(monkeypatch):
    monkeypatch.setattr(tpu_probe.tpu, "sync", lambda: None)
    result = tpu_probe.kitchen_probes(torch.device("cpu"), False)
    names = {operation["name"] for operation in result["operations"]}
    assert names == {
        "comfy_kitchen.quantize_int8_rowwise",
        "comfy_kitchen.quantize_int8_tensorwise",
        "comfy_kitchen.quantize_and_rotate_rowwise",
        "comfy_kitchen.quantize_int8_convrot_weight",
        "comfy_kitchen.dequantize_int8_simple_dtype",
        "comfy_kitchen.dequantize_int8_convrot_weight_dtype",
        "comfy_kitchen.int8_linear",
        "comfy_kitchen.int8_linear_convrot",
    }
    assert all(operation["error"] is None for operation in result["operations"])
    assert all(operation["status"] == "passed" for operation in result["operations"])


def test_comfy_kitchen_probe_blocks_tensorwise_dependents(monkeypatch):
    import comfy_kitchen
    monkeypatch.setattr(tpu_probe.tpu, "sync", lambda: None)
    def fail_tensorwise(weight):
        raise RuntimeError("quantizer failed")
    monkeypatch.setattr(comfy_kitchen, "quantize_int8_tensorwise", fail_tensorwise)
    result = tpu_probe.kitchen_probes(torch.device("cpu"), False)
    operations = {operation["name"]: operation for operation in result["operations"]}
    assert operations["comfy_kitchen.quantize_int8_tensorwise"]["status"] == "failed"
    assert operations["comfy_kitchen.dequantize_int8_simple_dtype"]["status"] == "blocked"
    assert operations["comfy_kitchen.int8_linear"]["status"] == "blocked"
    assert operations["comfy_kitchen.int8_linear"]["blocked_by"] == "comfy_kitchen.quantize_int8_tensorwise"
    assert operations["comfy_kitchen.quantize_int8_convrot_weight"]["status"] == "passed"
    assert operations["comfy_kitchen.int8_linear_convrot"]["status"] == "passed"


def test_colab_notebook_targets_existing_pr_branch():
    notebook = json.loads(Path("colab/ComfyUI_TPU_v5e.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "https://github.com/kevinmetten/ComfyUI-TPU.git" in source
    assert "codex/build-comfyui-backend-for-google-colab-tpu" in source
    assert "YOUR_GITHUB_USER" not in source
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"ComfyUI_TPU_v5e.ipynb:cell-{index}", "exec")


@pytest.mark.parametrize(
    ("stack", "requirements", "compatible"),
    [
        ({"torch": "2.9.0", "torch-xla": "2.9.0", "torchaudio": "2.9.0", "torchvision": "0.24.0", "libtpu": "0.0.10"}, ["torch==2.9.0"], True),
        ({"torch": "2.9.0", "torch-xla": None, "torchaudio": "2.9.0", "torchvision": "0.24.0", "libtpu": "0.0.10"}, ["torch==2.9.0"], False),
        ({"torch": "2.9.0", "torch-xla": "2.8.0", "torchaudio": "2.9.0", "torchvision": "0.24.0", "libtpu": "0.0.10"}, ["torch==2.9.0"], False),
        ({"torch": "2.9.0", "torch-xla": "2.9.0", "torchaudio": "2.9.0", "torchvision": "0.23.0", "libtpu": "0.0.10"}, ["torch==2.8.0"], False),
    ],
)
def test_tpu_stack_compatibility(stack, requirements, compatible):
    assert tpu_colab_setup.stack_is_compatible(stack, requirements) is compatible


def test_tpu_stack_resolution_uses_latest_official_xla_release():
    def run(command, capture_output, text):
        if "index" in command:
            return types.SimpleNamespace(returncode=0, stdout="Available versions: 2.9.0, 2.8.1, 2.10.0.dev1\n", stderr="")
        report = Path(command[command.index("--report") + 1])
        report.write_text(json.dumps({"install": [
            {"metadata": {"name": "torch", "version": "2.9.0"}},
            {"metadata": {"name": "torchvision", "version": "0.24.0"}},
            {"metadata": {"name": "torchaudio", "version": "2.9.0"}},
            {"metadata": {"name": "torch_xla", "version": "2.9.0"}},
            {"metadata": {"name": "libtpu", "version": "0.0.10"}},
        ]}))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    assert tpu_colab_setup.resolve_tpu_stack(run) == {"torch": "2.9.0", "torchvision": "0.24.0", "torchaudio": "2.9.0", "torch-xla": "2.9.0", "libtpu": "0.0.10"}


def test_valid_tpu_stack_is_not_reinstalled(monkeypatch):
    stack = {"torch": "2.9.0", "torch-xla": "2.9.0", "torchaudio": "2.9.0", "torchvision": "0.24.0", "libtpu": "0.0.10"}
    monkeypatch.setattr(tpu_colab_setup, "installed_stack", lambda: stack)
    monkeypatch.setattr(tpu_colab_setup, "stack_is_compatible", lambda value: True)
    monkeypatch.setattr(tpu_colab_setup, "resolve_tpu_stack", lambda: pytest.fail("valid stack should not resolve or install"))
    assert tpu_colab_setup.ensure_tpu_stack() == (stack, False)


@pytest.mark.parametrize("current", [
    {"torch": "2.9.0", "torch-xla": None, "torchaudio": "2.9.0", "torchvision": "0.24.0", "libtpu": "0.0.10"},
    {"torch": "2.9.0", "torch-xla": "2.8.0", "torchaudio": "2.9.0", "torchvision": "0.24.0", "libtpu": "0.0.10"},
    {"torch": "2.9.0", "torch-xla": "2.9.0", "torchaudio": "2.9.0", "torchvision": "0.23.0", "libtpu": "0.0.10"},
])
def test_incomplete_or_mismatched_stack_is_replaced(monkeypatch, current):
    selected = {"torch": "2.9.0", "torch-xla": "2.9.0", "torchaudio": "2.9.0", "torchvision": "0.24.0", "libtpu": "0.0.10"}
    states = iter((current, selected))
    installed = []
    monkeypatch.setattr(tpu_colab_setup, "installed_stack", lambda: next(states))
    monkeypatch.setattr(tpu_colab_setup, "stack_is_compatible", lambda value: value == selected)
    monkeypatch.setattr(tpu_colab_setup, "resolve_tpu_stack", lambda: selected)
    monkeypatch.setattr(tpu_colab_setup, "install_stack", lambda value: installed.append(value))
    monkeypatch.setattr(tpu_colab_setup.subprocess, "check_call", lambda command: None)
    assert tpu_colab_setup.ensure_tpu_stack() == (selected, True)
    assert installed == [selected]


def test_unresolvable_tpu_stack_has_actionable_error():
    result = types.SimpleNamespace(returncode=1, stdout="", stderr="official index unavailable")
    with pytest.raises(RuntimeError, match="official PyTorch/XLA TPU package index"):
        tpu_colab_setup.latest_xla_version(lambda *args, **kwargs: result)
