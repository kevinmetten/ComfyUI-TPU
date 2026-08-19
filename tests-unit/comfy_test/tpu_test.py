import sys
import types
import zipfile
from pathlib import Path

import pytest

from comfy import tpu
from tools import tpu_cache


@pytest.fixture(autouse=True)
def reset_tpu():
    tpu._torch_xla = None
    tpu._runtime = None
    tpu._device = None
    tpu._cache_dir = None


def test_initialize_configures_writable_cache_before_device(monkeypatch, tmp_path):
    calls = []
    runtime = types.ModuleType("torch_xla.runtime")
    runtime.initialize_cache = lambda path, readonly: calls.append(("cache", Path(path), readonly))
    runtime.device_type = lambda: "TPU"
    module = types.ModuleType("torch_xla")
    module.__path__ = []
    module.runtime = runtime
    module.device = lambda: calls.append(("device",)) or "xla:0"
    module.sync = lambda: calls.append(("sync",))
    module._XLAC = types.SimpleNamespace(_xla_memory_info=lambda device: {"kb_total": 1024, "kb_free": 256})
    monkeypatch.setitem(sys.modules, "torch_xla", module)
    monkeypatch.setitem(sys.modules, "torch_xla.runtime", runtime)

    assert tpu.initialize(tmp_path / "cache") == "xla:0"
    assert calls[:2] == [("cache", (tmp_path / "cache").resolve(), False), ("device",)]
    assert tpu.memory_info() == (1024 * 1024, 256 * 1024)
    tpu.sync()
    assert calls[-1] == ("sync",)


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
