#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / ".cache" / "tpu_xla"


def git_commit(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def export_cache(cache, archive):
    cache.mkdir(parents=True, exist_ok=True)
    import torch
    manifest = {
        "tpu_generation": "v5e",
        "torch_version": torch.__version__,
        "tpu_backend": "PyTorch/XLA",
        "tpu_backend_version": None,
        "libtpu_version": None,
        "comfyui_commit": git_commit(ROOT),
        "comfy_kitchen_version": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch_xla
        manifest["tpu_backend_version"] = torch_xla.__version__
    except ImportError:
        pass
    try:
        import comfy_kitchen
        manifest["comfy_kitchen_version"] = getattr(comfy_kitchen, "__version__", None)
    except ImportError:
        pass
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output = shutil.make_archive(str(archive.with_suffix("")), "zip", cache)
    print(output)


def import_cache(cache, archive):
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        root = cache.resolve()
        for member in source.infolist():
            destination = (root / member.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"Cache archive contains an unsafe path: {member.filename}")
        source.extractall(root)
    manifest = cache / "manifest.json"
    print(manifest.read_text(encoding="utf-8") if manifest.exists() else "Cache imported without a manifest")


def main():
    parser = argparse.ArgumentParser(description="Import or export the portable TPU compilation cache")
    parser.add_argument("action", choices=("import", "export"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    (import_cache if args.action == "import" else export_cache)(args.cache_dir.resolve(), args.archive.resolve())


if __name__ == "__main__":
    main()
