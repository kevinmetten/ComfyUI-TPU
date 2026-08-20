#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from tools.tpu_metadata import environment_info


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / ".cache" / "tpu_xla"


def git_commit(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def export_cache(cache, archive):
    cache.mkdir(parents=True, exist_ok=True)
    environment = environment_info()
    existing_manifest = cache / "manifest.json"
    created_at = None
    if existing_manifest.exists():
        try:
            created_at = json.loads(existing_manifest.read_text(encoding="utf-8")).get("created_at")
        except (json.JSONDecodeError, OSError):
            pass
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "tpu_generation": environment.get("TPU_TYPE") or environment.get("TPU_ACCELERATOR_TYPE"),
        "torch_version": environment["torch"]["version"],
        "tpu_backend": "PyTorch/XLA",
        "tpu_backend_version": environment["torch_xla"]["version"],
        "libtpu_version": environment["libtpu"]["version"],
        "python_version": environment["python"],
        "comfyui_commit": git_commit(ROOT),
        "comfy_kitchen_version": environment["comfy_kitchen"]["version"],
        "comfy_kitchen_commit": environment["comfy_kitchen"]["commit"],
        "comfy_kitchen_source": environment["comfy_kitchen"]["source"],
        "created_at": created_at or now,
        "updated_at": now,
    }
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
    if not manifest.exists():
        print("Cache imported without a manifest")
        return []
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        warning = f"Cache imported, but its manifest could not be read: {e}. Existing entries may miss and recompile."
        print(f"WARNING: {warning}", file=sys.stderr)
        return [warning]
    print(json.dumps(data, indent=2))
    current = environment_info()
    comparisons = {
        "torch_version": current["torch"]["version"],
        "tpu_backend_version": current["torch_xla"]["version"],
        "libtpu_version": current["libtpu"]["version"],
        "python_version": current["python"],
        "comfy_kitchen_version": current["comfy_kitchen"]["version"],
    }
    warnings = []
    for field, current_value in comparisons.items():
        imported_value = data.get(field)
        if imported_value and current_value and imported_value != current_value:
            warnings.append(f"Imported cache uses {field} {imported_value}, current environment uses {current_value}. Existing entries may miss and recompile.")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return warnings


def main():
    parser = argparse.ArgumentParser(description="Import or export the portable TPU compilation cache")
    parser.add_argument("action", choices=("import", "export"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    (import_cache if args.action == "import" else export_cache)(args.cache_dir.resolve(), args.archive.resolve())


if __name__ == "__main__":
    main()
