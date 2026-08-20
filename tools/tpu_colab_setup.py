#!/usr/bin/env python3
import argparse
import importlib.metadata
import json
import subprocess
import sys
import tempfile

from packaging.requirements import Requirement
from packaging.version import Version


TPU_WHEEL_INDEX = "https://storage.googleapis.com/libtpu-releases/index.html"
PACKAGES = ("torch", "torchvision", "torchaudio", "torch-xla", "libtpu")


def installed_stack(version=importlib.metadata.version):
    result = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _base_version(value):
    return Version(value).base_version if value else None


def stack_is_compatible(stack, torchvision_requires=None):
    if not all(stack.get(package) for package in PACKAGES):
        return False
    torch_version = _base_version(stack["torch"])
    if torch_version != _base_version(stack["torch-xla"]) or torch_version != _base_version(stack["torchaudio"]):
        return False
    if torchvision_requires is None:
        torchvision_requires = importlib.metadata.requires("torchvision") or []
    torch_requirements = [Requirement(item) for item in torchvision_requires if Requirement(item).name == "torch"]
    return bool(torch_requirements) and all(Version(torch_version) in requirement.specifier for requirement in torch_requirements)


def latest_xla_version(run=subprocess.run):
    command = [sys.executable, "-m", "pip", "index", "versions", "torch-xla", "--find-links", TPU_WHEEL_INDEX]
    result = run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Could not query the official PyTorch/XLA TPU package index:\n{result.stderr.strip()}")
    marker = "Available versions:"
    line = next((line for line in result.stdout.splitlines() if marker in line), None)
    if line is None:
        raise RuntimeError("The official PyTorch/XLA TPU package index returned no version list")
    versions = [Version(item.strip()) for item in line.split(marker, 1)[1].split(",")]
    stable = [version for version in versions if not version.is_prerelease and not version.is_devrelease]
    if not stable:
        raise RuntimeError("The official PyTorch/XLA TPU package index returned no stable release")
    return str(max(stable))


def resolve_tpu_stack(run=subprocess.run):
    xla_version = latest_xla_version(run)
    requested = [f"torch=={xla_version}", f"torchaudio=={xla_version}", f"torch_xla[tpu]=={xla_version}", "torchvision"]
    with tempfile.NamedTemporaryFile(suffix=".json") as report:
        command = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--report", report.name, "--find-links", TPU_WHEEL_INDEX, *requested]
        result = run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"No compatible current TPU PyTorch stack was resolved:\n{result.stderr.strip()}")
        data = json.load(report)
    resolved = {item["metadata"]["name"].lower().replace("_", "-"): item["metadata"]["version"] for item in data["install"]}
    stack = {package: resolved.get(package) for package in PACKAGES}
    if not stack_is_compatible(stack, [f"torch=={stack['torch']}"]):
        raise RuntimeError(f"The official package resolver returned an incompatible TPU stack: {stack}")
    return stack


def install_stack(stack, check_call=subprocess.check_call):
    specs = [
        f"torch=={stack['torch']}",
        f"torchvision=={stack['torchvision']}",
        f"torchaudio=={stack['torchaudio']}",
        f"torch_xla[tpu]=={stack['torch-xla']}",
        f"libtpu=={stack['libtpu']}",
    ]
    check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--find-links", TPU_WHEEL_INDEX, *specs])


def ensure_tpu_stack():
    current = installed_stack()
    if stack_is_compatible(current):
        return current, False
    selected = resolve_tpu_stack()
    install_stack(selected)
    installed = installed_stack()
    if installed != selected or not stack_is_compatible(installed):
        raise RuntimeError(f"TPU stack installation did not produce the resolved versions. Expected {selected}, found {installed}")
    subprocess.check_call([sys.executable, "-c", "import torch, torchvision, torchaudio, torch_xla"])
    return installed, True


def main():
    parser = argparse.ArgumentParser(description="Resolve and install a compatible current PyTorch/XLA TPU stack")
    parser.add_argument("--ensure", action="store_true", help="Install a compatible stack when the current environment is incomplete or incompatible")
    args = parser.parse_args()
    stack, installed = ensure_tpu_stack() if args.ensure else (installed_stack(), False)
    print(json.dumps({"stack": stack, "installed": installed, "index": TPU_WHEEL_INDEX}, indent=2))


if __name__ == "__main__":
    main()
