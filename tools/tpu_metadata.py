import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def package_info(distribution, module_name=None):
    info = {"version": None, "source": None, "commit": None}
    try:
        dist = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return info
    info["version"] = dist.version
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        data = json.loads(direct_url)
        info["source"] = data.get("url")
        info["commit"] = data.get("vcs_info", {}).get("commit_id")
    else:
        info["source"] = "PyPI"
    if module_name and info["source"] != "PyPI":
        try:
            module = __import__(module_name)
            module_path = Path(module.__file__).resolve()
            result = subprocess.run(["git", "-C", str(module_path.parent), "rev-parse", "HEAD"], capture_output=True, text=True)
            if result.returncode == 0:
                info["commit"] = result.stdout.strip()
                remote = subprocess.run(["git", "-C", str(module_path.parent), "remote", "get-url", "origin"], capture_output=True, text=True)
                if remote.returncode == 0:
                    info["source"] = remote.stdout.strip()
        except (ImportError, OSError):
            pass
    return info


def environment_info():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": package_info("torch"),
        "torch_xla": package_info("torch-xla", "torch_xla"),
        "libtpu": package_info("libtpu"),
        "comfy_kitchen": package_info("comfy-kitchen", "comfy_kitchen"),
        "PJRT_DEVICE": os.environ.get("PJRT_DEVICE"),
        "TPU_TYPE": os.environ.get("TPU_TYPE"),
        "TPU_ACCELERATOR_TYPE": os.environ.get("TPU_ACCELERATOR_TYPE"),
        "TPU_NAME": os.environ.get("TPU_NAME"),
        "python_executable": sys.executable,
    }
