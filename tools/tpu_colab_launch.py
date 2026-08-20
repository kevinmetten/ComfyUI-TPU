#!/usr/bin/env python3
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


LOCAL_URL = "http://127.0.0.1:8188"


def http_status(url=LOCAL_URL, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status
    except (urllib.error.URLError, TimeoutError):
        return None


def port_open(host="127.0.0.1", port=8188, timeout=1):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def log_tail(path, lines=40):
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def wait_for_server(server, log_path, timeout=120, poll_interval=2, status=http_status, sleep=time.sleep):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"ComfyUI exited with code {server.returncode} during startup:\n{log_tail(log_path)}")
        code = status()
        if code == 200:
            return code
        sleep(poll_interval)
    raise RuntimeError(f"ComfyUI did not return HTTP 200 within {timeout} seconds:\n{log_tail(log_path)}")


def wait_for_tunnel_url(tunnel, timeout=60, monotonic=time.monotonic):
    deadline = monotonic() + timeout
    output = []
    while monotonic() < deadline:
        if tunnel.poll() is not None:
            raise RuntimeError(f"cloudflared exited with code {tunnel.returncode}:\n{''.join(output)}")
        line = tunnel.stdout.readline()
        if not line:
            continue
        output.append(line)
        match = re.search(r"https://[^ ]+\.trycloudflare\.com", line)
        if match:
            return match.group(0)
    raise RuntimeError(f"cloudflared did not provide a tunnel URL:\n{''.join(output)}")


def heartbeat_state(server, tunnel, status=http_status):
    if server.poll() is not None:
        return False, f"ComfyUI exited with code {server.returncode}"
    if tunnel.poll() is not None:
        return False, f"cloudflared exited with code {tunnel.returncode}"
    code = status()
    if code != 200:
        return False, f"ComfyUI HTTP health check returned {code}"
    return True, "alive"


def run_colab_server(root, cloudflared, heartbeat_seconds=60):
    root = Path(root)
    log_path = root / "comfyui-tpu-startup.log"
    if port_open():
        status = http_status()
        raise RuntimeError(f"Port 8188 is already occupied (HTTP status {status}). Stop or reuse that server instead of starting a duplicate.")
    log_file = log_path.open("w", encoding="utf-8")
    server = subprocess.Popen([sys.executable, "main.py", "--tpu", "--listen", "0.0.0.0"], cwd=root, stdout=log_file, stderr=subprocess.STDOUT)
    tunnel = None
    try:
        wait_for_server(server, log_path)
        tunnel = subprocess.Popen([str(cloudflared), "tunnel", "--url", LOCAL_URL], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        url = wait_for_tunnel_url(tunnel)
        print("=" * 40)
        print("COMFYUI TPU IS LIVE")
        print(f"URL: {url}")
        print("=" * 40)
        while True:
            healthy, message = heartbeat_state(server, tunnel)
            print(f"[{time.strftime('%H:%M:%S')}]\nComfyUI PID={server.pid}\nTunnel PID={tunnel.pid}\nHTTP={http_status()}\n{message}")
            if not healthy:
                raise RuntimeError(f"{message}\nComfyUI log:\n{log_tail(log_path)}")
            time.sleep(heartbeat_seconds)
    except KeyboardInterrupt:
        print("Stopping ComfyUI TPU tunnel.")
    finally:
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
        if server.poll() is None:
            server.terminate()
        log_file.close()
