"""Candidate argv builders; real container qualification remains explicit."""

from __future__ import annotations

import csv
import io
import subprocess

from .config import Config


def command(engine: str, model_path: str, config: Config, port: int = 8000) -> list[str]:
    if engine not in ("vllm", "sglang"):
        raise ValueError(f"Unsupported engine: {engine}")
    if not model_path or not 1 <= port <= 65535:
        raise ValueError("A model path and valid port are required")
    e, h = config.experiment, config.hardware
    common = [
        "--served-model-name", "gemma4-bench",
        "--host", "127.0.0.1", "--port", str(port),
        "--dtype", h["dtype"], "--kv-cache-dtype", h["kv_cache_dtype"],
    ]
    if engine == "vllm":
        return [
            "vllm", "serve", model_path, *common,
            "--tensor-parallel-size", "1", "--max-model-len", str(e["max_model_len"]),
            "--max-num-seqs", str(e["max_running_requests"]),
            "--gpu-memory-utilization", str(h["memory_fraction"]),
            "--no-enable-prefix-caching", "--generation-config", "vllm",
            "--disable-log-requests",
        ]
    return [
        "sglang", "serve", "--model-path", model_path, *common,
        "--tp", "1", "--context-length", str(e["max_model_len"]),
        "--max-running-requests", str(e["max_running_requests"]),
        "--mem-fraction-static", str(h["memory_fraction"]),
        "--disable-radix-cache", "--sampling-defaults", "openai",
        "--stream-interval", "1",
    ]


def validate_hardware(devices: list[dict]) -> None:
    if len(devices) != 1:
        raise ValueError("Exactly one GPU is required")
    device = devices[0]
    if "H200" not in device.get("name", ""):
        raise ValueError("The allocated GPU is not an H200")
    if not 130_000 <= device.get("memory_mib", 0) <= 150_000:
        raise ValueError("The allocated GPU does not have the full H200 memory class")
    if device.get("mig_mode") != "Disabled":
        raise ValueError("MIG/partition state must be confirmed disabled")


def inspect_hardware() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,uuid,mig.mode.current",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    devices = []
    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        if len(row) != 5:
            raise ValueError("Unexpected nvidia-smi output")
        name, memory, driver, device_id, mig = [item.strip() for item in row]
        from .provenance import digest_json
        devices.append({
            "name": name, "memory_mib": int(memory), "driver": driver,
            "device_id_sha256": digest_json(device_id), "mig_mode": mig,
        })
    validate_hardware(devices)
    return devices
