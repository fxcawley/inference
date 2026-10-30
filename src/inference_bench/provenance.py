"""Canonical identities and explicit qualification checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


def parse_json(text: str) -> Any:
    return json.loads(
        text, object_pairs_hook=_unique, parse_constant=_invalid_constant,
    )


def read_json(path: Path) -> Any:
    return parse_json(path.read_text(encoding="utf-8"))


def is_hash(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[a-f0-9]{{{length}}}", value) is not None


def qualification_errors(sources: dict, engines: dict) -> list[str]:
    """Pending metadata cannot accidentally pass a formal-run gate."""
    errors = []
    if sources.get("schema_version") != 1 or sources.get("status") != "complete":
        errors.append("source preparation is incomplete")
    for name in ("model", "dataset"):
        source = sources.get(name, {})
        if not is_hash(source.get("revision"), 40):
            errors.append(f"{name} needs an immutable upstream commit")
        files = source.get("files")
        if not isinstance(files, dict) or not files:
            errors.append(f"{name} file hashes are missing")
        else:
            for filename, record in files.items():
                if (
                    Path(filename).is_absolute() or ".." in Path(filename).parts
                    or not isinstance(record, dict) or not is_hash(record.get("sha256"))
                    or type(record.get("bytes")) is not int or record["bytes"] <= 0
                ):
                    errors.append(f"{name} has an invalid file record: {filename}")
    if not sources.get("dataset", {}).get("license"):
        errors.append("dataset license/attribution is unresolved")
    if not is_hash(sources.get("prompts", {}).get("manifest_sha256")):
        errors.append("prepared prompts are not pinned")
    if engines.get("schema_version") != 1:
        errors.append("unsupported engine-lock schema")
    client = engines.get("client", {})
    if not re.fullmatch(r"3\.12\.\d+", str(client.get("python_version"))):
        errors.append("common client Python patch version is unresolved")
    if not is_hash(client.get("requirements_sha256")):
        errors.append("common client dependency lock is unresolved")
    for name in ("vllm", "sglang"):
        engine = engines.get("engines", {}).get(name, {})
        if not re.fullmatch(r"[^@\s]+@sha256:[a-f0-9]{64}", engine.get("image", "")):
            errors.append(f"{name} image is not pinned by digest")
        qualification = engine.get("qualification", {})
        if qualification.get("status") != "passed" or not is_hash(
            qualification.get("report_sha256")
        ):
            errors.append(f"{name} real-model qualification is pending")
    return errors


def measurement_errors(configuration: dict, sources: dict, engines: dict, harness: dict) -> list[str]:
    errors = qualification_errors(sources, engines)
    experiment, workload = configuration["experiment"], configuration["workload"]
    if sources.get("model", {}).get("repo_id") != experiment["model"]:
        errors.append("model lock differs from configuration")
    dataset = sources.get("dataset", {})
    if any(dataset.get(key) != workload[target] for key, target in (
        ("repo_id", "dataset"), ("configuration", "dataset_config"), ("split", "split"),
    )):
        errors.append("dataset lock differs from configuration")
    if harness.get("dirty") is not False or not is_hash(harness.get("commit"), 40):
        errors.append("formal measurements require a clean committed harness")
    if experiment["repetitions"] != 3 or experiment["requests_per_cell"] not in (200, 256):
        errors.append("formal protocol requires three repetitions and 200 or 256 requests per cell")
    if experiment["concurrency"] not in ([1, 4, 16], [1, 4, 16, 32]):
        errors.append("formal concurrency matrix differs from the full/compact protocol")
    expected = {
        "short": {"input_tokens": 512, "output_tokens": 128},
        "prefill": {"input_tokens": 4096, "output_tokens": 128},
        "decode": {"input_tokens": 512, "output_tokens": 512},
    }
    profiles = configuration["profiles"]
    if (
        set(profiles) not in (set(expected), {"short", "prefill"})
        or any(shape != expected.get(name) for name, shape in profiles.items())
    ):
        errors.append("formal workload shapes differ from the full/compact protocol")
    return errors


def git_identity(directory: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(directory), "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def environment() -> dict:
    packages = {}
    for name in ("httpx", "httpcore", "anyio", "modal", "torch", "transformers", "sglang", "vllm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    limits = {}
    for name in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            limits[name] = path.read_text().strip()
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "available_logical_cpus": os.cpu_count(),
        "cgroup_v2_limits": limits,
        "packages": packages,
    }
