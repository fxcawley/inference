"""Local atomic writes and self-contained, checked experiment directories."""

from __future__ import annotations

import gzip
import os
import tempfile
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .provenance import (
    canonical_json, digest_file, digest_json, measurement_errors, parse_json, read_json,
    source_tree_hash,
)

MARKERS = {"CHECKSUMS.json", "COMPLETE.json"}


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, (canonical_json(value) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    data = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    atomic_write(path, gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return [parse_json(line) for line in source if line.strip()]


def safe_path(root: Path, relative: str) -> Path:
    name = PurePosixPath(relative)
    if not relative or name.is_absolute() or ".." in name.parts or "\\" in relative:
        raise ValueError(f"Unsafe artifact path: {relative}")
    result = root.joinpath(*name.parts)
    if any(p.is_symlink() for p in (root, result, *result.parents) if p == root or root in p.parents):
        raise ValueError(f"Symlink in artifact path: {relative}")
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escaped its directory: {relative}")
    return result


def _files(root: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        safe_path(root, relative)
        if path.is_file() and relative not in MARKERS:
            result[relative] = path
    return result


def seal(root: Path) -> dict:
    if (root / "COMPLETE.json").exists():
        raise ValueError("Refusing to overwrite a completed artifact")
    files = _files(root)
    if not files:
        raise ValueError("Cannot seal an empty artifact")
    manifest = {
        "schema_version": 1,
        "files": {
            name: {"sha256": digest_file(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        },
    }
    write_json(root / "CHECKSUMS.json", manifest)
    write_json(root / "COMPLETE.json", {
        "schema_version": 1, "manifest_sha256": digest_file(root / "CHECKSUMS.json"),
    })
    return manifest


def verify(root: Path) -> dict:
    root = Path(root)
    marker = read_json(safe_path(root, "COMPLETE.json"))
    manifest_path = safe_path(root, "CHECKSUMS.json")
    if marker.get("schema_version") != 1 or marker.get("manifest_sha256") != digest_file(manifest_path):
        raise ValueError("Completion marker does not match checksum manifest")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported checksum manifest")
    if not manifest["files"] or set(_files(root)) != set(manifest["files"]):
        raise ValueError("Artifact file set differs from the sealed manifest")
    for relative, expected in manifest["files"].items():
        path = safe_path(root, relative)
        if path.stat().st_size != expected["bytes"] or digest_file(path) != expected["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    return manifest


IDENTITY_FIELDS = (
    "campaign_id", "evidence_kind", "protocol_sha256", "config_sha256",
    "sources_sha256", "engines_sha256", "harness_sha256",
)


def campaign_phase(manifest: dict) -> str:
    kind = manifest["evidence_kind"]
    expected = "pilot" if kind == "pilot" else "measurement"
    phase = manifest.get("phase", expected)
    if phase not in ("measurement", "pilot") or (kind != "synthetic" and phase != expected):
        raise ValueError("Campaign phase differs from its evidence kind")
    return phase


def create_campaign(
    root: Path, config, sources: dict, engines: dict, *, campaign_id: str,
    evidence_kind: str, harness: dict, phase: str | None = None,
) -> dict:
    if evidence_kind not in ("measurement", "pilot", "synthetic"):
        raise ValueError("Unknown evidence kind")
    phase = campaign_phase({
        "evidence_kind": evidence_kind,
        **({"phase": phase} if phase is not None else {}),
    })
    harness = {**harness, "source_sha256": source_tree_hash()}
    if evidence_kind == "measurement":
        problems = measurement_errors(config.data, sources, engines, harness)
        if problems:
            raise ValueError("Formal campaign is not ready: " + "; ".join(problems))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    identities = {
        "config_sha256": config.fingerprint,
        "sources_sha256": digest_json(sources), "engines_sha256": digest_json(engines),
        "harness_sha256": harness["source_sha256"],
    }
    manifest = {
        "schema_version": 1, "campaign_id": campaign_id, "evidence_kind": evidence_kind,
        **identities, "protocol_sha256": digest_json(identities),
        "harness": harness, "phase": phase,
        "schedule": [c.to_dict() for c in config.cells(phase=phase)],
    }
    write_json(root / "campaign.json", manifest)
    write_json(root / "protocol.json", config.data)
    write_json(root / "sources.lock.json", sources)
    write_json(root / "engines.lock.json", engines)
    return manifest


def bundle(root: Path, output: Path, *, allow_synthetic: bool = False) -> dict:
    root, output = Path(root), Path(output)
    verify(root)
    campaign = read_json(root / "campaign.json")
    if campaign.get("evidence_kind") == "synthetic" and not allow_synthetic:
        raise ValueError("Synthetic fixtures require --allow-synthetic")
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("The bundle must be written outside its sealed input")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing a published bundle.
    with output.open("xb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in sorted(root.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(root).as_posix()
                    safe_path(root, relative)
                    info = tarfile.TarInfo(relative)
                    info.size = path.stat().st_size
                    info.mode = 0o644
                    with path.open("rb") as source:
                        archive.addfile(info, source)
    receipt = {"file": output.name, "sha256": digest_file(output), "bytes": output.stat().st_size}
    write_json(output.with_name(output.name + ".sha256.json"), receipt)
    return receipt
