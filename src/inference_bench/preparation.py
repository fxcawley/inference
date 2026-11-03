"""CPU preparation boundaries; optional libraries are imported only on demand."""

from __future__ import annotations

from pathlib import Path

from .artifacts import read_jsonl, seal, write_json, write_jsonl
from .config import Config
from .provenance import digest_file, digest_json, is_hash, read_json
from .workloads import prepare_prompts, tokenize_corpus


def inspect_model(directory: Path) -> dict:
    directory = Path(directory)
    config = read_json(directory / "config.json")
    if config.get("quantization_config"):
        raise ValueError("The baseline does not accept a quantized checkpoint")
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise ValueError("No standard safetensors checkpoint shards were found")
    index = directory / "model.safetensors.index.json"
    if index.exists():
        mapping = read_json(index).get("weight_map", {})
        if not mapping or any(
            Path(name).name != name or not (directory / name).is_file()
            for name in mapping.values()
        ):
            raise ValueError("Checkpoint shard index is incomplete or invalid")
    if not (directory / "tokenizer_config.json").is_file():
        raise ValueError("Tokenizer configuration is missing")
    if not any((directory / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise ValueError("Tokenizer data is missing")
    files = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("Resolve model snapshot links before sealing its manifest")
        if path.is_file():
            files[relative.as_posix()] = {
                "bytes": path.stat().st_size, "sha256": digest_file(path),
            }
    return {
        "files": files, "architectures": config.get("architectures"),
        "declared_dtype": config.get("dtype", config.get("torch_dtype")),
        "text_config": config.get("text_config"),
        "qualification": "metadata_only",
    }


def resolve_sources(config: Config, *, api=None) -> dict:
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("Source resolution requires the preparation dependencies") from exc
        api = HfApi()
    model = api.model_info(config.experiment["model"])
    dataset = api.dataset_info(config.data["workload"]["dataset"])
    if not is_hash(model.sha, 40) or not is_hash(dataset.sha, 40):
        raise ValueError("Upstream did not return immutable commit identities")
    return {
        "schema_version": 1, "status": "resolved_not_downloaded",
        "model": {"repo_id": config.experiment["model"], "revision": model.sha, "files": {}},
        "dataset": {
            "repo_id": config.data["workload"]["dataset"], "revision": dataset.sha,
            "configuration": config.data["workload"]["dataset_config"],
            "split": config.data["workload"]["split"],
            "license": None, "files": {},
        },
        "prompts": {"manifest_sha256": None, "generator_version": None},
    }


def download_model(source: dict, directory: Path) -> dict:
    if not is_hash(source.get("revision"), 40):
        raise ValueError("Download requires an immutable model revision")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Model download requires the preparation dependencies") from exc
    snapshot_download(
        repo_id=source["repo_id"], revision=source["revision"], local_dir=directory,
        allow_patterns=[
            "*.safetensors", "*.json", "*.model", "*.jinja", "*.txt", "README.md", "LICENSE*",
        ],
    )
    return {**source, **inspect_model(directory)}


def prepare_local(
    config: Config, model_directory: Path, corpus_file: Path, output: Path, *, tokenizer=None,
) -> dict:
    """The caller supplies pinned public corpus rows as JSONL with a text field."""
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Load the tokenizer in the qualified CPU image; Transformers is not a controller dependency"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            model_directory, local_files_only=True, trust_remote_code=False,
        )
    rows = read_jsonl(corpus_file)
    if any(not isinstance(row, dict) or not isinstance(row.get("text"), str) for row in rows):
        raise ValueError("Corpus JSONL must contain objects with a text field")
    tokens, offsets = tokenize_corpus([row["text"] for row in rows], tokenizer)
    prompts, manifest = prepare_prompts(config, tokens, tokenizer)
    manifest.update(corpus_file_sha256=digest_file(corpus_file), source_offsets_sha256=digest_json(offsets))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    write_json(output / "source-offsets.json", offsets)
    write_jsonl(output / "measured.jsonl.gz", (p for p in prompts if p["phase"] == "measurement"))
    write_jsonl(output / "warmup.jsonl.gz", (p for p in prompts if p["phase"] == "warmup"))
    write_jsonl(output / "pilot.jsonl.gz", (p for p in prompts if p["phase"] == "pilot"))
    write_jsonl(output / "pilot-warmup.jsonl.gz", (p for p in prompts if p["phase"] == "pilot_warmup"))
    seal(output)
    return manifest
