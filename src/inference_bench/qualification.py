"""Shared conformance probes and verifiable engine-qualification reports."""

from __future__ import annotations

import re
from pathlib import Path
from statistics import median

from .artifacts import IDENTITY_FIELDS, campaign_phase, read_jsonl, safe_path, verify
from .config import Config
from .engines import (
    MODEL_PLACEHOLDER, command, engine_settings_sha256, normalized_argv, resolved_settings,
    validate_hardware,
)
from .metrics import quantile
from .provenance import digest_file, is_hash, read_json

# Probes run at C=1 before cells, outside every timing window. Their expectations
# live in source, so changing them changes the frozen harness identity.
PROBES = (
    {"name": "eos_honored", "message": "Reply with exactly one word: yes", "max_tokens": 64,
     "ignore_eos": False, "check": "eos_stop"},
    {"name": "eos_ignored", "message": "Reply with exactly one word: yes", "max_tokens": 64,
     "ignore_eos": True, "check": "exact_length"},
    {"name": "arithmetic", "message": "What is 17 + 25? Reply with only the number.",
     "max_tokens": 32, "ignore_eos": False, "check": "contains", "expected": "42"},
    {"name": "capital", "message": "What is the capital of France? Reply with one word.",
     "max_tokens": 32, "ignore_eos": False, "check": "contains", "expected": "paris"},
    {"name": "passage_summary", "message": None, "max_tokens": 256, "ignore_eos": False,
     "check": "plausible_text"},
)
PROBE_BY_NAME = {probe["name"]: probe for probe in PROBES}
REQUIRED_CHECKS = {
    "pilot_cells", "warmup_records", "warmup_coverage", "client_capacity", "hardware",
    "served_model", "engine_settings", "conformance", "kv_capacity", "cpu_preflight",
}
MAX_LOOP_LAG_P95_MS = 2.0
MAX_FIRST_REQUEST_RATIO = 1.5


def valid_token_ids(value) -> bool:
    return isinstance(value, list) and bool(value) and all(type(i) is int and i >= 0 for i in value)


def select_probes(prompts: list[dict]) -> list[tuple[dict, dict]]:
    found = {}
    for prompt in prompts:
        if prompt.get("phase") != "conformance":
            continue
        name = prompt.get("probe_name")
        if name not in PROBE_BY_NAME or name in found:
            raise ValueError("Prepared conformance probes are unknown or duplicated")
        if not valid_token_ids(prompt.get("stop_token_ids")):
            raise ValueError("Conformance probes need the pinned model's stop token IDs")
        found[name] = prompt
    if set(found) != set(PROBE_BY_NAME):
        raise ValueError("Prepared prompts must include every conformance probe")
    return [(probe, found[probe["name"]]) for probe in PROBES]


def plausible_text(text: str) -> bool:
    stripped = text.strip()
    words = re.findall(r"[^\W\d_]+", stripped.lower())
    printable = sum(char.isprintable() or char in "\n\t" for char in stripped)
    return (
        len(stripped) >= 40 and "\ufffd" not in stripped
        and printable >= 0.95 * len(stripped)
        and len(words) >= 8 and len(set(words)) >= 0.3 * len(words)
    )


def evaluate_probe(record: dict, probe: dict) -> dict:
    """A small behavioral sanity check, not an accuracy comparison."""
    text = record.get("response_text") or ""
    result = {
        "name": probe["name"], "status": "failed", "reason": None,
        "finish_reason": record.get("finish_reason"),
        "output_tokens": record.get("actual_output_tokens"), "text_preview": text[:200],
    }
    if record.get("probe_name") != probe["name"] or record.get("requested_output_tokens") != probe["max_tokens"]:
        result["reason"] = "Record does not match the canonical probe"
        return result
    if record.get("status") != "success":
        result["reason"] = f'Request {record.get("status")}: {record.get("error_message")}'
        return result
    tokens, finish = record["actual_output_tokens"], record["finish_reason"]
    natural_stop = finish == "stop" and tokens < probe["max_tokens"]
    if probe["check"] == "eos_stop":
        passed, reason = natural_stop, "EOS was not honored before the token limit"
    elif probe["check"] == "exact_length":
        passed = finish == "length" and tokens == probe["max_tokens"]
        reason = "Ignoring EOS did not reach the fixed target"
    elif probe["check"] == "contains":
        passed = natural_stop and probe["expected"] in text.lower()
        reason = "Expected answer or natural stop is missing"
    else:
        passed, reason = plausible_text(text), "Response is empty, garbled, or highly repetitive"
    result.update(status="passed" if passed else "failed", reason=None if passed else reason)
    return result


def qualify(
    root: Path, engine: str, *, preflight: Path | None = None,
    accepted: dict[str, str] | None = None, allow_synthetic: bool = False,
) -> dict:
    """Evaluate one engine's sealed pilot block; never contacts a server."""
    from .analysis import load_campaign, load_cell_summary

    if engine not in ("vllm", "sglang"):
        raise ValueError(f"Unsupported engine: {engine}")
    root = Path(root)
    manifest, config, cells = load_campaign(root, allow_synthetic=allow_synthetic)
    if campaign_phase(manifest) != "pilot":
        raise ValueError("Qualification requires a pilot campaign")
    sources = read_json(root / "sources.lock.json")
    lock = read_json(root / "engines.lock.json")["engines"][engine]
    rows = [cell for cell in cells if cell["engine"] == engine]
    blocks = {safe_path(root, cell["artifact"]).parent.parent for cell in rows if cell["artifact"]}
    if len(blocks) != 1:
        raise ValueError("Qualification needs exactly one pilot block for the engine")
    block = blocks.pop()
    verify(block)
    metadata = read_json(block / "manifest.json")
    if any(metadata.get(key) != manifest[key] for key in IDENTITY_FIELDS) or metadata.get("engine") != engine:
        raise ValueError("Pilot block provenance differs from its campaign")
    checks = []

    def add(name: str, status: str, **evidence):
        checks.append({"name": name, "status": status, "evidence": evidence})

    def label(cell: dict) -> str:
        return f'{cell["profile"]}-c{cell["concurrency"]}'

    invalid = sorted(label(c) for c in rows if not c["valid"] or c["attempt"] != 1)
    add("pilot_cells", "failed" if invalid else "passed", cells=len(rows), invalid=invalid)
    warm_invalid, first_ratios, lags = [], {}, {}
    for cell in rows:
        if not cell["artifact"]:
            warm_invalid.append(label(cell))
            continue
        directory = safe_path(root, cell["artifact"])
        try:
            warm_valid = load_cell_summary(directory, config, phase="pilot", warmup=True)["valid"]
        except (OSError, ValueError):
            warm_valid = False
        if not warm_valid:
            warm_invalid.append(label(cell))
        if not cell["valid"]:
            continue
        if cell["concurrency"] == 1:
            records = sorted(read_jsonl(directory / "requests.jsonl.gz"), key=lambda r: r["sequence_ordinal"])
            latencies = [r["done_ns"] - r["dispatch_ns"] for r in records]
            first_ratios[label(cell)] = latencies[0] / median(latencies)
        try:
            lags[label(cell)] = quantile(read_json(directory / "client-diagnostics.json")["loop_lag_ms"], 0.95)
        except (OSError, KeyError, TypeError, ValueError):
            lags[label(cell)] = None
    add("warmup_records", "failed" if warm_invalid else "passed", invalid=warm_invalid)
    worst_first = max(first_ratios.values(), default=None)
    add(
        "warmup_coverage",
        "passed" if worst_first is not None and worst_first <= MAX_FIRST_REQUEST_RATIO else "review",
        first_request_to_median_e2e=first_ratios, threshold=MAX_FIRST_REQUEST_RATIO,
    )
    add(
        "client_capacity",
        "passed" if lags and None not in lags.values() and max(lags.values()) <= MAX_LOOP_LAG_P95_MS
        else "failed",
        loop_lag_p95_ms=lags, threshold_ms=MAX_LOOP_LAG_P95_MS,
    )
    try:
        devices = read_json(block / "environment.json")["devices"]
        validate_hardware(devices)
        add("hardware", "passed", devices=devices)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        add("hardware", "failed", error=str(exc))
    try:
        models = read_json(block / "models.json")
        entry = next(m for m in models["data"] if isinstance(m, dict) and m.get("id") == "gemma4-bench")
    except (OSError, KeyError, TypeError, ValueError, StopIteration):
        entry = None
    if entry is None:
        status = "failed"
    elif entry.get("max_model_len") is None:
        status = "review"
    else:
        status = "passed" if entry["max_model_len"] == config.experiment["max_model_len"] else "failed"
    add("served_model", status, model=entry)
    expected = normalized_argv(engine, command(engine, MODEL_PLACEHOLDER, config))
    try:
        argv = read_json(block / "server-command.json")["argv"]
        matches = normalized_argv(engine, argv) == expected
    except (OSError, KeyError, TypeError, ValueError):
        argv, matches = None, False
    add("engine_settings", "passed" if matches else "failed", argv=argv)
    path = block / "conformance" / "requests.jsonl.gz"
    records = read_jsonl(path) if path.is_file() else []
    probes = []
    for record in records:
        if (
            any(record.get(key) != manifest[key] for key in IDENTITY_FIELDS)
            or record.get("phase") != "conformance" or record.get("probe_name") not in PROBE_BY_NAME
        ):
            probes.append({"name": record.get("probe_name"), "status": "failed", "reason": "Probe provenance differs"})
        else:
            probes.append(evaluate_probe(record, PROBE_BY_NAME[record["probe_name"]]))
    complete = sorted(p["name"] for p in probes if isinstance(p["name"], str)) == sorted(PROBE_BY_NAME)
    add(
        "conformance",
        "passed" if complete and len(probes) == len(PROBES) and all(p["status"] == "passed" for p in probes)
        else "failed",
        probes=probes,
    )
    log_path = block / "server.log"
    settings = resolved_settings(
        engine, log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else "",
    )
    required = max(config.experiment["concurrency"]) * max(
        shape["input_tokens"] + shape["output_tokens"] for shape in config.profiles.values()
    )
    capacity = settings["values"]["kv_cache_tokens"]
    add(
        "kv_capacity", "passed" if capacity is not None and capacity >= required else "review",
        kv_cache_tokens=capacity, required_tokens=required,
        note="Engine KV accounting differs; review sliding-window and preemption behavior before interpreting",
    )
    preflight_sha256 = None
    if preflight is None:
        add("cpu_preflight", "review", problems=["CPU preflight report was not supplied"])
    else:
        preflight = Path(preflight)
        verify(preflight)
        file = safe_path(preflight, f"{engine}.json")
        data = read_json(file)
        preflight_sha256 = digest_file(file)
        problems = []
        if data.get("upstream_image") != lock["image"]:
            problems.append("Preflight inspected a different image")
        if not data.get("engine_executable"):
            problems.append("Engine executable was not found")
        if not isinstance(data.get("cli_help"), dict) or data["cli_help"].get("returncode") != 0:
            problems.append("Engine CLI help did not complete")
        if data.get("flags_not_listed_in_help") != []:
            problems.append(f'Flags missing from CLI help: {data.get("flags_not_listed_in_help")}')
        try:
            if normalized_argv(engine, data.get("candidate_argv")) != expected:
                problems.append("Preflight argv differs from the pilot settings")
        except ValueError:
            problems.append("Preflight argv is invalid")
        add("cpu_preflight", "failed" if problems else "passed", problems=problems)
    for name, reason in (accepted or {}).items():
        target = next((c for c in checks if c["name"] == name), None)
        if target is None or target["status"] != "review" or not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Only checks requiring review can be accepted, with a reason: {name}")
        target.update(status="accepted", acceptance_reason=reason.strip())
    statuses = {check["status"] for check in checks}
    return {
        "schema_version": 1, "engine": engine, "evidence_kind": manifest["evidence_kind"],
        "status": "failed" if "failed" in statuses else "review_required" if "review" in statuses else "passed",
        "bindings": {
            "campaign_id": manifest["campaign_id"], "evidence_kind": manifest["evidence_kind"],
            "protocol_sha256": manifest["protocol_sha256"], "config_sha256": manifest["config_sha256"],
            "sources_sha256": manifest["sources_sha256"], "engines_sha256": manifest["engines_sha256"],
            "harness_sha256": manifest["harness_sha256"], "image": lock["image"],
            "model_revision": sources.get("model", {}).get("revision"),
            "engine_settings_sha256": engine_settings_sha256(engine, config),
            "preflight_sha256": preflight_sha256,
        },
        "checks": checks, "resolved_settings": settings,
    }


def report_errors(
    name: str, entry: dict, sources: dict, base: Path, *,
    configuration: dict | None = None, harness: dict | None = None,
) -> list[str]:
    """Verify the locked report itself, rather than trusting a typed status and hash."""
    qualification = entry.get("qualification", {})
    relative = qualification.get("report")
    if not isinstance(relative, str) or not relative:
        return [f"{name} qualification report path is missing"]
    try:
        path = safe_path(Path(base), relative)
    except ValueError:
        return [f"{name} qualification report path is unsafe"]
    if not path.is_file():
        return [f"{name} qualification report is missing"]
    if digest_file(path) != qualification.get("report_sha256"):
        return [f"{name} qualification report differs from its locked hash"]
    try:
        report = read_json(path)
        bindings, checks = report["bindings"], report["checks"]
        names = {check["name"] for check in checks}
        passed = all(check["status"] in ("passed", "accepted") for check in checks)
    except (KeyError, TypeError, ValueError):
        return [f"{name} qualification report is invalid"]
    if not isinstance(bindings, dict) or report.get("schema_version") != 1 or report.get("engine") != name:
        return [f"{name} qualification report is invalid"]
    errors = []
    if report.get("status") != "passed" or not passed or names != REQUIRED_CHECKS:
        errors.append(f"{name} qualification report did not pass every required check")
    if report.get("evidence_kind") != "pilot" or bindings.get("evidence_kind") != "pilot":
        errors.append(f"{name} qualification does not use real pilot evidence")
    if bindings.get("image") != entry.get("image"):
        errors.append(f"{name} qualification used a different image")
    revision = sources.get("model", {}).get("revision")
    if not is_hash(revision, 40) or bindings.get("model_revision") != revision:
        errors.append(f"{name} qualification used a different model revision")
    if configuration is not None and bindings.get("engine_settings_sha256") != engine_settings_sha256(
        name, Config(configuration),
    ):
        errors.append(f"{name} engine settings differ from the qualified settings")
    if harness is not None and bindings.get("harness_sha256") != harness.get("source_sha256"):
        errors.append(f"{name} measurement harness differs from the qualified harness")
    return errors
