"""Clearly labelled synthetic evidence for exercising the offline report path."""

from __future__ import annotations

import copy
from pathlib import Path

from .artifacts import IDENTITY_FIELDS, atomic_write, create_campaign, seal, write_json, write_jsonl
from .budget import Ledger
from .config import Config
from .engines import command
from .metrics import summarize
from .provenance import digest_json
from .qualification import PROBES

# Deliberately labelled fixture values. They exercise parsers and gates only.
SYNTHETIC_LOG = {
    "vllm": "SYNTHETIC FIXTURE LOG - no model was run\nGPU KV cache size: 400,000 tokens\n"
            "Chunked prefill is enabled with max_num_batched_tokens=8192.\n",
    "sglang": "SYNTHETIC FIXTURE LOG - no model was run\nmax_total_num_tokens=400000, "
              "chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=32\n",
}
SYNTHETIC_REPLIES = {
    "eos_honored": "yes", "arithmetic": "42", "capital": "Paris",
    "passage_summary": (
        "SYNTHETIC FIXTURE summary text describing a public passage, its main events, "
        "the people involved, and why the account remains notable for later readers."
    ),
}


def _probe_record(context: dict, probe: dict, prompt: dict) -> dict:
    reply = SYNTHETIC_REPLIES.get(probe["name"], "SYNTHETIC FIXTURE continuation")
    tokens = probe["max_tokens"] if probe["ignore_eos"] else min(probe["max_tokens"] - 1, 2)
    return {
        **context, "probe_name": probe["name"], "schema_version": 1,
        "request_id": prompt["request_id"], "sequence_ordinal": 0,
        "prompt_hash": prompt["prompt_sha256"], "phase": "conformance",
        "expected_input_tokens": len(prompt["prompt_ids"]),
        "requested_output_tokens": probe["max_tokens"], "dispatch_ns": 0, "headers_ns": 1000,
        "first_content_ns": 2000, "last_content_ns": 3000, "done_ns": 4000, "error_ns": None,
        "http_status": 200, "finish_reason": "length" if probe["ignore_eos"] else "stop",
        "actual_input_tokens": len(prompt["prompt_ids"]), "actual_output_tokens": tokens,
        "content_event_count": 1, "status": "success", "error_category": None,
        "error_message": None, "response_text": reply, "sampling_seed": 0,
    }


def _cell_records(metadata: dict, shape: dict, count: int, phase: str):
    workers = [i * 1000 for i in range(metadata["concurrency"])]
    records, events, prompts = [], [], []
    for ordinal in range(count):
        worker = min(range(metadata["concurrency"]), key=workers.__getitem__)
        dispatch = workers[worker]
        # Arbitrary deterministic timings, never an engine performance prediction.
        factor = 1 + metadata["repetition"] * 0.01 + (0.03 if metadata["engine"] == "sglang" else 0)
        first = dispatch + round(shape["input_tokens"] * 1_000_000 * factor)
        last = first + round((shape["output_tokens"] - 1) * 20_000_000 * factor)
        done = last + 1_000_000
        workers[worker] = done
        request_id = f'{phase}:{metadata["profile"]}:r{metadata["repetition"]}:{ordinal:04d}'
        ids = list(range(100 + ordinal, 100 + ordinal + shape["input_tokens"]))
        prompts.append({
            "request_id": request_id, "prompt_ids": ids, "prompt_sha256": digest_json(ids),
            "phase": phase, "profile": metadata["profile"], "repetition": metadata["repetition"],
        })
        records.append({
            **metadata, "schema_version": 1, "request_id": request_id,
            "sequence_ordinal": ordinal, "prompt_hash": digest_json(ids),
            "phase": "warmup" if phase == "pilot_warmup" else phase,
            "expected_input_tokens": len(ids), "requested_output_tokens": shape["output_tokens"],
            "dispatch_ns": dispatch, "headers_ns": dispatch + 1000,
            "first_content_ns": first, "last_content_ns": last, "done_ns": done,
            "error_ns": None, "http_status": 200, "finish_reason": "length",
            "actual_input_tokens": len(ids), "actual_output_tokens": shape["output_tokens"],
            "content_event_count": 2, "status": "success",
            "error_category": None, "error_message": None, "response_text": "SYNTHETIC FIXTURE",
        })
        events.extend([
            {"request_id": request_id, "received_ns": first, "data": "SYNTHETIC"},
            {"request_id": request_id, "received_ns": last, "data": " FIXTURE"},
            {"request_id": request_id, "received_ns": done, "data": "[DONE]"},
        ])
    return records, events, prompts


def make_example(
    root: Path, config: Config, sources: dict, engines: dict, *, phase: str = "measurement",
) -> dict:
    data = copy.deepcopy(config.data)
    if phase == "measurement":
        data["experiment"].update(requests_per_cell=8, concurrency=[1, 4], repetitions=3)
    data["profiles"] = {
        "short": {"input_tokens": 16, "output_tokens": 4},
        "prefill": {"input_tokens": 32, "output_tokens": 4},
        "decode": {"input_tokens": 16, "output_tokens": 8},
    }
    config = Config(data)
    campaign = create_campaign(
        root, config, sources, engines, campaign_id="synthetic-example",
        evidence_kind="synthetic", harness={
            "commit": None, "dirty": True, "note": "Generated fixture; no model or GPU",
        }, phase=phase,
    )
    context = {key: campaign[key] for key in IDENTITY_FIELDS}
    saved_prompts = {}
    blocks = {}
    for cell in config.cells(phase=phase):
        metadata = {**context, **cell.to_dict(), "attempt": 1, "cell_id": cell.key}
        path = (
            root / "runs" / cell.engine / f"r{cell.repetition}" / "attempt-1"
            / "cells" / f"{cell.profile}-c{cell.concurrency}"
        )
        shape = config.profiles[cell.profile]
        count = config.request_count(cell.concurrency, phase=phase)
        records, events, prompts = _cell_records(metadata, shape, count, phase)
        saved_prompts.update((p["request_id"], p) for p in prompts)
        summary = summarize(records, expected_requests=count)
        write_json(path / "manifest.json", metadata)
        write_jsonl(path / "requests.jsonl.gz", records)
        write_jsonl(path / "stream-events.jsonl.gz", events)
        write_json(path / "summary.json", summary)
        write_json(path / "status.json", {"status": "complete"})
        write_json(path / "client-diagnostics.json", {
            "loop_lag_ms": [0.1, 0.2, 0.3, 0.2], "lag_interval_ms": 10, "synthetic": True,
        })
        if phase == "pilot":
            warm_count = config.warmup_count(cell.concurrency)
            warmup, _, prompts = _cell_records(metadata, shape, warm_count, "pilot_warmup")
            saved_prompts.update((p["request_id"], p) for p in prompts)
            write_jsonl(path / "warmup.jsonl.gz", warmup)
            block = blocks.setdefault(path.parent.parent, {"cells": [], "timed_ns": 0})
            block["cells"].append(cell.to_dict())
            block["timed_ns"] += round(1e9 * (
                summary["wall_time_s"] + summarize(warmup, expected_requests=warm_count)["wall_time_s"]
            ))
        seal(path)
    probes = []
    for probe in PROBES:
        ids = list(range(500, 520))
        probes.append({
            "request_id": f'conformance:{probe["name"]}', "probe_name": probe["name"],
            "phase": "conformance", "profile": None, "repetition": None, "prompt_ids": ids,
            "prompt_sha256": digest_json(ids), "input_tokens": len(ids), "stop_token_ids": [1],
        })
    for path, block in blocks.items():
        engine = block["cells"][0]["engine"]
        write_json(path / "manifest.json", {
            **context, "engine": engine, "phase": phase, "cells": block["cells"], "attempt": 1,
        })
        write_json(path / "status.json", {"status": "complete"})
        write_json(path / "timing.json", {
            "started_ns": 0, "ready_ns": 1_000_000_000,
            "finished_ns": block["timed_ns"] + 3_000_000_000,
        })
        write_json(path / "environment.json", {"synthetic": True, "devices": [
            {"name": "NVIDIA H200", "memory_mib": 143771, "mig_mode": "Disabled", "synthetic": True},
        ]})
        write_json(path / "models.json", {"data": [{"id": "gemma4-bench", "max_model_len": 8192}]})
        write_json(path / "server-command.json", {
            "argv": command(engine, "/models/synthetic-fixture", config),
        })
        atomic_write(path / "server.log", SYNTHETIC_LOG[engine].encode("utf-8"))
        write_jsonl(path / "conformance" / "requests.jsonl.gz", [
            _probe_record(context, probe, prompt) for probe, prompt in zip(PROBES, probes)
        ])
        seal(path)
    prompt_file = "measured.jsonl.gz" if phase == "measurement" else "pilot.jsonl.gz"
    write_jsonl(root / "prompts" / prompt_file, saved_prompts.values())
    write_jsonl(root / "prompts" / "conformance.jsonl.gz", probes)
    # This is a zero-usage fixture ledger, not an invented billing receipt.
    ledger = Ledger(root / "budget-ledger.json", config)
    write_json(ledger.path, ledger.read())
    seal(root)
    return campaign
