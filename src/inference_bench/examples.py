"""Clearly labelled synthetic evidence for exercising the offline report path."""

from __future__ import annotations

import copy
from pathlib import Path

from .artifacts import IDENTITY_FIELDS, create_campaign, seal, write_json, write_jsonl
from .budget import Ledger
from .config import Config
from .metrics import summarize
from .provenance import digest_json


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
    for path, block in blocks.items():
        write_json(path / "manifest.json", {
            **context, "engine": block["cells"][0]["engine"], "phase": phase,
            "cells": block["cells"], "attempt": 1,
        })
        write_json(path / "status.json", {"status": "complete"})
        write_json(path / "timing.json", {
            "started_ns": 0, "ready_ns": 1_000_000_000,
            "finished_ns": block["timed_ns"] + 3_000_000_000,
        })
        seal(path)
    prompt_file = "measured.jsonl.gz" if phase == "measurement" else "pilot.jsonl.gz"
    write_jsonl(root / "prompts" / prompt_file, saved_prompts.values())
    # This is a zero-usage fixture ledger, not an invented billing receipt.
    ledger = Ledger(root / "budget-ledger.json", campaign["campaign_id"], config)
    write_json(ledger.path, ledger.read())
    seal(root)
    return campaign
