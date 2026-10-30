"""Clearly labelled synthetic evidence for exercising the offline report path."""

from __future__ import annotations

import copy
from pathlib import Path

from .artifacts import create_campaign, seal, write_json, write_jsonl
from .config import Config
from .metrics import summarize
from .provenance import digest_json


def make_example(root: Path, config: Config, sources: dict, engines: dict) -> dict:
    data = copy.deepcopy(config.data)
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
        },
    )
    from .artifacts import IDENTITY_FIELDS
    context = {key: campaign[key] for key in IDENTITY_FIELDS}
    saved_prompts = {}
    for cell in config.cells():
        metadata = {**context, **cell.to_dict(), "attempt": 1, "cell_id": cell.key}
        path = (
            root / "runs" / cell.engine / f"r{cell.repetition}" / "attempt-1"
            / "cells" / f"{cell.profile}-c{cell.concurrency}"
        )
        shape = config.profiles[cell.profile]
        workers = [i * 1000 for i in range(cell.concurrency)]
        records, events = [], []
        for ordinal in range(config.experiment["requests_per_cell"]):
            worker = min(range(cell.concurrency), key=workers.__getitem__)
            dispatch = workers[worker]
            # Arbitrary deterministic timings, never an engine performance prediction.
            factor = 1 + cell.repetition * 0.01 + (0.03 if cell.engine == "sglang" else 0)
            first = dispatch + round(shape["input_tokens"] * 1_000_000 * factor)
            last = first + round((shape["output_tokens"] - 1) * 20_000_000 * factor)
            done = last + 1_000_000
            workers[worker] = done
            request_id = f"measurement:{cell.profile}:r{cell.repetition}:{ordinal:04d}"
            ids = list(range(100 + ordinal, 100 + ordinal + shape["input_tokens"]))
            saved_prompts[request_id] = {
                "request_id": request_id, "prompt_ids": ids, "prompt_sha256": digest_json(ids),
                "phase": "measurement", "profile": cell.profile, "repetition": cell.repetition,
            }
            row = {
                **metadata, "schema_version": 1, "request_id": request_id,
                "sequence_ordinal": ordinal, "prompt_hash": digest_json(ids),
                "phase": "measurement", "expected_input_tokens": len(ids),
                "requested_output_tokens": shape["output_tokens"],
                "dispatch_ns": dispatch, "headers_ns": dispatch + 1000,
                "first_content_ns": first, "last_content_ns": last, "done_ns": done,
                "error_ns": None, "http_status": 200, "finish_reason": "length",
                "actual_input_tokens": len(ids), "actual_output_tokens": shape["output_tokens"],
                "content_event_count": 2, "status": "success",
                "error_category": None, "error_message": None,
                "response_text": "SYNTHETIC FIXTURE",
            }
            records.append(row)
            events.extend([
                {"request_id": request_id, "received_ns": first, "data": "SYNTHETIC"},
                {"request_id": request_id, "received_ns": last, "data": " FIXTURE"},
                {"request_id": request_id, "received_ns": done, "data": "[DONE]"},
            ])
        write_json(path / "manifest.json", metadata)
        write_jsonl(path / "requests.jsonl.gz", records)
        write_jsonl(path / "stream-events.jsonl.gz", events)
        write_json(path / "summary.json", summarize(records, expected_requests=8))
        write_json(path / "status.json", {"status": "complete"})
        seal(path)
    write_jsonl(root / "prompts" / "measured.jsonl.gz", saved_prompts.values())
    seal(root)
    return campaign
