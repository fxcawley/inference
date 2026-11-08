"""Read-only campaign plans from local artifacts and the persistent budget ledger."""

from __future__ import annotations

import copy
import math
from decimal import Decimal
from pathlib import Path

from .analysis import load_campaign, load_cell_summary
from .artifacts import IDENTITY_FIELDS, campaign_phase, safe_path, verify
from .budget import BudgetError, Ledger
from .config import Config
from .provenance import read_json, source_tree_hash

TIMING_HEADROOM = Decimal("1.25")
# The runner cannot observe container initialization or the final Volume commit.
# Keep the planned 180-second startup allowance and 30-second shutdown allowance.
CONTAINER_OVERHEAD_SECONDS = 210


def _ledger_summary(path: Path, manifest: dict, config: Config) -> tuple[dict, list[str]]:
    path = Path(path)
    if not path.is_file():
        raise BudgetError("An existing project ledger is required; a missing ledger is not zero usage")
    summary = Ledger(path, config).summary()
    blockers = []
    if path.with_name(path.name + ".lock").exists():
        blockers.append("Ledger is locked; resolve the previous controller")
    if summary["unresolved"]:
        blockers.append("Outstanding reservations require confirmation that remote work stopped")
    if summary["unreconciled_overruns"]:
        blockers.append("Reservation overruns require billing reconciliation through their end")
    if manifest["harness_sha256"] != source_tree_hash():
        blockers.append("Current harness source differs from the frozen campaign")
    return summary, blockers


def _check_budget_links(root: Path, manifest: dict, cells: list[dict], summary: dict) -> None:
    if manifest["evidence_kind"] == "synthetic":
        return
    if any(
        cell["attempt"] and cell.get("reservation_id") not in summary["reservation_ids"]
        for cell in cells
    ):
        raise BudgetError("Recorded attempts lack their budget reservation history")
    # An allocation can fail before its first cell. Its receipt still belongs to
    # the budget history, even when there are no request records to inspect.
    for directory in sorted(root.glob("runs/*/*/*")):
        safe_path(root, directory.relative_to(root).as_posix())
        if not directory.is_dir():
            continue
        path = safe_path(directory, "manifest.json")
        if not path.is_file():
            raise BudgetError("A block lacks its reservation receipt; recover its identity first")
        if safe_path(directory, "COMPLETE.json").exists():
            verify(directory)
        block = read_json(path)
        if (
            any(block.get(key) != manifest[key] for key in IDENTITY_FIELDS)
            or block.get("phase") != campaign_phase(manifest)
            or block.get("reservation_id") not in summary["reservation_ids"]
            or not isinstance(block.get("cells"), list) or not block["cells"]
            or any(c not in manifest["schedule"] or c["engine"] != block.get("engine") for c in block["cells"])
        ):
            raise BudgetError("Block receipt differs from its campaign or budget history")
        prefix = directory.relative_to(root).as_posix() + "/cells/"
        if any(
            cell["attempt"] and cell["artifact"].startswith(prefix)
            and cell["reservation_id"] != block["reservation_id"]
            for cell in cells
        ):
            raise BudgetError("A cell's reservation differs from its block receipt")


def protocol_candidates(config: Config) -> list[tuple[str, Config]]:
    """Apply only the agreed symmetric reductions, in their declared order."""
    if (
        config.experiment["concurrency"] != [1, 4, 16, 32]
        or config.experiment["repetitions"] != 3
        or config.experiment["requests_per_cell"] != 256
        or set(config.profiles) != {"short", "prefill", "decode"}
    ):
        raise ValueError("Forecasting starts with a pilot of the full proposed matrix")
    data = copy.deepcopy(config.data)
    result = [("full", Config(copy.deepcopy(data)))]
    data["experiment"]["concurrency"] = [1, 4, 16]
    result.append(("without-c32", Config(copy.deepcopy(data))))
    del data["profiles"]["decode"]
    result.append(("two-profiles", Config(copy.deepcopy(data))))
    data["experiment"]["requests_per_cell"] = 200
    result.append(("two-profiles-200", Config(copy.deepcopy(data))))
    return result


def _pilot_observations(root: Path, manifest: dict, config: Config, cells: list[dict]):
    if campaign_phase(manifest) != "pilot":
        raise ValueError("A forecast requires pilot evidence, not formal measurements")
    if len(cells) != len(manifest["schedule"]) or any(
        not cell["valid"] or cell["attempt"] != 1 for cell in cells
    ):
        raise ValueError("Pilot coverage must be complete and valid for both engines")
    paired = {}
    blocks = {}
    for cell in cells:
        directory = safe_path(root, cell["artifact"])
        warmup = load_cell_summary(directory, config, phase="pilot", warmup=True)
        if not warmup["valid"]:
            raise ValueError("Pilot warmup is incomplete or invalid")
        measured_s = Decimal(str(cell["wall_time_s"]))
        warmup_s = Decimal(str(warmup["wall_time_s"]))
        key = (cell["profile"], cell["concurrency"])
        pair = paired.setdefault(key, {"seconds_per_request": Decimal(0), "warmup_s": Decimal(0)})
        pair["seconds_per_request"] = max(
            pair["seconds_per_request"], measured_s / cell["expected_requests"],
        )
        pair["warmup_s"] = max(pair["warmup_s"], warmup_s)
        block = blocks.setdefault(directory.parent.parent, {"timed_s": Decimal(0), "engines": set()})
        block["timed_s"] += measured_s + warmup_s
        block["engines"].add(cell["engine"])
    if len(blocks) != 2:
        raise ValueError("Pilot forecast requires one complete block per engine")
    startup_s, overhead_s = Decimal(0), Decimal(0)
    seen_engines = set()
    for directory, observed in blocks.items():
        verify(directory)
        metadata = read_json(directory / "manifest.json")
        engine = metadata.get("engine")
        if (
            any(metadata.get(key) != manifest[key] for key in IDENTITY_FIELDS)
            or metadata.get("phase") != "pilot"
            or engine not in ("vllm", "sglang")
            or observed["engines"] != {engine}
            or engine in seen_engines
            or metadata.get("cells") != [c for c in manifest["schedule"] if c["engine"] == engine]
            or read_json(directory / "status.json").get("status") != "complete"
        ):
            raise ValueError("Pilot block provenance, schedule, or completion state differs")
        seen_engines.add(engine)
        timing = read_json(directory / "timing.json")
        times = [timing.get(k) for k in ("started_ns", "ready_ns", "finished_ns")]
        if any(type(t) is not int or t < 0 for t in times) or times != sorted(times):
            raise ValueError("Pilot block needs monotonic startup and completion timings")
        startup = Decimal(times[1] - times[0]) / 1_000_000_000
        overhead = Decimal(times[2] - times[1]) / 1_000_000_000 - observed["timed_s"]
        if overhead < 0:
            raise ValueError("Pilot cell windows exceed their block duration")
        startup_s, overhead_s = max(startup_s, startup), max(overhead_s, overhead)
    return paired, startup_s, overhead_s


def estimate_protocols(
    config: Config, paired: dict, startup_s: Decimal, overhead_s: Decimal,
) -> list[dict]:
    pilot_requests = sum(
        config.request_count(c, phase="pilot") + config.warmup_count(c)
        for _ in config.profiles for c in config.experiment["concurrency"]
    )
    estimates = []
    for name, candidate in protocol_candidates(config):
        target_requests = sum(
            candidate.request_count(c) + candidate.warmup_count(c)
            for _ in candidate.profiles for c in candidate.experiment["concurrency"]
        )
        # Serialization and artifact writes grow with the request volume. In the
        # absence of a finer breakdown, scale all observed noncell overhead.
        overhead_scale = max(Decimal(1), Decimal(target_requests) / pilot_requests)
        per_block = startup_s + overhead_s * overhead_scale + CONTAINER_OVERHEAD_SECONDS
        for profile in candidate.profile_names:
            for concurrency in candidate.experiment["concurrency"]:
                observation = paired[(profile, concurrency)]
                per_block += (
                    observation["warmup_s"]
                    + observation["seconds_per_request"] * candidate.experiment["requests_per_cell"]
                )
        block_seconds = math.ceil(per_block * TIMING_HEADROOM)
        total_seconds = block_seconds * 2 * candidate.experiment["repetitions"]
        estimates.append({
            "name": name, "config": candidate.data, "config_sha256": candidate.fingerprint,
            "cells": len(candidate.cells()), "blocks": 2 * candidate.experiment["repetitions"],
            "block_seconds": block_seconds, "total_seconds": total_seconds,
            "noncell_overhead_scale": str(overhead_scale),
            "nominal_container_usd": str(candidate.rate_per_second * total_seconds),
        })
    return estimates


def forecast_campaign(
    root: Path, ledger_path: Path, *, allow_synthetic: bool = False,
) -> dict:
    root = Path(root)
    manifest, config, cells = load_campaign(root, allow_synthetic=allow_synthetic)
    summary, blockers = _ledger_summary(ledger_path, manifest, config)
    _check_budget_links(root, manifest, cells, summary)
    if any(summary["stages"][stage]["reservations"] for stage in ("formal", "rerun")):
        blockers.append("Formal work has already started; its protocol cannot be resized")
    paired, startup_s, overhead_s = _pilot_observations(root, manifest, config, cells)
    estimates = estimate_protocols(config, paired, startup_s, overhead_s)
    available = summary["remaining"]
    multiplier = Decimal(summary["billing"]["rate_multiplier"])
    for estimate in estimates:
        reasons = []
        estimate["projected_container_usd"] = str(Decimal(estimate["nominal_container_usd"]) * multiplier)
        if estimate["block_seconds"] > config.experiment["block_timeout_s"]:
            reasons.append("block duration exceeds its limit")
        if estimate["total_seconds"] > available["formal_seconds"]:
            reasons.append("formal stage time allowance exceeded")
        if estimate["total_seconds"] > available["gpu_seconds"]:
            reasons.append("remaining campaign time allowance exceeded")
        if Decimal(estimate["projected_container_usd"]) > Decimal(available["gpu_container_usd"]):
            reasons.append("remaining campaign dollar allowance exceeded")
        estimate.update(fits_budget=not reasons, reasons=reasons)
    recommended = next((e["name"] for e in estimates if e["fits_budget"]), None) if not blockers else None
    return {
        "schema_version": 1, "advisory_only": True,
        "campaign_id": manifest["campaign_id"], "evidence_kind": manifest["evidence_kind"],
        "pilot_protocol_sha256": manifest["protocol_sha256"],
        "basis": {
            "timing_headroom": str(TIMING_HEADROOM),
            "container_overhead_seconds_per_block": CONTAINER_OVERHEAD_SECONDS,
            "observed_startup_seconds": str(startup_s),
            "observed_noncell_overhead_seconds": str(overhead_s),
            "billing_rate_multiplier": str(multiplier),
            "method": (
                "Slower engine per cell; maximum observed startup and warmup; "
                "noncell overhead scaled up with request volume"
            ),
        },
        "budget": summary, "available": available, "blockers": blockers,
        "candidates": estimates, "recommended_protocol": recommended,
        "stop_reason": None if recommended else (
            "Resolve blockers before freezing" if blockers else "No agreed protocol fits; stop after the pilot"
        ),
    }


def resume_plan(root: Path, ledger_path: Path, *, allow_synthetic: bool = False) -> dict:
    """Inspect incomplete work without changing artifacts, releasing leases, or dispatching."""
    root = Path(root)
    manifest, config, cells = load_campaign(
        root, allow_synthetic=allow_synthetic, allow_incomplete=True,
    )
    if campaign_phase(manifest) != "measurement":
        raise ValueError("Recovery planning applies to the frozen formal schedule")
    summary, blockers = _ledger_summary(ledger_path, manifest, config)
    _check_budget_links(root, manifest, cells, summary)
    available = summary["remaining"]
    rate = config.rate_per_second * Decimal(summary["billing"]["rate_multiplier"])
    maximum_reservation = {
        stage: max(0, min(
            config.experiment["block_timeout_s"], available[f"{stage}_seconds"],
            available["gpu_seconds"], math.floor(Decimal(available["gpu_container_usd"]) / rate),
        ))
        for stage in ("formal", "rerun")
    }
    history = {}
    for row in cells:
        key = (row["engine"], row["profile"], row["concurrency"], row["repetition"])
        history.setdefault(key, []).append(row)
    results, eligible = [], []
    for cell in config.cells():
        key = (cell.engine, cell.profile, cell.concurrency, cell.repetition)
        attempts = sorted(history[key], key=lambda row: row["attempt"])
        valid = next((row for row in attempts if row["valid"]), None)
        row = {
            **cell.to_dict(), "cell_id": cell.key,
            "attempts": [
                {k: a.get(k) for k in ("attempt", "status", "valid", "artifact", "replacement_for")}
                for a in attempts if a["attempt"]
            ],
        }
        if valid:
            row.update(state="complete", selected_artifact=valid["artifact"], next_attempt=None)
        elif attempts[-1]["attempt"] == 2:
            row.update(state="exhausted", selected_artifact=None, next_attempt=None)
        else:
            attempt = attempts[-1]["attempt"] + 1
            stage = "formal" if attempt == 1 else "rerun"
            row.update(
                state="pending", next_attempt=attempt, budget_stage=stage,
                replacement_for=attempts[-1]["artifact"] if attempt == 2 else None,
            )
            if blockers:
                row["state"] = "blocked"
            elif maximum_reservation[stage] <= 30:
                row["state"] = "budget_stopped"
            else:
                eligible.append({
                    **cell.to_dict(), "attempt": attempt, "budget_stage": stage,
                    "replacement_for": row["replacement_for"],
                    "requires_new_reservation": True, "requires_fresh_process": True,
                    "maximum_reservation_seconds": maximum_reservation[stage],
                })
        results.append(row)
    return {
        "schema_version": 1, "advisory_only": True,
        "campaign_id": manifest["campaign_id"], "evidence_kind": manifest["evidence_kind"],
        "protocol_sha256": manifest["protocol_sha256"], "budget": summary, "available": available,
        "blockers": blockers, "cells": results, "eligible_cells": eligible,
        "counts": {
            state: sum(row["state"] == state for row in results)
            for state in ("complete", "pending", "blocked", "budget_stopped", "exhausted")
        },
    }
