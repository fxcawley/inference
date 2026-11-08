"""Serial GPU block dispatch with conservative leases; the remote backend is injected."""

from __future__ import annotations

import math
import time
from typing import Callable, Protocol

from .artifacts import IDENTITY_FIELDS, campaign_phase
from .budget import BudgetError, Ledger
from .config import CELL_FIELDS, Cell, Config
from .provenance import source_tree_hash

# Container start before the worker's first instruction; matches planned_gpu_options().
STARTUP_ALLOWANCE_S = 180
# Time after a lease deadline for a worker's own shutdown before the controller cancels it.
TERMINATION_GRACE_S = 60


class Backend(Protocol):
    """The contract a remote adapter must satisfy.

    `spawn` submits one block and returns a durable call ID without waiting; it never
    retries. `poll` waits up to `timeout_s` and returns None while the call may still
    run. Afterwards it returns {"state": "returned", "result": {...}} or
    {"state": "failed", "error": "..."}. It must work from a new controller process
    given only the call ID. `cancel` requests termination of every container for the call.
    """

    def spawn(self, payload: dict) -> str: ...
    def poll(self, call_id: str, timeout_s: float) -> dict | None: ...
    def cancel(self, call_id: str) -> None: ...


def plan_blocks(config: Config, *, phase: str, eligible: list[dict] | None = None) -> list[dict]:
    """Fresh-process blocks in frozen schedule order: one per repetition, engine, and attempt."""
    scheduled = [cell.to_dict() for cell in config.cells(phase=phase)]
    if eligible is None:
        cells = [{**cell, "attempt": 1, "replacement_for": None} for cell in scheduled]
    else:
        if phase != "measurement":
            raise ValueError("Only formal cells can be replanned after an interruption")
        cells = []
        for item in eligible:
            cell = {key: item[key] for key in CELL_FIELDS}
            attempt, original = item["attempt"], item.get("replacement_for")
            if cell not in scheduled:
                raise ValueError("A replanned cell is not in the frozen schedule")
            if attempt not in (1, 2) or (attempt == 2) != (isinstance(original, str) and bool(original)):
                raise ValueError("A replacement attempt needs exactly one original artifact")
            cells.append({**cell, "attempt": attempt, "replacement_for": original})
        if len({tuple(c[key] for key in CELL_FIELDS) for c in cells}) != len(cells):
            raise ValueError("A cell can appear in only one planned block")
    blocks = {}
    for cell in cells:
        block = blocks.setdefault((cell["repetition"], cell["engine"], cell["attempt"]), {
            "phase": phase, "engine": cell["engine"], "repetition": cell["repetition"],
            "attempt": cell["attempt"],
            "stage": "pilot" if phase == "pilot" else "formal" if cell["attempt"] == 1 else "rerun",
            "cells": [], "replacement_for": {},
        })
        block["cells"].append({key: cell[key] for key in CELL_FIELDS})
        if cell["replacement_for"]:
            block["replacement_for"][Cell(**block["cells"][-1]).key] = cell["replacement_for"]
    return list(blocks.values())


def block_payload(block: dict, *, campaign: dict, config: Config, reservation: dict, lease_s: int) -> dict:
    if campaign_phase(campaign) != block["phase"] or campaign["config_sha256"] != config.fingerprint:
        raise ValueError("Block phase or configuration differs from the campaign")
    return {
        "schema_version": 1, "phase": block["phase"], "engine": block["engine"],
        "config": config.data, "cells": block["cells"],
        "context": {
            **{key: campaign[key] for key in IDENTITY_FIELDS},
            "attempt": block["attempt"], "reservation_id": reservation["id"],
        },
        "cell_context": {
            key: {"replacement_for": original} for key, original in block["replacement_for"].items()
        },
        "deadline_unix": reservation["deadline_unix"], "minimum_start_s": lease_s,
        "directory": f'runs/{block["engine"]}/r{block["repetition"]}/a{block["attempt"]}-{reservation["id"]}',
    }


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def charge_seconds(reservation: dict, terminal: dict, end_unix: float) -> int:
    """Seconds from the earliest plausible container start to termination, rounded up.

    Queue time is excluded only when the worker reports its start. A platform restart can
    still make this low, so reconcile billing after every stage.
    """
    start = reservation["created_unix"]
    result = terminal.get("result") if terminal["state"] == "returned" else None
    started = result.get("started_unix") if isinstance(result, dict) else None
    if _finite(started) and start <= started <= end_unix:
        start = max(start, started - STARTUP_ALLOWANCE_S)
    return max(1, math.ceil(end_unix - start))


def _poll(backend: Backend, call_id: str, timeout_s: float, errors: list[str]) -> dict | None:
    try:
        record = backend.poll(call_id, timeout_s)
    except Exception as exc:
        errors.append(f"poll: {type(exc).__name__}: {exc}"[:300])
        return None
    if record is None:
        return None
    if (
        isinstance(record, dict) and record.get("state") in ("returned", "failed")
        and (record["state"] == "failed" or isinstance(record.get("result"), dict))
    ):
        return record
    errors.append("poll: invalid terminal record")
    return None


def _cancel(backend: Backend, call_id: str, errors: list[str]) -> dict | None:
    try:
        backend.cancel(call_id)
    except Exception as exc:
        errors.append(f"cancel: {type(exc).__name__}: {exc}"[:300])
    return _poll(backend, call_id, TERMINATION_GRACE_S, errors)


def execute(
    blocks: list[dict], *, campaign: dict, config: Config, ledger: Ledger, backend: Backend,
    lease_s: int, queue_s: int = 600, poll_s: float = 15,
    clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Run blocks one at a time; stop at the first uncertainty, failure, or budget limit."""
    if type(lease_s) is not int or type(queue_s) is not int or lease_s <= 60 or queue_s < 0:
        raise ValueError("Lease and queue allowances must be whole seconds")
    if campaign["harness_sha256"] != source_tree_hash():
        raise ValueError("Local harness source differs from the frozen campaign")
    completed = []

    def stop(reason: str, **details) -> dict:
        return {"completed_blocks": completed, "stopped": {"reason": reason, **details}}

    for index, block in enumerate(blocks):
        summary = ledger.summary()
        if summary["unresolved"] or summary["unreconciled_overruns"]:
            return stop("Resolve unresolved or overrun reservations before dispatch", block=index)
        try:
            reservation = ledger.reserve(block["stage"], lease_s + queue_s, now=clock())
        except BudgetError as exc:
            return stop(f"Budget limit: {exc}", block=index)
        payload = block_payload(block, campaign=campaign, config=config, reservation=reservation, lease_s=lease_s)
        try:
            call_id = backend.spawn(payload)
        except Exception as exc:
            return stop(
                "Submission failed; confirm that no call started, then settle the reservation explicitly",
                block=index, reservation_id=reservation["id"], error=f"{type(exc).__name__}: {exc}"[:300],
            )
        ledger.attach_call(reservation["id"], call_id)
        errors, cancelled, terminal = [], False, None
        until = reservation["deadline_unix"] + TERMINATION_GRACE_S
        while terminal is None and (remaining := until - clock()) > 0:
            before = clock()
            terminal = _poll(backend, call_id, min(poll_s, remaining), errors)
            if terminal is None and clock() - before < min(poll_s, remaining) / 2:
                sleep(min(poll_s, remaining))
        if terminal is None:
            cancelled = True
            terminal = _cancel(backend, call_id, errors)
        if terminal is None:
            return stop(
                "The call did not confirm termination; its reservation remains unresolved",
                block=index, reservation_id=reservation["id"], call_id=call_id, errors=errors,
            )
        charged = charge_seconds(reservation, terminal, clock())
        settled = ledger.settle(reservation["id"], charged, confirmed_stopped=True)
        result = terminal.get("result") or {}
        completed.append({
            "block": index, "engine": block["engine"], "repetition": block["repetition"],
            "attempt": block["attempt"], "stage": block["stage"], "call_id": call_id,
            "reservation_id": reservation["id"], "directory": payload["directory"],
            "charged_seconds": charged, "overrun": settled["status"] == "overrun",
            "cancelled": cancelled, "state": terminal["state"], "status": result.get("status"),
            "error": terminal.get("error") or result.get("error"), "poll_errors": errors,
        })
        if cancelled or completed[-1]["overrun"] or result.get("status") != "complete":
            return stop("The block did not complete; inspect it and replan with resume-plan", block=index)
    return {"completed_blocks": completed, "stopped": None}


def resolve_outstanding(
    ledger: Ledger, backend: Backend, *, clock: Callable[[], float] = time.time,
) -> list[dict]:
    """Settle an earlier controller's calls only after the backend confirms they ended."""
    report = []
    for record in [r for r in ledger.read()["reservations"] if r["status"] == "reserved"]:
        entry = {"reservation_id": record["id"], "call_id": record["remote_call_id"]}
        report.append(entry)
        if record["remote_call_id"] is None:
            entry["action"] = "manual: confirm that no call started, then settle explicitly"
            continue
        errors = []
        terminal, cancelled = _poll(backend, record["remote_call_id"], 0, errors), False
        if terminal is None:
            if clock() < record["deadline_unix"] + TERMINATION_GRACE_S:
                entry.update(action="running within its lease; check again later", errors=errors)
                continue
            cancelled = True
            terminal = _cancel(backend, record["remote_call_id"], errors)
            if terminal is None:
                entry.update(action="termination unconfirmed; reservation kept", errors=errors)
                continue
        now = clock()
        result = terminal.get("result") if terminal["state"] == "returned" else None
        finished = result.get("finished_unix") if isinstance(result, dict) else None
        if cancelled:
            end = now
        elif _finite(finished) and record["created_unix"] <= finished <= now:
            end = finished
        else:
            # The end time is unknown. A worker cannot normally run past its lease.
            end = min(now, record["deadline_unix"])
        charged = charge_seconds(record, terminal, end)
        settled = ledger.settle(record["id"], charged, confirmed_stopped=True)
        entry.update(
            action="settled", state=terminal["state"], charged_seconds=charged,
            overrun=settled["status"] == "overrun", errors=errors,
        )
    return report
