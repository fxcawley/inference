"""Conservative single-controller reservations; never a distributed lock."""

from __future__ import annotations

import math
import os
import time
import uuid
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .artifacts import write_json
from .config import Config
from .provenance import digest_json, is_hash, read_json

STAGES = ("pilot", "formal", "rerun")
LEDGER_KEYS = {"schema_version", "ledger_id", "policy_sha256", "reservations", "billing"}
RESERVATION_KEYS = {
    "id", "stage", "status", "reserved_seconds", "reserved_usd",
    "created_unix", "deadline_unix", "remote_call_id",
}
BILLING_KEYS = {
    "id", "through_unix", "recorded_unix", "gpu_container_usd", "other_usd",
    "source_sha256", "note",
}


class BudgetError(ValueError):
    pass


def _usd(value) -> Decimal:
    try:
        amount = Decimal(value) if isinstance(value, str) else None
    except InvalidOperation:
        amount = None
    if amount is None or not amount.is_finite() or amount < 0:
        raise BudgetError("Dollar amounts must be finite, nonnegative decimal strings")
    return amount


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


class Ledger:
    """One persistent ledger per budget policy, shared by pilot and formal campaigns."""

    def __init__(self, path: Path, config: Config):
        self.path = Path(path)
        self.config = config
        self.ledger_id = config.data["budget"]["ledger_id"]
        self.policy = digest_json({
            "budget": config.data["budget"],
            "pricing": config.data["pricing"],
            "hardware": config.hardware,
        })

    @contextmanager
    def _exclusive(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise BudgetError("Ledger is locked; resolve the other controller before continuing") from exc
        try:
            os.close(fd)
            yield
        finally:
            lock.unlink()

    def read(self) -> dict:
        if not self.path.exists():
            return {
                "schema_version": 2, "ledger_id": self.ledger_id,
                "policy_sha256": self.policy, "reservations": [], "billing": [],
            }
        data = read_json(self.path)
        if (
            not isinstance(data, dict) or set(data) != LEDGER_KEYS
            or type(data["schema_version"]) is not int or data["schema_version"] != 2
            or not isinstance(data["reservations"], list) or not isinstance(data["billing"], list)
        ):
            raise BudgetError("Unsupported or invalid ledger; preserve it and reconcile before continuing")
        if data["ledger_id"] != self.ledger_id or data["policy_sha256"] != self.policy:
            raise BudgetError("Existing ledger/policy differs; do not reset the budget")
        seen = set()
        for record in data["reservations"]:
            try:
                status = record["status"]
                expected = RESERVATION_KEYS if status == "reserved" else RESERVATION_KEYS | {"charged_seconds", "charged_usd"}
                valid = (
                    set(record) == expected
                    and isinstance(record["id"], str) and bool(record["id"])
                    and record["id"] not in seen
                    and record["stage"] in STAGES
                    and status in ("reserved", "settled", "overrun")
                    and type(record["reserved_seconds"]) is int and record["reserved_seconds"] > 0
                    and _usd(record["reserved_usd"]) == self.config.rate_per_second * record["reserved_seconds"]
                    and _finite(record["created_unix"]) and _finite(record["deadline_unix"])
                    and record["deadline_unix"] > record["created_unix"]
                    and record["deadline_unix"] == record["created_unix"] + record["reserved_seconds"]
                    and (record["remote_call_id"] is None or (
                        isinstance(record["remote_call_id"], str) and bool(record["remote_call_id"].strip())
                    ))
                )
                if status != "reserved":
                    valid = valid and (
                        type(record["charged_seconds"]) is int and record["charged_seconds"] >= 0
                        and _usd(record["charged_usd"]) == self.config.rate_per_second * record["charged_seconds"]
                        and (status == "overrun") == (record["charged_seconds"] > record["reserved_seconds"])
                    )
            except (KeyError, TypeError, OverflowError, BudgetError) as exc:
                raise BudgetError("Invalid ledger reservation; preserve it and reconcile before continuing") from exc
            if not valid:
                raise BudgetError("Invalid ledger reservation; preserve it and reconcile before continuing")
            seen.add(record["id"])
        previous = None
        for record in data["billing"]:
            try:
                valid = (
                    isinstance(record, dict) and set(record) == BILLING_KEYS
                    and isinstance(record["id"], str) and bool(record["id"]) and record["id"] not in seen
                    and _finite(record["through_unix"]) and _finite(record["recorded_unix"])
                    and record["through_unix"] <= record["recorded_unix"]
                    and is_hash(record["source_sha256"])
                    and isinstance(record["note"], str) and len(record["note"]) <= 500
                )
                gpu, other = _usd(record["gpu_container_usd"]), _usd(record["other_usd"])
                if valid and previous is not None:
                    valid = (
                        record["through_unix"] > previous["through_unix"]
                        and gpu >= _usd(previous["gpu_container_usd"])
                        and other >= _usd(previous["other_usd"])
                    )
            except (KeyError, TypeError, OverflowError, BudgetError) as exc:
                raise BudgetError("Invalid billing record; preserve it and reconcile before continuing") from exc
            if not valid:
                raise BudgetError("Invalid billing record; preserve it and reconcile before continuing")
            seen.add(record["id"])
            previous = record
        return data

    @staticmethod
    def _seconds(record: dict) -> int:
        return (
            record["reserved_seconds"] if record["status"] == "reserved"
            else record["charged_seconds"]
        )

    @staticmethod
    def _dollars(record: dict) -> Decimal:
        return Decimal(
            record["reserved_usd"] if record["status"] == "reserved"
            else record["charged_usd"]
        )

    @staticmethod
    def _end(record: dict) -> float:
        """The latest time a lease could have run, including any charged overrun."""
        if record["status"] == "reserved":
            return record["deadline_unix"]
        return max(record["deadline_unix"], record["created_unix"] + record["charged_seconds"])

    def _unreconciled_overruns(self, data: dict) -> list[str]:
        # Billing through an overrun's full charged interval captures its actual usage.
        through = data["billing"][-1]["through_unix"] if data["billing"] else None
        return [
            r["id"] for r in data["reservations"]
            if r["status"] == "overrun" and (through is None or through < self._end(r))
        ]

    def _exposure(self, data: dict) -> dict:
        """Billing can only increase exposure; a lower bill never releases allowance."""
        records = data["reservations"]
        latest = data["billing"][-1] if data["billing"] else None
        through = latest["through_unix"] if latest else None
        billed_gpu = Decimal(latest["gpu_container_usd"]) if latest else Decimal(0)
        billed_other = Decimal(latest["other_usd"]) if latest else Decimal(0)
        covered = sum(
            (self._dollars(r) for r in records if latest and self._end(r) <= through),
            Decimal(0),
        )
        ledger = sum((self._dollars(r) for r in records), Decimal(0))
        adjustment = max(Decimal(0), billed_gpu - covered)
        return {
            "ledger_gpu_usd": ledger, "covered_gpu_usd": covered,
            "billed_gpu_usd": billed_gpu, "billed_other_usd": billed_other,
            "billed_through_unix": through, "gpu_adjustment_usd": adjustment,
            "gpu_exposure_usd": ledger + adjustment,
            # Settled charges use the frozen list price. Scale new reservations if
            # reconciled bills show that price was too low.
            "rate_multiplier": billed_gpu / covered if covered and billed_gpu > covered else Decimal(1),
            "other_exposure_usd": max(Decimal(str(self.config.data["budget"]["other_usd"])), billed_other),
        }

    def _remaining(self, data: dict, exposure: dict) -> dict:
        limits = self.config.data["budget"]
        records = data["reservations"]
        dollars = min(
            Decimal(str(limits["gpu_container_usd"])),
            Decimal(str(limits["total_usd"])) - exposure["other_exposure_usd"]
            - Decimal(str(limits["reserve_usd"])),
        ) - exposure["gpu_exposure_usd"]
        return {
            "gpu_container_usd": str(dollars),
            "gpu_seconds": limits["max_gpu_seconds"] - sum(self._seconds(r) for r in records),
            **{
                f"{stage}_seconds": limits[f"{stage}_seconds"] - sum(
                    self._seconds(r) for r in records if r["stage"] == stage
                )
                for stage in STAGES
            },
        }

    def reserve(self, stage: str, seconds: int, *, now: float | None = None) -> dict:
        if stage not in STAGES:
            raise BudgetError("Unknown budget stage")
        if type(seconds) is not int or seconds <= 0:
            raise BudgetError("Reservation duration must be a positive integer")
        if seconds > self.config.experiment["block_timeout_s"]:
            raise BudgetError("Reservation exceeds the maximum block duration")
        now = time.time() if now is None else now
        if not _finite(now) or now + seconds <= now:
            raise BudgetError("Reservation start/deadline must be finite and increasing")
        cost = self.config.rate_per_second * seconds
        with self._exclusive():
            data = self.read()
            records = data["reservations"]
            if any(r["status"] == "reserved" for r in records):
                raise BudgetError("An unresolved call retains its reservation; GPU work is serial")
            if self._unreconciled_overruns(data):
                raise BudgetError("A reservation overran; record billing through its end before further work")
            exposure = self._exposure(data)
            if exposure["billed_through_unix"] is not None and now < exposure["billed_through_unix"]:
                raise BudgetError("A new lease cannot start inside an already reconciled billing period")
            remaining = self._remaining(data, exposure)
            if seconds > remaining["gpu_seconds"]:
                raise BudgetError("Campaign GPU time limit exceeded")
            if seconds > remaining[f"{stage}_seconds"]:
                raise BudgetError(f"{stage} time limit exceeded")
            projected = cost * exposure["rate_multiplier"]
            limits = self.config.data["budget"]
            if exposure["gpu_exposure_usd"] + projected > Decimal(str(limits["gpu_container_usd"])):
                raise BudgetError("GPU-container dollar allocation exceeded")
            if projected > Decimal(remaining["gpu_container_usd"]):
                raise BudgetError("Total project dollar allocation exceeded")
            record = {
                "id": uuid.uuid4().hex, "stage": stage, "status": "reserved",
                "reserved_seconds": seconds, "reserved_usd": str(cost),
                "created_unix": now, "deadline_unix": now + seconds,
                "remote_call_id": None,
            }
            records.append(record)
            write_json(self.path, data)
            return dict(record)

    def attach_call(self, reservation_id: str, call_id: str) -> None:
        if not isinstance(call_id, str) or not call_id.strip():
            raise BudgetError("Call ID cannot be empty")
        with self._exclusive():
            data = self.read()
            record = self._find(data, reservation_id)
            if record["status"] != "reserved" or record["remote_call_id"] not in (None, call_id):
                raise BudgetError("Cannot replace an existing call/reservation identity")
            record["remote_call_id"] = call_id
            write_json(self.path, data)

    @staticmethod
    def _find(data: dict, reservation_id: str) -> dict:
        for record in data["reservations"]:
            if record["id"] == reservation_id:
                return record
        raise BudgetError("Unknown reservation")

    def settle(
        self, reservation_id: str, charged_seconds: int, *, confirmed_stopped: bool,
    ) -> dict:
        if confirmed_stopped is not True:
            raise BudgetError("A deadline or missing response is not confirmation that a call stopped")
        if type(charged_seconds) is not int or charged_seconds < 0:
            raise BudgetError("Charge duration must be a nonnegative integer, rounded up")
        with self._exclusive():
            data = self.read()
            record = self._find(data, reservation_id)
            if record["status"] != "reserved":
                raise BudgetError("Reservation has already been settled")
            record["charged_seconds"] = charged_seconds
            record["charged_usd"] = str(charged_seconds * self.config.rate_per_second)
            record["status"] = (
                "overrun" if charged_seconds > record["reserved_seconds"] else "settled"
            )
            write_json(self.path, data)
            return dict(record)

    def record_billing(
        self, through_unix: float, gpu_container_usd: str, other_usd: str, source_sha256: str,
        *, note: str = "", now: float | None = None,
    ) -> dict:
        """Append cumulative gross project charges from a saved billing export."""
        now = time.time() if now is None else now
        if not _finite(through_unix) or not _finite(now) or through_unix > now:
            raise BudgetError("Billing must cover a finite period that has already ended")
        gpu, other = _usd(gpu_container_usd), _usd(other_usd)
        if not is_hash(source_sha256):
            raise BudgetError("Record the SHA-256 of the saved billing export")
        if not isinstance(note, str) or len(note) > 500:
            raise BudgetError("Billing notes are limited to 500 characters")
        with self._exclusive():
            data = self.read()
            for record in data["reservations"]:
                if record["created_unix"] < through_unix < self._end(record):
                    raise BudgetError("Billing period ends inside a lease; choose a time when no call could run")
                if self._end(record) <= through_unix and record["status"] == "reserved":
                    raise BudgetError("Settle every call in the billed period before reconciling")
            if data["billing"]:
                previous = data["billing"][-1]
                if (
                    through_unix <= previous["through_unix"]
                    or gpu < Decimal(previous["gpu_container_usd"])
                    or other < Decimal(previous["other_usd"])
                ):
                    raise BudgetError("Billing records are cumulative; time and charges cannot decrease")
            record = {
                "id": uuid.uuid4().hex, "through_unix": through_unix, "recorded_unix": now,
                "gpu_container_usd": str(gpu), "other_usd": str(other),
                "source_sha256": source_sha256, "note": note,
            }
            data["billing"].append(record)
            write_json(self.path, data)
            return dict(record)

    def summary(self) -> dict:
        data = self.read()
        records = data["reservations"]
        exposure = self._exposure(data)
        warnings = []
        if exposure["billed_gpu_usd"] and not exposure["covered_gpu_usd"]:
            warnings.append("GPU billing has no matching settled reservations; check attribution")
        elif exposure["gpu_adjustment_usd"]:
            warnings.append("Billed GPU-container cost exceeds settled ledger charges")
        if exposure["billed_other_usd"] > Decimal(str(self.config.data["budget"]["other_usd"])):
            warnings.append("Non-GPU charges exceed their allocation and now consume the reserve")
        return {
            "ledger_id": self.ledger_id,
            "exposure_usd": str(exposure["gpu_exposure_usd"]),
            "exposure_seconds": sum(self._seconds(r) for r in records),
            "unresolved": [r["id"] for r in records if r["status"] == "reserved"],
            "overruns": [r["id"] for r in records if r["status"] == "overrun"],
            "unreconciled_overruns": self._unreconciled_overruns(data),
            "reservation_ids": [r["id"] for r in records],
            "stages": {
                stage: {
                    "reservations": sum(r["stage"] == stage for r in records),
                    "exposure_seconds": sum(self._seconds(r) for r in records if r["stage"] == stage),
                }
                for stage in STAGES
            },
            "billing": {
                "records": len(data["billing"]),
                "billed_through_unix": exposure["billed_through_unix"],
                "gpu_container_usd": str(exposure["billed_gpu_usd"]),
                "other_usd": str(exposure["billed_other_usd"]),
                "gpu_adjustment_usd": str(exposure["gpu_adjustment_usd"]),
                "rate_multiplier": str(exposure["rate_multiplier"]),
                "other_exposure_usd": str(exposure["other_exposure_usd"]),
            },
            "remaining": self._remaining(data, exposure),
            "warnings": warnings,
        }
