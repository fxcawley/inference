"""Conservative single-controller reservations; never a distributed lock."""

from __future__ import annotations

import os
import math
import time
import uuid
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .artifacts import write_json
from .config import Config
from .provenance import digest_json, read_json


class BudgetError(ValueError):
    pass


class Ledger:
    def __init__(self, path: Path, campaign_id: str, config: Config):
        if not campaign_id or not campaign_id.strip():
            raise BudgetError("Campaign ID is required")
        self.path = Path(path)
        self.campaign_id = campaign_id
        self.config = config
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
                "schema_version": 1, "campaign_id": self.campaign_id,
                "policy_sha256": self.policy, "reservations": [],
            }
        data = read_json(self.path)
        if (
            not isinstance(data, dict)
            or type(data.get("schema_version")) is not int or data["schema_version"] != 1
            or data.get("campaign_id") != self.campaign_id
            or data.get("policy_sha256") != self.policy
        ):
            raise BudgetError("Existing campaign/policy differs; do not reset the budget")
        if not isinstance(data.get("reservations"), list):
            raise BudgetError("Invalid ledger")
        seen = set()
        for record in data["reservations"]:
            try:
                base = {
                    "id", "stage", "status", "reserved_seconds", "reserved_usd",
                    "created_unix", "deadline_unix", "remote_call_id",
                }
                status = record["status"]
                expected = base if status == "reserved" else base | {"charged_seconds", "charged_usd"}
                valid = (
                    set(record) == expected
                    and isinstance(record["id"], str) and bool(record["id"])
                    and record["id"] not in seen
                    and record["stage"] in ("pilot", "formal", "rerun")
                    and status in ("reserved", "settled", "overrun")
                    and type(record["reserved_seconds"]) is int and record["reserved_seconds"] > 0
                    and isinstance(record["reserved_usd"], str)
                    and Decimal(record["reserved_usd"]) == self.config.rate_per_second * record["reserved_seconds"]
                    and all(type(record[k]) in (int, float) and math.isfinite(record[k])
                            for k in ("created_unix", "deadline_unix"))
                    and record["deadline_unix"] > record["created_unix"]
                    and record["deadline_unix"] == record["created_unix"] + record["reserved_seconds"]
                    and (record["remote_call_id"] is None or (
                        isinstance(record["remote_call_id"], str) and bool(record["remote_call_id"].strip())
                    ))
                )
                if status != "reserved":
                    valid = valid and (
                        type(record["charged_seconds"]) is int and record["charged_seconds"] >= 0
                        and isinstance(record["charged_usd"], str)
                        and Decimal(record["charged_usd"]) == self.config.rate_per_second * record["charged_seconds"]
                        and (status == "overrun") == (record["charged_seconds"] > record["reserved_seconds"])
                    )
                if not valid:
                    raise BudgetError("Invalid ledger reservation; preserve it and reconcile before continuing")
                seen.add(record["id"])
            except (KeyError, TypeError, InvalidOperation, OverflowError) as exc:
                raise BudgetError("Invalid ledger reservation; preserve it and reconcile before continuing") from exc
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

    def reserve(self, stage: str, seconds: int, *, now: float | None = None) -> dict:
        if stage not in ("pilot", "formal", "rerun"):
            raise BudgetError("Unknown budget stage")
        if type(seconds) is not int or seconds <= 0:
            raise BudgetError("Reservation duration must be a positive integer")
        if seconds > self.config.experiment["block_timeout_s"]:
            raise BudgetError("Reservation exceeds the maximum block duration")
        now = time.time() if now is None else now
        if type(now) not in (int, float) or not math.isfinite(now) or now + seconds <= now:
            raise BudgetError("Reservation start/deadline must be finite and increasing")
        cost = self.config.rate_per_second * seconds
        limits = self.config.data["budget"]
        with self._exclusive():
            data = self.read()
            records = data["reservations"]
            if any(r["status"] == "reserved" for r in records):
                raise BudgetError("An unresolved call retains its reservation; GPU work is serial")
            if any(r["status"] == "overrun" for r in records):
                raise BudgetError("A previous reservation overran; reconcile before further work")
            if seconds + sum(self._seconds(r) for r in records) > limits["max_gpu_seconds"]:
                raise BudgetError("Campaign GPU time limit exceeded")
            stage_seconds = sum(self._seconds(r) for r in records if r["stage"] == stage)
            if stage_seconds + seconds > limits[f"{stage}_seconds"]:
                raise BudgetError(f"{stage} time limit exceeded")
            exposure = cost + sum((self._dollars(r) for r in records), Decimal(0))
            if exposure > Decimal(str(limits["gpu_container_usd"])):
                raise BudgetError("GPU-container dollar allocation exceeded")
            if exposure + Decimal(str(limits["other_usd"])) + Decimal(str(limits["reserve_usd"])) > Decimal(str(limits["total_usd"])):
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

    def summary(self) -> dict:
        records = self.read()["reservations"]
        return {
            "campaign_id": self.campaign_id,
            "exposure_usd": str(sum((self._dollars(r) for r in records), Decimal(0))),
            "exposure_seconds": sum(self._seconds(r) for r in records),
            "unresolved": [r["id"] for r in records if r["status"] == "reserved"],
            "overruns": [r["id"] for r in records if r["status"] == "overrun"],
            "reservation_ids": [r["id"] for r in records],
            "stages": {
                stage: {
                    "reservations": sum(r["stage"] == stage for r in records),
                    "exposure_seconds": sum(self._seconds(r) for r in records if r["stage"] == stage),
                }
                for stage in ("pilot", "formal", "rerun")
            },
        }
