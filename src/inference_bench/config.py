"""Strict offline configuration, reproducible schedules, and priced bounds."""

from __future__ import annotations

import math
import random
import re
import tomllib
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from .provenance import digest_json

SECTIONS = {
    "experiment": {
        "model", "seed", "concurrency", "repetitions", "requests_per_cell",
        "max_model_len", "max_running_requests", "warmup_min", "warmup_waves",
        "request_timeout_s", "ready_timeout_s", "block_timeout_s",
    },
    "hardware": {
        "gpu", "cpu_cores", "memory_mib", "dtype", "kv_cache_dtype", "memory_fraction",
    },
    "budget": {
        "total_usd", "gpu_container_usd", "other_usd", "reserve_usd", "max_gpu_seconds",
        "pilot_seconds", "formal_seconds", "rerun_seconds",
    },
    "pricing": {
        "checked_on", "gpu_per_second", "cpu_per_core_second", "memory_per_gib_second",
    },
    "workload": {"dataset", "dataset_config", "split", "instruction", "enable_thinking"},
}


def _keys(value: Any, expected: set[str], where: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(f"{where}: missing={expected - actual}, unknown={actual - expected}")


def _number(value: Any, name: str, *, integer: bool = False, minimum: float = 0) -> None:
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'} >= {minimum}")


@dataclass(frozen=True)
class Cell:
    engine: str
    profile: str
    concurrency: int
    repetition: int

    @property
    def key(self) -> str:
        return f"{self.engine}/r{self.repetition}/{self.profile}-c{self.concurrency}"

    def to_dict(self) -> dict:
        return {
            "engine": self.engine, "profile": self.profile,
            "concurrency": self.concurrency, "repetition": self.repetition,
        }


@dataclass(frozen=True)
class Config:
    data: dict

    def __post_init__(self) -> None:
        _keys(self.data, {"schema_version", "profiles", *SECTIONS}, "configuration")
        if type(self.data["schema_version"]) is not int or self.data["schema_version"] != 1:
            raise ValueError("Unsupported configuration schema")
        for section, keys in SECTIONS.items():
            _keys(self.data[section], keys, section)
        e, h, b, p, w = (self.data[k] for k in SECTIONS)
        for key in SECTIONS["experiment"] - {"model", "concurrency"}:
            _number(e[key], key, integer=True, minimum=0 if key == "seed" else 1)
        if not isinstance(e["model"], str) or not e["model"]:
            raise ValueError("A model repository is required")
        concurrency = e["concurrency"]
        if not isinstance(concurrency, list) or not concurrency:
            raise ValueError("Concurrency must be a nonempty list")
        for c in concurrency:
            _number(c, "concurrency", integer=True, minimum=1)
        if sorted(set(concurrency)) != concurrency or max(concurrency) > e["max_running_requests"]:
            raise ValueError("Concurrency must be sorted, unique, and within the server limit")
        if e["requests_per_cell"] < max(concurrency):
            raise ValueError("Each cell needs at least one full wave of requests")
        if not isinstance(self.profiles, dict) or not self.profiles:
            raise ValueError("At least one workload profile is required")
        for name, shape in self.profiles.items():
            if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
                raise ValueError(f"Invalid profile name: {name}")
            _keys(shape, {"input_tokens", "output_tokens"}, f"profile {name}")
            for key, value in shape.items():
                _number(value, key, integer=True, minimum=1)
            if sum(shape.values()) > e["max_model_len"]:
                raise ValueError(f"{name} exceeds the model context")
        if h["gpu"] != "H200" or h["dtype"] != "bfloat16" or h["kv_cache_dtype"] != "auto":
            raise ValueError("This baseline requires H200, BF16, and a qualified auto KV dtype")
        _number(h["cpu_cores"], "cpu_cores", minimum=1)
        _number(h["memory_mib"], "memory_mib", integer=True, minimum=1024)
        _number(h["memory_fraction"], "memory_fraction", minimum=0.01)
        if h["memory_fraction"] >= 1:
            raise ValueError("Memory fraction must be less than one")
        for key, value in b.items():
            _number(value, key, integer=key.endswith("seconds"), minimum=0)
        if sum(Decimal(str(b[k])) for k in ("gpu_container_usd", "other_usd", "reserve_usd")) > Decimal(str(b["total_usd"])):
            raise ValueError("Stage allocations exceed total budget")
        if sum(b[f"{stage}_seconds"] for stage in ("pilot", "formal", "rerun")) > b["max_gpu_seconds"]:
            raise ValueError("Stage time limits exceed the project time limit")
        date.fromisoformat(p["checked_on"])
        for key in SECTIONS["pricing"] - {"checked_on"}:
            _number(p[key], key, minimum=0.000000001)
        if self.rate_per_second * b["max_gpu_seconds"] > Decimal(str(b["gpu_container_usd"])):
            raise ValueError("GPU-container time allowance exceeds its dollar allocation")
        if w["enable_thinking"] is not False:
            raise ValueError("Thinking is disabled for the baseline")
        for key in SECTIONS["workload"] - {"enable_thinking"}:
            if not isinstance(w[key], str) or not w[key].strip():
                raise ValueError(f"{key} must be a nonempty string")

    @property
    def experiment(self) -> dict:
        return self.data["experiment"]

    @property
    def hardware(self) -> dict:
        return self.data["hardware"]

    @property
    def profiles(self) -> dict:
        return self.data["profiles"]

    @property
    def profile_names(self) -> list[str]:
        order = {"short": 0, "prefill": 1, "decode": 2}
        return sorted(self.profiles, key=lambda name: (order.get(name, 3), name))

    @property
    def fingerprint(self) -> str:
        return digest_json(self.data)

    @property
    def rate_per_second(self) -> Decimal:
        p, h = self.data["pricing"], self.hardware
        return (
            Decimal(str(p["gpu_per_second"]))
            + Decimal(str(p["cpu_per_core_second"])) * Decimal(str(h["cpu_cores"]))
            + Decimal(str(p["memory_per_gib_second"])) * Decimal(h["memory_mib"]) / 1024
        )

    def cells(self) -> list[Cell]:
        result = []
        for repetition in range(1, self.experiment["repetitions"] + 1):
            pairs = [(p, c) for p in self.profile_names for c in self.experiment["concurrency"]]
            random.Random(self.experiment["seed"] + repetition).shuffle(pairs)
            engines = ("vllm", "sglang") if repetition % 2 else ("sglang", "vllm")
            for engine in engines:
                result.extend(Cell(engine, p, c, repetition) for p, c in pairs)
        return result

    def describe(self) -> dict:
        cells = self.cells()
        n = self.experiment["requests_per_cell"]
        return {
            "config_sha256": self.fingerprint,
            "cells": len(cells), "measured_requests": n * len(cells),
            "input_tokens": n * sum(self.profiles[c.profile]["input_tokens"] for c in cells),
            "output_tokens": n * sum(self.profiles[c.profile]["output_tokens"] for c in cells),
            "container_usd_per_hour": str(self.rate_per_second * 3600),
            "budget": self.data["budget"],
            "runtime_forecast": None,
            "schedule": [c.to_dict() for c in cells],
        }


def load_config(path: Path | str) -> Config:
    with Path(path).open("rb") as source:
        return Config(tomllib.load(source))
