"""Pure metric definitions. No engine summaries or estimated token counts."""

from __future__ import annotations

import math
from statistics import median


def quantile(values: list[float], q: float) -> float | None:
    if not 0 <= q <= 1:
        raise ValueError("Quantile must lie between zero and one")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Quantiles require finite numeric observations")
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def request_metrics(row: dict) -> dict:
    if row["status"] != "success":
        raise ValueError("Request metrics require a successful request")
    times = [row[k] for k in (
        "dispatch_ns", "headers_ns", "first_content_ns", "last_content_ns", "done_ns",
    )]
    if any(type(t) is not int or t < 0 for t in times) or times != sorted(times):
        raise ValueError("Missing or non-monotonic successful-request timestamps")
    for key in ("actual_input_tokens", "actual_output_tokens"):
        if type(row[key]) is not int or row[key] < 1:
            raise ValueError("Successful requests require authoritative token counts")
    if (
        row["actual_input_tokens"] != row["expected_input_tokens"]
        or row["actual_output_tokens"] != row["requested_output_tokens"]
        or row["http_status"] != 200
        or type(row["content_event_count"]) is not int
        or row["content_event_count"] < 1
    ):
        raise ValueError("Successful request violates the measurement contract")
    tpot = None
    reason = None
    if row["actual_output_tokens"] <= 1:
        reason = "one_or_fewer_output_tokens"
    elif row["content_event_count"] <= 1:
        reason = "one_content_event"
    else:
        tpot = (row["last_content_ns"] - row["first_content_ns"]) / (
            1_000_000 * (row["actual_output_tokens"] - 1)
        )
    return {
        "ttft_ms": (row["first_content_ns"] - row["dispatch_ns"]) / 1_000_000,
        "e2e_ms": (row["done_ns"] - row["dispatch_ns"]) / 1_000_000,
        "tpot_ms_per_token": tpot, "tpot_unobserved_reason": reason,
    }


def summarize(rows: list[dict], *, expected_requests: int) -> dict:
    if type(expected_requests) is not int or expected_requests < 1:
        raise ValueError("Expected request count must be positive")
    if len(rows) > expected_requests:
        raise ValueError("More attempted requests than expected")
    if len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate request records")
    if len({r["phase"] for r in rows}) > 1:
        raise ValueError("Warmup/pilot/measurement phases cannot be pooled")
    if any(r["status"] not in ("success", "failed", "cancelled") for r in rows):
        raise ValueError("A request has no terminal status")
    success = [r for r in rows if r["status"] == "success"]
    observations = [request_metrics(r) for r in success]
    starts, ends = [], []
    for row in rows:
        start = row["dispatch_ns"]
        end = row["done_ns"] if row["status"] == "success" else row["error_ns"]
        if type(start) is not int or type(end) is not int or end < start:
            raise ValueError("Attempted requests need valid dispatch and terminal timestamps")
        starts.append(start)
        ends.append(end)
    wall_ns = max(ends) - min(starts) if rows else None
    if wall_ns is not None and wall_ns <= 0:
        raise ValueError("Measured window must have positive duration")
    wall_s = wall_ns / 1_000_000_000 if wall_ns is not None else None
    output_tokens = sum(r["actual_output_tokens"] for r in success)
    failed = len(rows) - len(success)
    result = {
        "valid": len(success) == expected_requests,
        "expected_requests": expected_requests, "attempted_requests": len(rows),
        "successful_requests": len(success), "failed_requests": failed,
        "unattempted_requests": expected_requests - len(rows),
        "failure_rate": failed / len(rows) if rows else None,
        "wall_time_s": wall_s, "successful_output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / wall_s if wall_s else None,
        "requests_per_s": len(success) / wall_s if wall_s else None,
        "tpot_unobserved_requests": sum(
            observation["tpot_ms_per_token"] is None for observation in observations
        ),
    }
    for name in ("ttft_ms", "e2e_ms", "tpot_ms_per_token"):
        values = [m[name] for m in observations if m[name] is not None]
        result[f"{name}_p50"] = quantile(values, 0.5)
        result[f"{name}_p95"] = quantile(values, 0.95)
    return result


AGGREGATE_METRICS = (
    "output_tokens_per_s", "requests_per_s", "ttft_ms_p50", "ttft_ms_p95",
    "e2e_ms_p50", "e2e_ms_p95", "tpot_ms_per_token_p50", "tpot_ms_per_token_p95",
)


def select_attempts(cells: list[dict]) -> dict:
    """The first valid attempt, otherwise the first attempt; never the fastest."""
    chosen = {}
    seen_attempts = set()
    for cell in sorted(cells, key=lambda c: c["attempt"]):
        key = (cell["engine"], cell["profile"], cell["concurrency"], cell["repetition"])
        attempt_key = (*key, cell["attempt"])
        if attempt_key in seen_attempts:
            raise ValueError("Duplicate cell attempt")
        seen_attempts.add(attempt_key)
        if key not in chosen or (not chosen[key]["valid"] and cell["valid"]):
            chosen[key] = cell
    return chosen


RATIO_METRICS = ("output_tokens_per_s", "ttft_ms_p50", "ttft_ms_p95", "e2e_ms_p95")


def paired_ratios(cells: list[dict], *, expected_repetitions: int) -> list[dict]:
    """SGLang divided by vLLM within each matched profile/concurrency/repetition."""
    chosen = select_attempts(cells)
    groups = {}
    for (engine, profile, concurrency, repetition), baseline in chosen.items():
        other = chosen.get(("sglang", profile, concurrency, repetition))
        if engine != "vllm" or not baseline["valid"] or not other or not other["valid"]:
            continue
        groups.setdefault((profile, concurrency), {})[repetition] = {
            metric: other[metric] / baseline[metric]
            if baseline[metric] and other[metric] is not None else None
            for metric in RATIO_METRICS
        }
    result = []
    for (profile, concurrency), repeats in sorted(groups.items()):
        row = {
            "profile": profile, "concurrency": concurrency, "matched_repetitions": len(repeats),
            "complete": set(repeats) == set(range(1, expected_repetitions + 1)),
        }
        for metric in RATIO_METRICS:
            values = [repeats[r][metric] for r in sorted(repeats) if repeats[r][metric] is not None]
            for repetition in sorted(repeats):
                row[f"{metric}_ratio_r{repetition}"] = repeats[repetition][metric]
            row[f"{metric}_ratio_median"] = median(values) if values else None
            row[f"{metric}_ratio_min"] = min(values) if values else None
            row[f"{metric}_ratio_max"] = max(values) if values else None
        result.append(row)
    return result


def aggregate(cells: list[dict], *, expected_repetitions: int) -> list[dict]:
    """Select the first valid attempt, retain incomplete groups, never pool requests."""
    chosen = select_attempts(cells)
    groups = {}
    for key, cell in chosen.items():
        groups.setdefault(key[:3], []).append(cell)
    result = []
    for (engine, profile, concurrency), repeats in sorted(groups.items()):
        valid = [r for r in repeats if r["valid"]]
        repetitions = {r["repetition"] for r in valid}
        row = {
            "engine": engine, "profile": profile, "concurrency": concurrency,
            "valid_repetitions": len(valid), "expected_repetitions": expected_repetitions,
            "complete": repetitions == set(range(1, expected_repetitions + 1)),
        }
        for name in AGGREGATE_METRICS:
            values = [r[name] for r in valid if r[name] is not None]
            row[name] = median(values) if values else None
            row[name + "_min"] = min(values) if values else None
            row[name + "_max"] = max(values) if values else None
        result.append(row)
    return result
