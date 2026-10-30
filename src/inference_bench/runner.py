"""Owned process groups, absolute leases, and durable block/cell outcomes."""

from __future__ import annotations

import asyncio
import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from . import engines
from .artifacts import seal, write_json, write_jsonl
from .client import http_client, run_cell
from .config import Cell, Config
from .provenance import environment, source_tree_hash
from .workloads import sampling_seed, select_prompts


class Deadline:
    def __init__(self, unix: float):
        if not math.isfinite(unix):
            raise ValueError("Deadline must be finite")
        self.unix = unix
        self.monotonic = time.monotonic() + max(0, unix - time.time())

    def remaining(self) -> float:
        return max(0, min(self.unix - time.time(), self.monotonic - time.monotonic()))

    def check(self) -> None:
        if self.remaining() <= 0:
            raise TimeoutError("The original block deadline has expired")


class ProcessGroup:
    """Linux/WSL process ownership, including cleanup after the parent exits."""

    def __init__(
        self, argv: list[str], log_path: Path, deadline: Deadline, *,
        grace_s: float = 5, log_limit: int = 8_388_608, env: dict | None = None,
    ):
        self.argv, self.log_path, self.deadline = argv, Path(log_path), deadline
        self.grace_s, self.log_limit, self.env = grace_s, log_limit, env
        self.process = None
        self._lock = threading.Lock()
        self._stopped = False
        self.log_error = None
        self.truncated_bytes = 0

    def __enter__(self):
        if os.name != "posix":
            raise RuntimeError("Run the process controller on Linux or inside WSL")
        self.deadline.check()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            self.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, env=self.env,
        )
        self.reader = threading.Thread(target=self._drain_log, daemon=True)
        self.reader.start()
        self.timer = threading.Timer(self.deadline.remaining(), self.stop)
        self.timer.daemon = True
        self.timer.start()
        return self

    def _drain_log(self):
        written = 0
        tail = b""
        try:
            with self.log_path.open("wb") as output:
                while block := self.process.stdout.read1(65536):
                    keep = min(len(block), max(0, self.log_limit - written))
                    if keep:
                        output.write(block[:keep])
                        output.flush()
                        written += keep
                    rest = block[keep:]
                    if rest:
                        self.truncated_bytes += len(rest)
                        tail = (tail + rest)[-1_048_576:]
                if self.truncated_bytes:
                    output.write(b"\n[log truncated; final bytes follow]\n" + tail)
        except OSError as exc:
            self.log_error = str(exc)
            # Keep draining so a logging error cannot deadlock the server.
            while self.process.stdout.read1(65536):
                pass
        finally:
            self.process.stdout.close()

    def check(self):
        self.deadline.check()
        if self.log_error:
            raise RuntimeError(f"Server log failed: {self.log_error}")
        if self.process.poll() is not None:
            raise RuntimeError(f"Server exited with code {self.process.returncode}")

    def stop(self):
        with self._lock:
            if self._stopped or self.process is None:
                return
            self._stopped = True
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=self.grace_s)
            except subprocess.TimeoutExpired:
                pass
            # The leader may already have exited while grandchildren remain.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=max(1, self.grace_s))

    def __exit__(self, exc_type, exc, tb):
        self.timer.cancel()
        self.stop()
        self.reader.join(timeout=5)
        if self.reader.is_alive():
            raise RuntimeError("Server log pipe remained open after process-group cleanup")


async def wait_ready(client, origin: str, server: ProcessGroup, timeout_s: float) -> dict:
    import httpx
    try:
        async with asyncio.timeout(min(timeout_s, server.deadline.remaining())):
            while True:
                server.check()
                try:
                    response = await client.get(origin + "/health", timeout=2)
                    if response.status_code == 200:
                        models = await client.get(origin + "/v1/models", timeout=2)
                        models.raise_for_status()
                        data = models.json()
                        if any(m.get("id") == "gemma4-bench" for m in data.get("data", [])):
                            return data
                except (OSError, ValueError, httpx.HTTPError):
                    pass
                await asyncio.sleep(0.25)
    except TimeoutError as exc:
        raise TimeoutError("Server readiness deadline expired") from exc


def _gpu_sample() -> dict:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=3, check=True,
    )
    times = os.times()
    return {
        "unix": time.time(), "gpu_csv": result.stdout.strip(),
        "cpu_times_s": {
            key: getattr(times, key)
            for key in ("user", "system", "children_user", "children_system")
        },
    }


async def _sample_gpu(rows: list[dict]):
    while True:
        try:
            rows.append(await asyncio.to_thread(_gpu_sample))
        except (OSError, subprocess.SubprocessError) as exc:
            rows.append({"unix": time.time(), "error": type(exc).__name__})
        await asyncio.sleep(1)


async def run_block(
    directory: Path, config: Config, *, engine: str, model_path: str,
    cells: list[Cell], prompts: list[dict], context: dict, deadline_unix: float,
    phase: str = "measurement", port: int = 8000,
) -> dict:
    """Run one fresh server. Remote wrappers own reservation/Volume commits."""
    if phase not in ("measurement", "pilot"):
        raise ValueError("Block phase must be measurement or pilot")
    if not cells or any(c.engine != engine for c in cells):
        raise ValueError("A block must contain cells for one engine")
    if len({c.repetition for c in cells}) != 1:
        raise ValueError("A fresh process is required for each repetition")
    # Leave shutdown and final artifact writes inside the original reservation.
    deadline = Deadline(deadline_unix - 30)
    deadline.check()
    if context.get("harness_sha256") != source_tree_hash():
        raise ValueError("Worker harness source differs from the frozen campaign")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    history = []
    samples = []
    sampler = None
    outcome = {"status": "started", "error": None, "completed_cells": 0}

    def transition(state: str):
        history.append({"state": state, "unix": time.time()})
        write_json(directory / "status.json", {**outcome, "state": state, "history": history})

    transition("started")
    write_json(directory / "manifest.json", {
        **context, "engine": engine, "phase": phase, "deadline_unix": deadline_unix,
        "cells": [c.to_dict() for c in cells],
    })
    try:
        hardware = engines.inspect_hardware()
        write_json(directory / "environment.json", {**environment(), "devices": hardware})
        transition("hardware_checked")
        argv = engines.command(engine, model_path, config, port)
        write_json(directory / "server-command.json", {"argv": argv})
        env = {
            **os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
        transition("server_starting")
        with ProcessGroup(argv, directory / "server.log", deadline, env=env) as server:
            async with asyncio.timeout(deadline.remaining()):
                async with http_client(config.experiment["max_running_requests"]) as client:
                    origin = f"http://127.0.0.1:{port}"
                    await wait_ready(client, origin, server, config.experiment["ready_timeout_s"])
                    transition("ready")
                    sampler = asyncio.create_task(_sample_gpu(samples))
                    for cell in cells:
                        server.check()
                        metadata = {**context, **cell.to_dict(), "cell_id": cell.key}
                        cell_dir = directory / "cells" / f"{cell.profile}-c{cell.concurrency}"
                        cell_dir.mkdir(parents=True, exist_ok=False)
                        write_json(cell_dir / "manifest.json", metadata)
                        write_json(cell_dir / "status.json", {"status": "interrupted"})
                        shape = config.profiles[cell.profile]
                        warmup_n = max(
                            config.experiment["warmup_min"],
                            config.experiment["warmup_waves"] * cell.concurrency,
                        )
                        warmup = select_prompts(
                            prompts, cell.profile, cell.repetition, "warmup", warmup_n,
                        )
                        transition("warming")
                        warmed = await run_cell(
                            origin, warmup, concurrency=cell.concurrency,
                            output_tokens=shape["output_tokens"],
                            seed=sampling_seed(config.experiment["seed"], cell.profile, cell.repetition, "warmup"),
                            timeout_s=config.experiment["request_timeout_s"], phase="warmup",
                            context=metadata, client=client,
                        )
                        write_jsonl(cell_dir / "warmup.jsonl.gz", warmed["records"])
                        if not warmed["summary"]["valid"]:
                            raise RuntimeError("Warmup failed; measured work was not started")
                        measured = select_prompts(
                            prompts, cell.profile, cell.repetition, "measurement",
                            config.experiment["requests_per_cell"],
                        )
                        transition("measuring")
                        result = await run_cell(
                            origin, measured, concurrency=cell.concurrency,
                            output_tokens=shape["output_tokens"],
                            seed=sampling_seed(config.experiment["seed"], cell.profile, cell.repetition, phase),
                            timeout_s=config.experiment["request_timeout_s"], phase=phase,
                            context=metadata, client=client,
                        )
                        write_jsonl(cell_dir / "requests.jsonl.gz", result["records"])
                        write_jsonl(cell_dir / "stream-events.jsonl.gz", result["events"])
                        write_json(cell_dir / "summary.json", result["summary"])
                        write_json(cell_dir / "client-diagnostics.json", {
                            "loop_lag_ms": result["client_loop_lag_ms"],
                            "lag_interval_ms": result["client_lag_interval_ms"],
                        })
                        write_json(cell_dir / "status.json", {
                            "status": "complete" if result["summary"]["valid"] else "failed",
                        })
                        seal(cell_dir)
                        outcome["completed_cells"] += 1
                        if not result["summary"]["valid"]:
                            raise RuntimeError("Measured cell failed; remaining cells were stopped")
        outcome["status"] = "complete"
    except (Exception, asyncio.CancelledError) as exc:
        outcome.update(status="interrupted" if isinstance(exc, asyncio.CancelledError) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if sampler:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
        write_jsonl(directory / "gpu-host-samples.jsonl.gz", samples)
        transition(outcome["status"])
        seal(directory)
    return outcome
