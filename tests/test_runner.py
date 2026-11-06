import asyncio
import copy
import importlib.util
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

from inference_bench.artifacts import read_jsonl, safe_path, seal, verify, write_json, write_jsonl
from inference_bench.budget import BudgetError, Ledger
from inference_bench.config import Config, load_config
from inference_bench.engines import command, validate_hardware
from inference_bench.provenance import digest_file, read_json, source_tree_hash
from inference_bench.runner import Deadline, ProcessGroup, run_block, wait_ready

ROOT = Path(__file__).resolve().parents[1]


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ledger.json"
        self.config = load_config(ROOT / "configs/benchmark.toml")
        self.ledger = Ledger(self.path, self.config)

    def test_unresolved_call_survives_restart_and_holds_full_reservation(self):
        lease = self.ledger.reserve("pilot", 600, now=0)
        self.ledger.attach_call(lease["id"], "recorded-call")
        resumed = Ledger(self.path, self.config)
        self.assertEqual(resumed.summary()["exposure_seconds"], 600)
        with self.assertRaises(BudgetError):
            resumed.reserve("formal", 1, now=10_000_000)
        with self.assertRaises(BudgetError):
            resumed.settle(lease["id"], 10, confirmed_stopped=False)
        self.assertEqual(resumed.summary()["exposure_seconds"], 600)
        resumed.settle(lease["id"], 10, confirmed_stopped=True)
        self.assertEqual(resumed.summary()["exposure_seconds"], 10)
        self.assertEqual(Decimal(resumed.summary()["exposure_usd"]), self.config.rate_per_second * 10)

    def test_stage_and_campaign_time_limits(self):
        for stage, count in (("pilot", 1), ("formal", 4), ("rerun", 1)):
            for _ in range(count):
                lease = self.ledger.reserve(stage, 6000)
                self.ledger.settle(lease["id"], 6000, confirmed_stopped=True)
            with self.assertRaises(BudgetError):
                self.ledger.reserve(stage, 1)
        self.assertEqual(self.ledger.summary()["exposure_seconds"], 36000)
        self.assertLess(Decimal(self.ledger.summary()["exposure_usd"]), Decimal("60"))

    def test_overrun_is_charged_and_prevents_further_work(self):
        lease = self.ledger.reserve("pilot", 10)
        self.ledger.settle(lease["id"], 15, confirmed_stopped=True)
        self.assertEqual(self.ledger.summary()["exposure_seconds"], 15)
        with self.assertRaises(BudgetError):
            self.ledger.reserve("formal", 10)

    def test_pilot_and_formal_protocols_share_one_ledger_but_policy_cannot_reset(self):
        from inference_bench.planning import protocol_candidates
        lease = self.ledger.reserve("pilot", 10)
        self.ledger.settle(lease["id"], 10, confirmed_stopped=True)
        for _, compact in protocol_candidates(self.config):
            self.assertEqual(Ledger(self.path, compact).summary()["exposure_seconds"], 10)
        for section, key, value in (
            ("budget", "ledger_id", "another-ledger"), ("budget", "total_usd", 76),
            ("pricing", "gpu_per_second", 0.001),
        ):
            with self.subTest(key=key):
                data = copy.deepcopy(self.config.data)
                data[section][key] = value
                with self.assertRaises(BudgetError):
                    Ledger(self.path, Config(data)).summary()

    def test_billing_can_only_increase_exposure_and_scales_later_reservations(self):
        lease = self.ledger.reserve("pilot", 1000, now=0)
        self.ledger.settle(lease["id"], 1000, confirmed_stopped=True)
        charged = self.config.rate_per_second * 1000
        self.ledger.record_billing(2000, str(charged / 2), "1", "a" * 64, now=3000)
        summary = self.ledger.summary()
        self.assertEqual(Decimal(summary["exposure_usd"]), charged)
        self.assertEqual(Decimal(summary["billing"]["rate_multiplier"]), 1)
        self.ledger.record_billing(2500, str(charged * Decimal("1.5")), "1", "b" * 64, now=3000)
        summary = self.ledger.summary()
        self.assertEqual(Decimal(summary["exposure_usd"]), charged * Decimal("1.5"))
        self.assertEqual(Decimal(summary["billing"]["rate_multiplier"]), Decimal("1.5"))
        self.assertTrue(summary["warnings"])
        # 23,900 seconds fits at list price, but not at the reconciled 1.5x effective rate.
        data = copy.deepcopy(self.config.data)
        data["experiment"]["block_timeout_s"] = 40000
        wide = Config(data)
        with self.assertRaisesRegex(BudgetError, "allocation exceeded"):
            Ledger(self.path, wide).reserve("formal", 23900, now=4000)
        Ledger(self.path.with_name("unreconciled.json"), wide).reserve("formal", 23900, now=4000)

    def test_billing_period_cannot_split_a_lease_or_skip_an_unsettled_call(self):
        lease = self.ledger.reserve("pilot", 600, now=100)
        with self.assertRaisesRegex(BudgetError, "inside a lease"):
            self.ledger.record_billing(400, "0", "0", "a" * 64, now=1000)
        with self.assertRaisesRegex(BudgetError, "Settle every call"):
            self.ledger.record_billing(800, "0", "0", "a" * 64, now=1000)
        self.ledger.settle(lease["id"], 500, confirmed_stopped=True)
        self.ledger.record_billing(800, "1", "0.5", "a" * 64, now=1000)
        with self.assertRaisesRegex(BudgetError, "reconciled billing period"):
            self.ledger.reserve("pilot", 10, now=700)
        for through, gpu, other, digest in (
            (900, "0.9", "0.5", "b" * 64), (700, "2", "1", "b" * 64), (900, "2", "1", "not-a-hash"),
            (2000, "2", "1", "b" * 64), (900, "-1", "1", "b" * 64), (900, "NaN", "1", "b" * 64),
        ):
            with self.subTest(through=through, gpu=gpu, digest=digest):
                with self.assertRaises(BudgetError):
                    self.ledger.record_billing(through, gpu, other, digest, now=1000)
        data = self.ledger.read()
        data["billing"][0]["gpu_container_usd"] = "-5"
        write_json(self.path, data)
        with self.assertRaises(BudgetError):
            self.ledger.summary()

    def test_other_costs_beyond_their_allocation_consume_the_reserve(self):
        before = Decimal(self.ledger.summary()["remaining"]["gpu_container_usd"])
        self.ledger.record_billing(10, "0", "4", "a" * 64, now=20)
        self.assertEqual(Decimal(self.ledger.summary()["remaining"]["gpu_container_usd"]), before)
        self.ledger.record_billing(11, "0", "8", "b" * 64, now=20)
        summary = self.ledger.summary()
        self.assertEqual(Decimal(summary["remaining"]["gpu_container_usd"]), before - 3)
        self.assertTrue(any("reserve" in warning for warning in summary["warnings"]))

    def test_second_controller_lock_is_rejected(self):
        with self.ledger._exclusive():
            with self.assertRaises(BudgetError):
                self.ledger.reserve("pilot", 10)

    def test_corrupt_ledger_cannot_erase_or_reduce_exposure(self):
        self.ledger.reserve("pilot", 10, now=0)
        original = self.ledger.read()
        for field, value in (
            ("status", "unknown"), ("stage", "unknown"), ("reserved_seconds", -10),
            ("reserved_usd", "-1"), ("reserved_usd", "NaN"), ("deadline_unix", 11),
            ("remote_call_id", 123),
        ):
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(original)
                data["reservations"][0][field] = value
                write_json(self.path, data)
                with self.assertRaises(BudgetError):
                    self.ledger.summary()
        original["reservations"] *= 2
        write_json(self.path, original)
        with self.assertRaises(BudgetError):
            self.ledger.summary()

    def test_nonfinite_start_is_rejected_before_writing_a_reservation(self):
        for now in (float("inf"), float("nan"), True, 1e308):
            with self.assertRaises(BudgetError):
                self.ledger.reserve("pilot", 10, now=now)
        self.assertFalse(self.path.exists())


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        from inference_bench.examples import make_example
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.root = self.parent / "campaign"
        self.manifest = make_example(
            self.root, load_config(ROOT / "configs/benchmark.toml"),
            read_json(ROOT / "configs/sources.lock.json"),
            read_json(ROOT / "configs/engines.lock.json"),
        )
        self.config = Config(read_json(self.root / "protocol.json"))
        self.ledger = Ledger(self.parent / "ledger.json", self.config)
        write_json(self.ledger.path, self.ledger.read())
        self.cell = self.root / "runs/vllm/r1/attempt-1/cells/short-c1"
        self.missing = self.root / "runs/vllm/r1/attempt-1/cells/prefill-c4"
        self.unseal(self.root)

    @staticmethod
    def unseal(path):
        for name in ("COMPLETE.json", "CHECKSUMS.json"):
            (path / name).unlink(missing_ok=True)

    def interrupt(self, path):
        self.unseal(path)
        write_json(path / "status.json", {"status": "interrupted"})

    def plan(self):
        from inference_bench.planning import resume_plan
        return resume_plan(self.root, self.ledger.path, allow_synthetic=True)

    def test_recovery_preserves_completed_cells_and_plans_only_missing_or_invalid_work(self):
        self.interrupt(self.cell)
        shutil.rmtree(self.missing)
        before = {p.relative_to(self.parent): digest_file(p) for p in self.parent.rglob("*") if p.is_file()}
        result = self.plan()
        self.assertEqual(result["counts"]["complete"], 34)
        self.assertEqual(result["counts"]["pending"], 2)
        self.assertEqual(len(result["eligible_cells"]), 2)
        by_profile = {c["profile"]: c for c in result["eligible_cells"]}
        self.assertEqual(by_profile["short"]["attempt"], 2)
        self.assertEqual(by_profile["short"]["budget_stage"], "rerun")
        self.assertEqual(by_profile["short"]["replacement_for"], self.cell.relative_to(self.root).as_posix())
        self.assertEqual(by_profile["prefill"]["attempt"], 1)
        self.assertEqual(by_profile["prefill"]["budget_stage"], "formal")
        expected_order = [
            c.to_dict() for c in self.config.cells()
            if c.engine == "vllm" and c.repetition == 1 and (c.profile, c.concurrency) in (("short", 1), ("prefill", 4))
        ]
        self.assertEqual(
            [{k: c[k] for k in ("engine", "profile", "concurrency", "repetition")} for c in result["eligible_cells"]],
            expected_order,
        )
        self.assertTrue(all(c["requires_new_reservation"] and c["requires_fresh_process"] for c in result["eligible_cells"]))
        after = {p.relative_to(self.parent): digest_file(p) for p in self.parent.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_expired_remote_call_still_blocks_recovery_until_settled(self):
        self.interrupt(self.cell)
        reservation = self.ledger.reserve("formal", 60, now=0)
        self.ledger.attach_call(reservation["id"], "unresolved-call")
        result = self.plan()
        self.assertEqual(result["eligible_cells"], [])
        self.assertEqual(result["counts"]["blocked"], 1)
        self.assertEqual(result["budget"]["exposure_seconds"], 60)
        self.ledger.settle(reservation["id"], 10, confirmed_stopped=True)
        self.assertEqual(len(self.plan()["eligible_cells"]), 1)

    def test_first_valid_attempt_wins_and_second_failure_exhausts_replacements(self):
        from inference_bench.metrics import summarize
        second = self.root / "runs/vllm/r1/another-block/cells/short-c1"
        shutil.copytree(self.cell, second)
        metadata = read_json(second / "manifest.json")
        metadata.update(attempt=2, replacement_for=self.cell.relative_to(self.root).as_posix())
        records = read_jsonl(second / "requests.jsonl.gz")
        for record in records:
            record.update(attempt=2, replacement_for=metadata["replacement_for"])
            for field in ("dispatch_ns", "headers_ns", "first_content_ns", "last_content_ns", "done_ns"):
                record[field] //= 2
        write_json(second / "manifest.json", metadata)
        write_jsonl(second / "requests.jsonl.gz", records)
        write_json(second / "summary.json", summarize(records, expected_requests=8))
        self.unseal(second)
        seal(second)

        def selected():
            return next(c for c in self.plan()["cells"] if c["cell_id"] == "vllm/r1/short-c1")

        self.assertEqual(selected()["selected_artifact"], self.cell.relative_to(self.root).as_posix())
        self.interrupt(self.cell)
        self.assertEqual(selected()["selected_artifact"], second.relative_to(self.root).as_posix())
        self.interrupt(second)
        self.assertEqual(selected()["state"], "exhausted")
        self.assertEqual(self.plan()["eligible_cells"], [])

    def test_duplicate_third_and_orphan_attempts_are_rejected(self):
        extra = self.root / "runs/vllm/r1/extra-block/cells/short-c1"
        shutil.copytree(self.cell, extra)
        self.interrupt(extra)
        with self.assertRaisesRegex(ValueError, "Duplicate cell attempt"):
            self.plan()
        metadata = read_json(extra / "manifest.json")
        write_json(extra / "manifest.json", {**metadata, "attempt": 3})
        with self.assertRaisesRegex(ValueError, "replacement-attempt number"):
            self.plan()
        write_json(extra / "manifest.json", {
            **metadata, "attempt": 2, "replacement_for": self.cell.relative_to(self.root).as_posix(),
        })
        shutil.rmtree(self.cell)
        with self.assertRaisesRegex(ValueError, "retain and link"):
            self.plan()

    def test_missing_identity_or_corrupted_completed_evidence_stops_recovery(self):
        unknown = self.cell.parent / "unknown"
        unknown.mkdir()
        with self.assertRaisesRegex(ValueError, "lacks its attempt manifest"):
            self.plan()
        unknown.rmdir()
        (self.cell / "requests.jsonl.gz").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.plan()

    def test_current_harness_and_existing_budget_history_are_required(self):
        from inference_bench.planning import resume_plan
        self.interrupt(self.cell)
        with patch("inference_bench.planning.source_tree_hash", return_value="0" * 64):
            result = self.plan()
        self.assertEqual(result["eligible_cells"], [])
        self.assertTrue(any("harness source differs" in reason for reason in result["blockers"]))
        with self.assertRaisesRegex(BudgetError, "existing project ledger"):
            resume_plan(self.root, self.parent / "missing-ledger.json", allow_synthetic=True)
        self.assertFalse((self.parent / "missing-ledger.json").exists())

    def test_exhausted_replacement_budget_preserves_formal_allowance(self):
        self.interrupt(self.cell)
        shutil.rmtree(self.missing)
        reservation = self.ledger.reserve("rerun", 6000)
        self.ledger.settle(reservation["id"], 6000, confirmed_stopped=True)
        result = self.plan()
        self.assertEqual(result["counts"]["budget_stopped"], 1)
        self.assertEqual(result["available"]["rerun_seconds"], 0)
        self.assertEqual(result["available"]["formal_seconds"], 24000)
        self.assertEqual(len(result["eligible_cells"]), 1)
        self.assertEqual(result["eligible_cells"][0]["attempt"], 1)

    def test_status_without_completion_marker_never_becomes_valid_on_resume(self):
        self.unseal(self.cell)
        result = self.plan()
        cell = next(c for c in result["cells"] if c["cell_id"] == "vllm/r1/short-c1")
        self.assertEqual(cell["next_attempt"], 2)
        self.assertEqual(cell["attempts"][0]["status"], "interrupted")
        self.assertFalse(cell["attempts"][0]["valid"])

    def test_actual_allocations_require_receipts_even_before_first_cell(self):
        from inference_bench.planning import _check_budget_links
        # Test the accounting gate directly; the fixture itself remains synthetic.
        with self.assertRaisesRegex(BudgetError, "reservation receipt"):
            _check_budget_links(
                self.root, {**self.manifest, "evidence_kind": "measurement"}, [],
                self.ledger.summary(),
            )


class EngineTests(unittest.TestCase):
    def test_commands_preserve_controls_and_paths_as_single_arguments(self):
        config = load_config(ROOT / "configs/benchmark.toml")
        for engine, cache_flag in (
            ("vllm", "--no-enable-prefix-caching"), ("sglang", "--disable-radix-cache"),
        ):
            args = command(engine, "/models/a directory", config)
            self.assertIn("/models/a directory", args)
            self.assertIn(cache_flag, args)
            self.assertNotIn("--enforce-eager", args)
            self.assertNotIn("--speculative-algorithm", args)

    def test_wrong_gpu_partition_or_device_count_is_rejected(self):
        gpu = {"name": "NVIDIA H200", "memory_mib": 143771, "mig_mode": "Disabled"}
        validate_hardware([gpu])
        for devices in ([], [gpu, gpu], [{**gpu, "name": "NVIDIA H100"}],
                        [{**gpu, "name": "NVIDIA H200 NVL"}],
                        [{**gpu, "memory_mib": 80000}], [{**gpu, "mig_mode": "Enabled"}]):
            with self.assertRaises(ValueError):
                validate_hardware(devices)

    def test_modal_contract_does_not_register_a_gpu_during_offline_import(self):
        spec = importlib.util.spec_from_file_location("benchmark_modal_app", ROOT / "modal_app.py")
        module = importlib.util.module_from_spec(spec)
        # Test the resource contracts without an SDK or network connection.
        with patch.dict(sys.modules, {"modal": None}):
            spec.loader.exec_module(module)
        self.assertIsNone(module.app)
        self.assertNotIn("gpu", module.cpu_options())
        options = module.planned_gpu_options()
        self.assertEqual(options["gpu"], "H200")
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["retries"], 0)
        self.assertTrue(options["single_use_containers"])
        self.assertNotIn("region", options)


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux/WSL process lifecycle")
class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bench process ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_expired_lease_does_not_start_process(self):
        with patch("subprocess.Popen") as popen:
            with self.assertRaises(TimeoutError):
                with ProcessGroup(["unused"], self.root / "log", Deadline(time.time() - 1)):
                    pass
        popen.assert_not_called()

    def test_watchdog_stops_hung_process_without_event_loop(self):
        with ProcessGroup(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            self.root / "log", Deadline(time.time() + 0.2), grace_s=0.05,
        ) as process:
            process.process.wait(timeout=3)
            self.assertIsNotNone(process.process.returncode)

    def test_cleanup_kills_grandchild_even_after_parent_exits(self):
        pidfile = self.root / "child.pid"
        child = (
            "import os, signal, sys, time; from pathlib import Path;"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
        )
        parent = (
            "import subprocess, sys, time;"
            "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]);"
            "time.sleep(60)"
        )
        with self.assertRaisesRegex(RuntimeError, "simulated failure"):
            with ProcessGroup(
                [sys.executable, "-c", parent, child, str(pidfile)],
                self.root / "server.log", Deadline(time.time() + 10), grace_s=0.05,
            ):
                until = time.monotonic() + 3
                while not pidfile.exists() and time.monotonic() < until:
                    time.sleep(0.01)
                self.assertTrue(pidfile.exists())
                pid = int(pidfile.read_text())
                raise RuntimeError("simulated failure")
        status = Path(f"/proc/{pid}/status")
        def alive():
            try:
                return "State:\tZ" not in status.read_text()
            except (FileNotFoundError, ProcessLookupError):
                return False

        until = time.monotonic() + 3
        while alive() and time.monotonic() < until:
            time.sleep(0.01)
        self.assertFalse(alive())

    def test_backward_wall_clock_does_not_extend_lease(self):
        now = time.time()
        deadline = Deadline(now + 5)
        with patch("inference_bench.runner.time.time", return_value=now - 100):
            self.assertLessEqual(deadline.remaining(), 5)

    def test_artifact_paths_cannot_escape_or_follow_symlinks(self):
        for relative in ("../private", "/absolute", "a\\..\\private"):
            with self.assertRaises(ValueError):
                safe_path(self.root, relative)
        (self.root / "link").symlink_to("/tmp", target_is_directory=True)
        with self.assertRaises(ValueError):
            safe_path(self.root, "link/file")


def probe_prompts(stops=(1,)):
    from inference_bench.qualification import PROBES
    return [
        {
            "request_id": f'conformance:{p["name"]}', "probe_name": p["name"], "phase": "conformance",
            "profile": None, "repetition": None, "prompt_ids": [5, 6], "stop_token_ids": list(stops),
        }
        for p in PROBES
    ]


def probe_result(prompt, *, broken=None, **kwargs):
    """Conformance replies from a fake server double; no socket is opened."""
    from inference_bench.qualification import PROBE_BY_NAME
    probe = PROBE_BY_NAME[kwargs["context"]["probe_name"]]
    honored = not kwargs["ignore_eos"]
    text = {"arithmetic": "42", "capital": "Paris", "eos_honored": "yes"}.get(
        probe["name"], "A plain sentence that summarizes several distinct details of the passage for readers.",
    )
    record = {
        **kwargs["context"], "status": "success", "requested_output_tokens": probe["max_tokens"],
        "finish_reason": "stop" if honored else "length",
        "actual_output_tokens": 3 if honored else probe["max_tokens"], "response_text": text,
    }
    if probe["name"] == broken:
        record.update(response_text="\ufffd\ufffd", finish_reason="length", actual_output_tokens=probe["max_tokens"])
    return {"records": [record], "events": [], "summary": None}


class BlockTests(unittest.IsolatedAsyncioTestCase):
    def fake_engine(self, broken=None):
        @asynccontextmanager
        async def fake_client(*args):
            yield object()

        cells = []

        async def fake_run_cell(origin, prompts, **kwargs):
            if kwargs["phase"] == "conformance":
                self.assertEqual(kwargs["ignore_eos"], kwargs["stop_token_ids"] is None)
                return probe_result(prompts[0], broken=broken, **kwargs)
            cells.append((prompts, kwargs))
            return {
                "records": [], "events": [], "summary": {"valid": True},
                "client_loop_lag_ms": [], "client_lag_interval_ms": 10,
            }

        patches = (
            patch("inference_bench.runner.engines.inspect_hardware", return_value=[]),
            patch("inference_bench.runner.ProcessGroup"),
            patch("inference_bench.runner.http_client", fake_client),
            patch("inference_bench.runner.wait_ready", new_callable=AsyncMock,
                  return_value={"data": [{"id": "gemma4-bench"}]}),
            patch("inference_bench.runner._sample_gpu", new_callable=AsyncMock),
            patch("inference_bench.runner.run_cell", fake_run_cell),
        )
        return patches, cells

    async def test_pilot_uses_separate_inputs_and_records_block_timings_without_network(self):
        config = load_config(ROOT / "configs/benchmark.toml")
        cell = next(c for c in config.cells(phase="pilot") if c.engine == "vllm" and c.concurrency == 4)
        prompts = probe_prompts() + [
            {"profile": cell.profile, "repetition": 1, "phase": phase, "request_id": f"{phase}-{i}"}
            for phase, count in (("pilot", 16), ("pilot_warmup", 8), ("measurement", 256))
            for i in range(count)
        ]
        patches, calls = self.fake_engine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                await run_block(
                    root, config, engine="vllm", model_path="/unused", cells=[cell], prompts=prompts,
                    context={"evidence_kind": "synthetic", "harness_sha256": source_tree_hash()},
                    deadline_unix=time.time() + 90, phase="pilot",
                )
            self.assertEqual(len(calls), 2)
            (warmup, _), (measured, measured_kwargs) = calls
            self.assertEqual(len(warmup), 8)
            self.assertEqual(len(measured), 16)
            self.assertTrue(all(p["phase"] == "pilot_warmup" for p in warmup))
            self.assertTrue(all(p["phase"] == "pilot" for p in measured))
            self.assertEqual(measured_kwargs["phase"], "pilot")
            checks = read_json(root / "conformance/checks.json")
            self.assertEqual([c["status"] for c in checks], ["passed"] * 5)
            self.assertEqual(read_json(root / "models.json")["data"][0]["id"], "gemma4-bench")
            states = [h["state"] for h in read_json(root / "status.json")["history"]]
            self.assertLess(states.index("conformance_passed"), states.index("warming"))
            timing = read_json(root / "timing.json")
            self.assertLessEqual(timing["started_ns"], timing["ready_ns"])
            self.assertLessEqual(timing["ready_ns"], timing["finished_ns"])
            verify(root)

    async def test_failed_probe_stops_block_before_warmup_or_measurement(self):
        config = load_config(ROOT / "configs/benchmark.toml")
        cell = next(c for c in config.cells(phase="pilot") if c.engine == "sglang")
        prompts = probe_prompts() + [
            {"profile": cell.profile, "repetition": 1, "phase": phase, "request_id": f"{phase}-{i}"}
            for phase, count in (("pilot", 64), ("pilot_warmup", 64)) for i in range(count)
        ]
        patches, calls = self.fake_engine(broken="passage_summary")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                with self.assertRaisesRegex(RuntimeError, "passage_summary"):
                    await run_block(
                        root, config, engine="sglang", model_path="/unused", cells=[cell],
                        prompts=prompts, phase="pilot", deadline_unix=time.time() + 90,
                        context={"evidence_kind": "pilot", "harness_sha256": source_tree_hash()},
                    )
            self.assertEqual(calls, [])
            self.assertFalse((root / "cells").exists())
            checks = {c["name"]: c["status"] for c in read_json(root / "conformance/checks.json")}
            self.assertEqual(checks["passage_summary"], "failed")
            self.assertEqual(read_json(root / "status.json")["state"], "failed")
            verify(root)

    async def test_missing_inputs_are_rejected_before_any_artifact_or_hardware_check(self):
        config = load_config(ROOT / "configs/benchmark.toml")
        cell = config.cells()[0]
        complete = probe_prompts() + [
            {"profile": cell.profile, "repetition": cell.repetition, "phase": phase, "request_id": f"{phase}-{i}"}
            for phase, count in (("measurement", 256), ("warmup", 64)) for i in range(count)
        ]
        for prompts, message in (
            (complete[1:], "every conformance probe"),
            (probe_prompts(stops=()) + complete[5:], "stop token"),
            ([p for p in complete if p["request_id"] != "measurement-255"], "cannot supply"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                with patch("inference_bench.runner.engines.inspect_hardware") as probe:
                    with self.assertRaisesRegex(ValueError, message):
                        await run_block(
                            Path(directory) / "block", config, engine=cell.engine, model_path="/unused",
                            cells=[cell], prompts=prompts, deadline_unix=time.time() + 90,
                            context={"harness_sha256": source_tree_hash()},
                        )
                    probe.assert_not_called()
                self.assertFalse((Path(directory) / "block").exists())

    async def test_changed_harness_is_rejected_before_hardware_or_model_start(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(ROOT / "configs/benchmark.toml")
            with patch("inference_bench.runner.engines.inspect_hardware") as probe:
                with self.assertRaisesRegex(ValueError, "harness source"):
                    await run_block(
                        Path(directory) / "block", config, engine="vllm",
                        model_path="/models/not-loaded", cells=[config.cells()[0]], prompts=[],
                        context={"harness_sha256": "0" * 64}, deadline_unix=time.time() + 90,
                    )
                probe.assert_not_called()

    async def test_readiness_deadline_bounds_a_hanging_http_operation(self):
        class Server:
            deadline = Deadline(time.time() + 10)
            def check(self):
                pass

        class HangingClient:
            async def get(self, *args, **kwargs):
                await asyncio.Event().wait()

        with self.assertRaisesRegex(TimeoutError, "readiness"):
            await asyncio.wait_for(
                wait_ready(HangingClient(), "http://127.0.0.1:1", Server(), 0.02), timeout=2,
            )

    async def test_failed_hardware_check_produces_durable_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "block"
            config = load_config(ROOT / "configs/benchmark.toml")
            cell = config.cells()[0]
            prompts = probe_prompts() + [
                {"profile": cell.profile, "repetition": cell.repetition, "phase": phase, "request_id": f"{phase}-{i}"}
                for phase, count in (("measurement", 256), ("warmup", 64)) for i in range(count)
            ]
            with patch("inference_bench.runner.engines.inspect_hardware", side_effect=ValueError("wrong GPU")):
                with self.assertRaisesRegex(ValueError, "wrong GPU"):
                    await run_block(
                        root, config, engine=cell.engine, model_path="/models/not-loaded",
                        cells=[cell], prompts=prompts,
                        context={"attempt": 1, "harness_sha256": source_tree_hash()},
                        deadline_unix=time.time() + 90,
                    )
            verify(root)
            self.assertEqual(read_json(root / "status.json")["state"], "failed")
            self.assertEqual(read_json(root / "manifest.json")["engine"], cell.engine)

    async def test_readiness_does_not_swallow_programming_errors(self):
        class Server:
            deadline = Deadline(time.time() + 10)
            def check(self):
                pass

        class BrokenClient:
            async def get(self, *args, **kwargs):
                raise RuntimeError("programming bug")

        with self.assertRaisesRegex(RuntimeError, "programming bug"):
            await wait_ready(BrokenClient(), "http://127.0.0.1:1", Server(), 1)
