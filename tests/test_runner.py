import asyncio
import copy
import importlib.util
import sys
import tempfile
import time
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from inference_bench.artifacts import safe_path, verify
from inference_bench.budget import BudgetError, Ledger
from inference_bench.config import Config, load_config
from inference_bench.engines import command, validate_hardware
from inference_bench.provenance import read_json, source_tree_hash
from inference_bench.runner import Deadline, ProcessGroup, run_block, wait_ready

ROOT = Path(__file__).resolve().parents[1]


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ledger.json"
        self.config = load_config(ROOT / "configs/benchmark.toml")
        self.ledger = Ledger(self.path, "campaign", self.config)

    def test_unresolved_call_survives_restart_and_holds_full_reservation(self):
        lease = self.ledger.reserve("pilot", 600, now=0)
        self.ledger.attach_call(lease["id"], "recorded-call")
        resumed = Ledger(self.path, "campaign", self.config)
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

    def test_policy_or_campaign_change_cannot_reset_usage(self):
        self.ledger.reserve("pilot", 10)
        with self.assertRaises(BudgetError):
            Ledger(self.path, "another-campaign", self.config).summary()
        data = copy.deepcopy(self.config.data)
        data["budget"]["total_usd"] = 76
        with self.assertRaises(BudgetError):
            Ledger(self.path, "campaign", Config(data)).summary()

    def test_second_controller_lock_is_rejected(self):
        with self.ledger._exclusive():
            with self.assertRaises(BudgetError):
                self.ledger.reserve("pilot", 10)


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


class BlockTests(unittest.IsolatedAsyncioTestCase):
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
            with patch("inference_bench.runner.engines.inspect_hardware", side_effect=ValueError("wrong GPU")):
                with self.assertRaisesRegex(ValueError, "wrong GPU"):
                    await run_block(
                        root, config, engine="vllm", model_path="/models/not-loaded",
                        cells=[config.cells()[0]], prompts=[],
                        context={"attempt": 1, "harness_sha256": source_tree_hash()},
                        deadline_unix=time.time() + 90,
                    )
            verify(root)
            self.assertEqual(read_json(root / "status.json")["state"], "failed")
            self.assertEqual(read_json(root / "manifest.json")["engine"], "vllm")

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
