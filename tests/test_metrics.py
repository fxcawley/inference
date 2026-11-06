import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from inference_bench.analysis import analyze, load_campaign
from inference_bench.artifacts import bundle, read_jsonl, seal, verify, write_json, write_jsonl
from inference_bench.config import load_config
from inference_bench.examples import make_example
from inference_bench.metrics import aggregate, quantile, request_metrics, summarize
from inference_bench.provenance import read_json

ROOT = Path(__file__).resolve().parents[1]


def request(request_id="r1", *, dispatch=0, done=700_000_000):
    return {
        "request_id": request_id, "phase": "measurement", "status": "success",
        "dispatch_ns": dispatch, "headers_ns": dispatch + 10_000_000,
        "first_content_ns": dispatch + 100_000_000, "last_content_ns": dispatch + 500_000_000,
        "done_ns": done, "error_ns": None, "actual_input_tokens": 512,
        "actual_output_tokens": 3, "expected_input_tokens": 512, "requested_output_tokens": 3,
        "http_status": 200, "content_event_count": 2,
    }


def reseal(path):
    (path / "COMPLETE.json").unlink()
    (path / "CHECKSUMS.json").unlink()
    seal(path)


class MetricTests(unittest.TestCase):
    def test_units_quantiles_and_finite_batch_window(self):
        a = request()
        b = request("r2", dispatch=500_000_000, done=1_500_000_000)
        metrics = request_metrics(a)
        self.assertEqual(metrics["ttft_ms"], 100)
        self.assertEqual(metrics["e2e_ms"], 700)
        self.assertEqual(metrics["tpot_ms_per_token"], 200)
        result = summarize([a, b], expected_requests=2)
        self.assertEqual(result["wall_time_s"], 1.5)
        self.assertEqual(result["output_tokens_per_s"], 4)
        self.assertEqual(result["e2e_ms_p95"], 985)
        self.assertAlmostEqual(quantile([1, 2, 3, 4], 0.95), 3.85)

    def test_missing_usage_is_never_estimated(self):
        row = request()
        row["actual_output_tokens"] = None
        with self.assertRaises(ValueError):
            summarize([row], expected_requests=1)

    def test_single_chunk_and_one_token_tpot_are_unobservable(self):
        row = request()
        row["content_event_count"] = 1
        self.assertIsNone(request_metrics(row)["tpot_ms_per_token"])
        row.update(content_event_count=2, actual_output_tokens=1, requested_output_tokens=1)
        self.assertIsNone(request_metrics(row)["tpot_ms_per_token"])

    def test_failure_window_and_unattempted_work_remain_visible(self):
        failed = request("failed", dispatch=300_000_000)
        failed.update(status="failed", done_ns=None, error_ns=2_000_000_000, actual_output_tokens=None)
        summary = summarize([request(), failed], expected_requests=4)
        self.assertFalse(summary["valid"])
        self.assertEqual(summary["failure_rate"], 0.5)
        self.assertEqual(summary["unattempted_requests"], 2)
        self.assertEqual(summary["output_tokens_per_s"], 1.5)

    def test_replacements_use_first_valid_not_fastest(self):
        base = {
            "engine": "vllm", "profile": "short", "concurrency": 1,
            **summarize([request()], expected_requests=1),
        }
        cells = [{**base, "attempt": 1, "repetition": r} for r in (1, 2, 3)]
        cells.append({**cells[0], "attempt": 2, "output_tokens_per_s": 100000})
        result = aggregate(cells, expected_repetitions=3)[0]
        self.assertEqual(result["output_tokens_per_s"], base["output_tokens_per_s"])
        self.assertTrue(result["complete"])
        cells[0]["valid"] = False
        result = aggregate(cells, expected_repetitions=3)[0]
        self.assertEqual(result["output_tokens_per_s_max"], 100000)

    def test_paired_ratios_match_repetitions_and_use_first_valid_attempts(self):
        from inference_bench.metrics import RATIO_METRICS, paired_ratios
        def cell(engine, repetition, tps, attempt=1, valid=True):
            return {
                "engine": engine, "profile": "short", "concurrency": 4, "repetition": repetition,
                "attempt": attempt, "valid": valid,
                **{metric: tps if metric == "output_tokens_per_s" else 1000 / tps for metric in RATIO_METRICS},
            }
        cells = [cell("vllm", r, 100) for r in (1, 2, 3)] + [
            cell("sglang", 1, 120), cell("sglang", 2, 50, valid=False),
            cell("sglang", 2, 110, attempt=2), cell("sglang", 3, 90),
        ]
        row = paired_ratios(cells, expected_repetitions=3)[0]
        self.assertEqual(row["matched_repetitions"], 3)
        self.assertTrue(row["complete"])
        self.assertAlmostEqual(row["output_tokens_per_s_ratio_r2"], 1.1)
        self.assertAlmostEqual(row["output_tokens_per_s_ratio_median"], 1.1)
        self.assertAlmostEqual(row["output_tokens_per_s_ratio_min"], 0.9)
        self.assertAlmostEqual(row["e2e_ms_p95_ratio_max"], 100 / 90)
        row = paired_ratios(cells[:-1], expected_repetitions=3)[0]
        self.assertEqual((row["matched_repetitions"], row["complete"]), (2, False))
        self.assertNotIn("output_tokens_per_s_ratio_r3", row)

    def test_mixed_phases_and_duplicate_ids_are_rejected(self):
        row = request("second")
        row["phase"] = "warmup"
        for rows in ([request(), row], [request(), request()]):
            with self.assertRaises(ValueError):
                summarize(rows, expected_requests=2)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.input = self.parent / "campaign"
        self.output = self.parent / "report"
        make_example(
            self.input, load_config(ROOT / "configs/benchmark.toml"),
            read_json(ROOT / "configs/sources.lock.json"),
            read_json(ROOT / "configs/engines.lock.json"),
        )

    def test_synthetic_analysis_is_explicit_and_regenerates_three_figures(self):
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            analyze(self.input, self.output)
        claims = analyze(self.input, self.output, allow_synthetic=True)
        self.assertFalse(claims["publishable_measurements"])
        self.assertEqual(claims["failed_or_missing_attempts"], 0)
        self.assertEqual(len(claims["paired_ratios"]), 6)
        self.assertTrue(all(r["complete"] and r["matched_repetitions"] == 3 for r in claims["paired_ratios"]))
        self.assertTrue(all(r["output_tokens_per_s_ratio_median"] < 1 for r in claims["paired_ratios"]))
        self.assertTrue((self.output / "paired-ratios.csv").is_file())
        self.assertIn("SGLang ÷ vLLM", (self.output / "tables.md").read_text(encoding="utf-8"))
        figures = list((self.output / "figures").glob("*.svg"))
        self.assertEqual(len(figures), 3)
        for path in figures:
            ET.parse(path)
            self.assertIn("SYNTHETIC FIXTURE", path.read_text())
        second = self.parent / "again"
        analyze(self.input, second, allow_synthetic=True)
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (second / path.relative_to(self.output)).read_bytes())

    def test_mixed_provenance_is_rejected_even_with_valid_checksums(self):
        path = next(self.input.glob("runs/**/cells/*/manifest.json"))
        metadata = read_json(path)
        for field in ("protocol_sha256", "harness_sha256"):
            with self.subTest(field=field):
                write_json(path, {**metadata, field: "0" * 64})
                reseal(path.parent)
                reseal(self.input)
                with self.assertRaisesRegex(ValueError, "Mixed campaign provenance"):
                    analyze(self.input, self.output, allow_synthetic=True)

    def test_summary_must_match_raw_records(self):
        path = next(self.input.glob("runs/**/cells/*/summary.json"))
        summary = read_json(path)
        summary["output_tokens_per_s"] *= 2
        write_json(path, summary)
        reseal(path.parent)
        reseal(self.input)
        with self.assertRaisesRegex(ValueError, "summary"):
            analyze(self.input, self.output, allow_synthetic=True)

    def test_synthetic_bundle_cannot_be_promoted_to_measured(self):
        path = self.input / "campaign.json"
        manifest = read_json(path)
        manifest["evidence_kind"] = "measurement"
        write_json(path, manifest)
        reseal(self.input)
        with self.assertRaisesRegex(ValueError, "Unqualified measurement"):
            analyze(self.input, self.output)

    def test_failure_is_visible_and_group_not_complete(self):
        path = next(self.input.glob("runs/**/cells/*/requests.jsonl.gz"))
        records = read_jsonl(path)
        records[0].update(status="failed", error_ns=records[0]["done_ns"], done_ns=None)
        write_jsonl(path, records)
        write_json(path.parent / "summary.json", summarize(records, expected_requests=8))
        write_json(path.parent / "status.json", {"status": "failed"})
        reseal(path.parent)
        reseal(self.input)
        result = analyze(self.input, self.output, allow_synthetic=True)
        self.assertEqual(result["failed_or_missing_attempts"], 1)
        self.assertEqual(sum(not g["complete"] for g in result["groups"]), 1)

    def test_checksums_reject_tampering_and_unlisted_files(self):
        file = self.input / "protocol.json"
        original = file.read_bytes()
        file.write_bytes(original + b"\n")
        with self.assertRaises(ValueError):
            verify(self.input)
        file.write_bytes(original)
        (self.input / "extra.txt").write_text("unexpected")
        with self.assertRaises(ValueError):
            verify(self.input)

    def test_bundle_is_deterministic_and_requires_synthetic_opt_in(self):
        with self.assertRaises(ValueError):
            bundle(self.input, self.parent / "blocked.tar.gz")
        a = bundle(self.input, self.parent / "a.tar.gz", allow_synthetic=True)
        b = bundle(self.input, self.parent / "b.tar.gz", allow_synthetic=True)
        self.assertEqual(a["sha256"], b["sha256"])
        with self.assertRaises(ValueError):
            analyze(self.input, self.input / "report", allow_synthetic=True)


class PilotAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "pilot"
        make_example(
            self.input, load_config(ROOT / "configs/benchmark.toml"),
            read_json(ROOT / "configs/sources.lock.json"),
            read_json(ROOT / "configs/engines.lock.json"), phase="pilot",
        )
        self.ledger = self.input / "budget-ledger.json"

    def test_pilot_analysis_and_forecast_preserve_synthetic_labels(self):
        from inference_bench.planning import forecast_campaign
        manifest, config, cells = load_campaign(self.input, allow_synthetic=True)
        self.assertEqual(len(cells), 24)
        self.assertEqual(sum(c["expected_requests"] for c in cells), 768)
        self.assertTrue(all(c["valid"] for c in cells))
        claims = analyze(self.input, self.root / "report", allow_synthetic=True)
        self.assertFalse(claims["publishable_measurements"])
        self.assertTrue(all(g["complete"] and g["expected_repetitions"] == 1 for g in claims["groups"]))
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            forecast_campaign(self.input, self.ledger)
        report = forecast_campaign(self.input, self.ledger, allow_synthetic=True)
        self.assertEqual(report["evidence_kind"], "synthetic")
        self.assertTrue(report["advisory_only"])
        self.assertEqual(report["recommended_protocol"], "full")
        self.assertEqual(report["basis"]["observed_startup_seconds"], "1")
        self.assertAlmostEqual(float(report["basis"]["observed_noncell_overhead_seconds"]), 2)
        self.assertEqual([c["cells"] for c in report["candidates"]], [72, 54, 36, 36])
        # The slower engine's durations determine both engines' six formal blocks.
        from decimal import Decimal
        import math
        expected = Decimal(211) + Decimal(2) * 3408 / 720
        for cell in cells:
            if cell["engine"] == "sglang":
                from inference_bench.analysis import load_cell_summary
                warmup = load_cell_summary(self.input / cell["artifact"], config, phase="pilot", warmup=True)
                expected += Decimal(str(warmup["wall_time_s"]))
                expected += Decimal(str(cell["wall_time_s"])) * 256 / cell["expected_requests"]
        self.assertEqual(report["candidates"][0]["total_seconds"], math.ceil(expected * Decimal("1.25")) * 6)
        self.assertEqual(manifest["phase"], "pilot")

    def test_forecast_requires_real_ledger_state_and_refuses_mid_campaign_resize(self):
        from inference_bench.budget import Ledger
        from inference_bench.planning import forecast_campaign
        with self.assertRaisesRegex(ValueError, "existing project ledger"):
            forecast_campaign(self.input, self.root / "missing.json", allow_synthetic=True)
        manifest, config, _ = load_campaign(self.input, allow_synthetic=True)
        ledger_path = self.root / "working-ledger.json"
        ledger_path.write_bytes(self.ledger.read_bytes())
        ledger = Ledger(ledger_path, config)
        reservation = ledger.reserve("pilot", 60, now=0)
        report = forecast_campaign(self.input, ledger_path, allow_synthetic=True)
        self.assertIsNone(report["recommended_protocol"])
        self.assertIn("Outstanding reservations", report["blockers"][0])
        ledger.settle(reservation["id"], 10, confirmed_stopped=True)
        reservation = ledger.reserve("formal", 60)
        ledger.settle(reservation["id"], 0, confirmed_stopped=True)
        report = forecast_campaign(self.input, ledger_path, allow_synthetic=True)
        self.assertIsNone(report["recommended_protocol"])
        self.assertTrue(any("cannot be resized" in reason for reason in report["blockers"]))

    def test_forecast_applies_reconciled_billing_rate_to_candidate_costs(self):
        from decimal import Decimal
        from inference_bench.budget import Ledger
        from inference_bench.planning import forecast_campaign
        _, config, _ = load_campaign(self.input, allow_synthetic=True)
        ledger_path = self.root / "billed-ledger.json"
        ledger_path.write_bytes(self.ledger.read_bytes())
        ledger = Ledger(ledger_path, config)
        lease = ledger.reserve("pilot", 1000, now=0)
        ledger.settle(lease["id"], 1000, confirmed_stopped=True)
        billed = config.rate_per_second * 1000 * 2
        ledger.record_billing(1500, str(billed), "0", "c" * 64, now=2000)
        report = forecast_campaign(self.input, ledger_path, allow_synthetic=True)
        self.assertEqual(report["basis"]["billing_rate_multiplier"], "2")
        for candidate in report["candidates"]:
            self.assertEqual(
                Decimal(candidate["projected_container_usd"]),
                2 * Decimal(candidate["nominal_container_usd"]),
            )

    def test_forecast_rejects_missing_warmup_and_inconsistent_block_timing(self):
        from inference_bench.planning import forecast_campaign
        timing_path = next(self.input.glob("runs/**/timing.json"))
        timing = read_json(timing_path)
        write_json(timing_path, {**timing, "finished_ns": timing["ready_ns"]})
        reseal(timing_path.parent)
        reseal(self.input)
        with self.assertRaisesRegex(ValueError, "windows exceed"):
            forecast_campaign(self.input, self.ledger, allow_synthetic=True)
        write_json(timing_path, timing)
        warmup_path = next(timing_path.parent.glob("cells/*/warmup.jsonl.gz"))
        warmup_path.unlink()
        reseal(warmup_path.parent)
        reseal(timing_path.parent)
        reseal(self.input)
        with self.assertRaises(OSError):
            forecast_campaign(self.input, self.ledger, allow_synthetic=True)

    def test_forecast_does_not_guess_missing_concurrency_observations(self):
        import shutil
        from inference_bench.planning import forecast_campaign
        missing = self.input / "runs/vllm/r1/attempt-1/cells/short-c32"
        block = missing.parent.parent
        shutil.rmtree(missing)
        reseal(block)
        reseal(self.input)
        with self.assertRaisesRegex(ValueError, "coverage must be complete"):
            forecast_campaign(self.input, self.ledger, allow_synthetic=True)
