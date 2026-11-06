import copy
import contextlib
import io
import re
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from inference_bench.config import Config, load_config
from inference_bench.provenance import canonical_json, parse_json, qualification_errors, read_json
from inference_bench.preparation import prepare_local
from inference_bench.workloads import normalize_row, prepare_prompts, tokenize_corpus

ROOT = Path(__file__).resolve().parents[1]


def unqualified_locks():
    sources = read_json(ROOT / "configs/sources.lock.json")
    engines = read_json(ROOT / "configs/engines.lock.json")
    sources["status"] = "unresolved"
    sources["prompts"]["manifest_sha256"] = None
    for name in ("vllm", "sglang"):
        engines["engines"][name]["qualification"] = {
            "status": "pending", "report_sha256": None,
        }
    return sources, engines


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/benchmark.toml")

    def test_matrix_and_pairing(self):
        plan = self.config.describe()
        self.assertEqual(plan["cells"], 72)
        self.assertEqual(plan["measured_requests"], 18432)
        self.assertEqual(plan["input_tokens"], 31457280)
        self.assertEqual(plan["output_tokens"], 4718592)
        cells = self.config.cells()
        for rep in (1, 2, 3):
            a = [(c.profile, c.concurrency) for c in cells if c.repetition == rep and c.engine == "vllm"]
            b = [(c.profile, c.concurrency) for c in cells if c.repetition == rep and c.engine == "sglang"]
            self.assertEqual(a, b)

    def test_canonical_json_round_trip_preserves_schedule(self):
        restored = Config(parse_json(canonical_json(self.config.data)))
        self.assertEqual(restored.fingerprint, self.config.fingerprint)
        self.assertEqual(restored.cells(), self.config.cells())
        self.assertEqual(restored.profile_names, ["short", "prefill", "decode"])

    def test_pilot_has_one_paired_repetition_and_bounded_requests(self):
        plan = self.config.describe(phase="pilot")
        self.assertEqual((plan["cells"], plan["measured_requests"], plan["warmup_requests"]), (24, 768, 672))
        schedule = plan["schedule"]
        self.assertEqual({c["repetition"] for c in schedule}, {1})
        for row in schedule:
            self.assertEqual(row["requests"], max(16, 2 * row["concurrency"]))
        a = [(c["profile"], c["concurrency"]) for c in schedule if c["engine"] == "vllm"]
        b = [(c["profile"], c["concurrency"]) for c in schedule if c["engine"] == "sglang"]
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            self.config.cells(phase="unknown")

    def test_invalid_configuration_is_rejected(self):
        cases = [
            ("experiment", "concurrency", [1, 4, 4]),
            ("experiment", "repetitions", True),
            ("hardware", "gpu", "H100"),
            ("hardware", "memory_fraction", float("nan")),
            ("budget", "total_usd", 2),
        ]
        for section, key, value in cases:
            with self.subTest(key=key):
                data = copy.deepcopy(self.config.data)
                data[section][key] = value
                with self.assertRaises(ValueError):
                    Config(data)

    def test_unresolved_sources_cannot_qualify(self):
        errors = qualification_errors(*unqualified_locks())
        self.assertIn("vllm real-model qualification is pending", errors)
        self.assertIn("sglang real-model qualification is pending", errors)
        self.assertIn("prepared prompts are not pinned", errors)


class TestTokenizer:
    """A small explicit template/tokenizer fixture, not Gemma behavior."""
    bos_token_id = 1
    all_special_ids = [1, 2, 3, 4]
    special = {"<BOS>": 1, "<END>": 2, "<USER>": 3, "<ASSISTANT>": 4}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        result = []
        for part in re.split(r"(<BOS>|<END>|<USER>|<ASSISTANT>)", text):
            if part in self.special:
                result.append(self.special[part])
            else:
                result.extend(100 + ord(char) for char in part)
        return result

    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
        }
        return "<BOS><USER>" + messages[0]["content"] + "<END><ASSISTANT>"


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        data = copy.deepcopy(load_config(ROOT / "configs/benchmark.toml").data)
        data["profiles"] = {"short": {"input_tokens": 128, "output_tokens": 3}}
        data["experiment"].update(
            repetitions=1, requests_per_cell=4, concurrency=[1, 4],
            warmup_min=2, warmup_waves=1,
        )
        self.config = Config(data)
        self.tokenizer = TestTokenizer()

    def test_exact_lengths_preserve_template_and_separate_phases(self):
        from inference_bench.qualification import PROBES, select_probes
        tokens, _ = tokenize_corpus([" Public prose. " * 1000], self.tokenizer)
        prompts, manifest = prepare_prompts(self.config, tokens, self.tokenizer, stop_token_ids=[2])
        again, other_manifest = prepare_prompts(self.config, tokens, self.tokenizer, stop_token_ids=[2])
        self.assertEqual(prompts, again)
        self.assertEqual(manifest, other_manifest)
        for prompt in prompts:
            if prompt["phase"] != "conformance":
                self.assertEqual(len(prompt["prompt_ids"]), 128)
            self.assertEqual(prompt["prompt_ids"][-2:], [2, 4])
            self.assertEqual(prompt["prompt_ids"].count(1), 1)
        probes = select_probes(prompts)
        self.assertEqual([p["name"] for p, _ in probes], [p["name"] for p in PROBES])
        self.assertTrue(all(prompt["stop_token_ids"] == [2] for _, prompt in probes))
        summary = dict((p["name"], prompt) for p, prompt in probes)["passage_summary"]
        self.assertEqual(len(summary["prompt_ids"]), 128)
        self.assertEqual(manifest["stop_token_ids"], [2])
        with self.assertRaises(ValueError):
            prepare_prompts(self.config, tokens, self.tokenizer, stop_token_ids=[])
        measured = {p["request_id"] for p in prompts if p["phase"] == "measurement"}
        warmed = {p["request_id"] for p in prompts if p["phase"] == "warmup"}
        pilot = {p["request_id"] for p in prompts if p["phase"] == "pilot"}
        pilot_warmup = {p["request_id"] for p in prompts if p["phase"] == "pilot_warmup"}
        self.assertEqual(len(measured | warmed | pilot | pilot_warmup) + len(probes), len(prompts))
        self.assertEqual(len(measured), 4)
        self.assertEqual(len(pilot), 16)

    def test_normalization_and_template_overflow(self):
        self.assertIsNone(normalize_row("  = Heading = "))
        self.assertEqual(normalize_row(" cafe\u0301 \n text "), "café text")
        data = copy.deepcopy(self.config.data)
        data["profiles"]["short"]["input_tokens"] = 2
        with self.assertRaises(ValueError):
            prepare_prompts(Config(data), [123] * 100, self.tokenizer, stop_token_ids=[2])

    def test_cpu_preparation_can_be_verified_without_transformers(self):
        from inference_bench.artifacts import verify, write_jsonl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            write_jsonl(corpus, [{"text": "Public prose. " * 1000}])
            manifest = prepare_local(
                self.config, root / "unused-model", corpus, root / "prepared",
                tokenizer=self.tokenizer,
            )
            verify(root / "prepared")
            self.assertEqual(manifest["count"], 33)
            self.assertEqual((manifest["stop_token_ids"], manifest["stop_token_sources"]), ([2], ["tokenizer"]))
            self.assertTrue((root / "prepared/pilot.jsonl.gz").is_file())
            self.assertTrue((root / "prepared/pilot-warmup.jsonl.gz").is_file())
            self.assertTrue((root / "prepared/conformance.jsonl.gz").is_file())

    def test_probe_stop_ids_come_from_the_pinned_generation_config(self):
        from inference_bench.artifacts import write_json
        from inference_bench.preparation import stop_token_ids
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "generation_config.json", {"eos_token_id": [106, 1]})
            self.assertEqual(stop_token_ids(root, self.tokenizer), ([1, 2, 106], ["generation_config.json", "tokenizer"]))
            write_json(root / "generation_config.json", {"eos_token_id": "1"})
            with self.assertRaises(ValueError):
                stop_token_ids(root, self.tokenizer)
            (root / "generation_config.json").unlink()

            class NoEOS:
                eos_token_id = None

            with self.assertRaises(ValueError):
                stop_token_ids(root, NoEOS())


class ForecastTests(unittest.TestCase):
    def test_formal_gate_accepts_only_the_declared_compact_variants(self):
        from inference_bench.planning import protocol_candidates
        from inference_bench.provenance import measurement_errors
        config = load_config(ROOT / "configs/benchmark.toml")
        sources, engines = unqualified_locks()
        for _, candidate in protocol_candidates(config):
            errors = measurement_errors(candidate.data, sources, engines, {})
            self.assertFalse(any("reduction order" in error for error in errors))
            self.assertTrue(any("qualification is pending" in error for error in errors))
        for profiles, concurrency, count in (
            (("short", "prefill", "decode"), [1, 4, 16, 32], 200),
            (("short", "prefill"), [1, 4, 16, 32], 256),
            (("short", "prefill", "decode"), [1, 4, 16], 200),
        ):
            data = copy.deepcopy(config.data)
            data["profiles"] = {p: data["profiles"][p] for p in profiles}
            data["experiment"].update(concurrency=concurrency, requests_per_cell=count)
            errors = measurement_errors(data, sources, engines, {})
            self.assertTrue(any("reduction order" in error for error in errors))

    def test_forecast_includes_overheads_and_uses_ordered_symmetric_reductions(self):
        from inference_bench.planning import estimate_protocols
        config = load_config(ROOT / "configs/benchmark.toml")
        for seconds_per_request, expected in (
            ("0.5", "full"), ("1", "without-c32"), ("1.5", "two-profiles"),
            ("2", "two-profiles-200"), ("3", None),
        ):
            with self.subTest(rate=seconds_per_request):
                observations = {
                    (profile, c): {"seconds_per_request": Decimal(seconds_per_request), "warmup_s": Decimal(5)}
                    for profile in config.profiles for c in config.experiment["concurrency"]
                }
                estimates = estimate_protocols(config, observations, Decimal(100), Decimal(20))
                self.assertEqual([e["cells"] for e in estimates], [72, 54, 36, 36])
                choice = next((e["name"] for e in estimates if e["total_seconds"] <= 24000), None)
                self.assertEqual(choice, expected)
                for estimate in estimates:
                    candidate = Config(estimate["config"])
                    self.assertEqual(candidate.experiment["repetitions"], 3)
                    self.assertEqual({c.engine for c in candidate.cells()}, {"vllm", "sglang"})
                    self.assertEqual(estimate["total_seconds"], 6 * estimate["block_seconds"])
                    self.assertEqual(
                        Decimal(estimate["nominal_container_usd"]),
                        config.rate_per_second * estimate["total_seconds"],
                    )
                if seconds_per_request == "0.5":
                    # ceil((100 startup + 20 * (3408 / 720) local overhead + 210 container allowance
                    #       + 12 * (5 warmup + 256 * 0.5 measured)) * 1.25) * 6 blocks.
                    self.assertEqual(estimates[0]["total_seconds"], 15006)


def qualified_fixture(base, config, harness_sha256):
    """Test-only locks and reports; they do not describe any real qualification."""
    from inference_bench.artifacts import write_json
    from inference_bench.engines import engine_settings_sha256
    from inference_bench.provenance import digest_file
    from inference_bench.qualification import REQUIRED_CHECKS
    file = {"config.json": {"sha256": "1" * 64, "bytes": 10}}
    sources = {
        "schema_version": 1, "status": "complete",
        "model": {"repo_id": config.experiment["model"], "revision": "a" * 40, "files": file},
        "dataset": {
            "repo_id": "Salesforce/wikitext", "configuration": "wikitext-2-raw-v1", "split": "train",
            "revision": "b" * 40, "license": "TEST FIXTURE", "files": file,
        },
        "prompts": {"manifest_sha256": "2" * 64, "generator_version": "3"},
    }
    engines = read_json(ROOT / "configs/engines.lock.json")
    engines["client"].update(python_version="3.12.3", requirements_sha256="3" * 64)
    for name, record in engines["engines"].items():
        report = {
            "schema_version": 1, "engine": name, "evidence_kind": "pilot", "status": "passed",
            "bindings": {
                "evidence_kind": "pilot", "image": record["image"], "model_revision": "a" * 40,
                "engine_settings_sha256": engine_settings_sha256(name, config),
                "harness_sha256": harness_sha256,
            },
            "checks": [{"name": check, "status": "passed"} for check in sorted(REQUIRED_CHECKS)],
        }
        path = Path(base) / "qualification" / f"{name}.json"
        write_json(path, report)
        record["qualification"] = {
            "status": "passed", "report": f"qualification/{name}.json", "report_sha256": digest_file(path),
        }
    return sources, engines


class QualificationTests(unittest.TestCase):
    def setUp(self):
        from inference_bench.examples import make_example
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(ROOT / "configs/benchmark.toml")
        self.pilot = self.root / "pilot"
        make_example(
            self.pilot, self.config, read_json(ROOT / "configs/sources.lock.json"),
            read_json(ROOT / "configs/engines.lock.json"), phase="pilot",
        )

    def preflight(self, **changes):
        from inference_bench.artifacts import seal, write_json
        from inference_bench.engines import command
        directory = self.root / f"preflight-{len(list(self.root.glob('preflight-*')))}"
        for name, record in read_json(ROOT / "configs/engines.lock.json")["engines"].items():
            write_json(directory / f"{name}.json", {
                "upstream_image": record["image"], "engine_executable": f"/usr/local/bin/{name}",
                "cli_help": {"returncode": 0}, "flags_not_listed_in_help": [],
                "candidate_argv": command(name, "/models/qualification-pending", self.config), **changes,
            })
        seal(directory)
        return directory

    def test_probe_evaluation_requires_real_stops_exact_lengths_and_answers(self):
        from inference_bench.qualification import PROBE_BY_NAME, evaluate_probe
        def record(name, **changes):
            probe = PROBE_BY_NAME[name]
            return {
                "probe_name": name, "status": "success", "requested_output_tokens": probe["max_tokens"],
                "finish_reason": "length" if probe["ignore_eos"] else "stop",
                "actual_output_tokens": probe["max_tokens"] if probe["ignore_eos"] else 2,
                "response_text": {"arithmetic": "42", "capital": "Paris"}.get(name, "yes"), **changes,
            }
        for name, changes, expected in (
            ("eos_honored", {}, "passed"),
            ("eos_honored", {"finish_reason": "length", "actual_output_tokens": 64}, "failed"),
            ("eos_ignored", {}, "passed"),
            ("eos_ignored", {"finish_reason": "stop", "actual_output_tokens": 2}, "failed"),
            ("arithmetic", {}, "passed"), ("arithmetic", {"response_text": "41"}, "failed"),
            ("capital", {"status": "failed", "error_message": "HTTP 500"}, "failed"),
            ("passage_summary", {"response_text": "the the the the the the the the the the the the"}, "failed"),
            ("passage_summary", {"response_text": "\ufffd" * 50}, "failed"),
        ):
            with self.subTest(name=name, changes=changes):
                self.assertEqual(evaluate_probe(record(name, **changes), PROBE_BY_NAME[name])["status"], expected)
        self.assertEqual(
            evaluate_probe(record("arithmetic", requested_output_tokens=33), PROBE_BY_NAME["arithmetic"])["status"],
            "failed",
        )

    def test_synthetic_pilot_qualification_requires_review_and_never_passes_the_gate(self):
        from inference_bench.artifacts import write_json
        from inference_bench.provenance import digest_file
        from inference_bench.qualification import qualify
        report = qualify(self.pilot, "vllm", allow_synthetic=True)
        self.assertEqual(report["status"], "review_required")
        statuses = {c["name"]: c["status"] for c in report["checks"]}
        self.assertEqual(statuses.pop("cpu_preflight"), "review")
        self.assertEqual(set(statuses.values()), {"passed"})
        self.assertEqual(report["resolved_settings"]["values"]["kv_cache_tokens"], 400000)
        with self.assertRaises(ValueError):
            qualify(self.pilot, "vllm", allow_synthetic=True, accepted={"conformance": "not a review item"})
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            qualify(self.pilot, "vllm")
        report = qualify(self.pilot, "sglang", preflight=self.preflight(), allow_synthetic=True)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["evidence_kind"], "synthetic")
        sources, engines = qualified_fixture(self.root / "configs", self.config, report["bindings"]["harness_sha256"])
        write_json(self.root / "configs/qualification/sglang.json", report)
        engines["engines"]["sglang"]["qualification"]["report_sha256"] = digest_file(
            self.root / "configs/qualification/sglang.json",
        )
        errors = qualification_errors(sources, engines, self.root / "configs")
        self.assertIn("sglang qualification does not use real pilot evidence", errors)

    def test_preflight_and_tampered_conformance_failures_are_reported(self):
        from inference_bench.artifacts import read_jsonl, seal, write_jsonl
        from inference_bench.qualification import qualify
        report = qualify(
            self.pilot, "vllm", preflight=self.preflight(flags_not_listed_in_help=["--disable-log-requests"]),
            allow_synthetic=True,
        )
        problems = next(c for c in report["checks"] if c["name"] == "cpu_preflight")["evidence"]["problems"]
        self.assertEqual(report["status"], "failed")
        self.assertIn("--disable-log-requests", problems[0])
        block = self.pilot / "runs/vllm/r1/attempt-1"
        path = block / "conformance/requests.jsonl.gz"
        records = read_jsonl(path)
        next(r for r in records if r["probe_name"] == "arithmetic")["response_text"] = "41"
        write_jsonl(path, records)
        for directory in (block, self.pilot):
            (directory / "COMPLETE.json").unlink()
            (directory / "CHECKSUMS.json").unlink()
            seal(directory)
        report = qualify(self.pilot, "vllm", preflight=self.preflight(), allow_synthetic=True)
        conformance = next(c for c in report["checks"] if c["name"] == "conformance")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            [p["name"] for p in conformance["evidence"]["probes"] if p["status"] != "passed"], ["arithmetic"],
        )

    def test_gate_verifies_report_contents_and_bindings(self):
        from inference_bench.artifacts import write_json
        from inference_bench.provenance import digest_file, measurement_errors, source_tree_hash
        base = self.root / "configs"
        harness = {"commit": "c" * 40, "dirty": False, "source_sha256": source_tree_hash()}
        sources, engines = qualified_fixture(base, self.config, source_tree_hash())
        self.assertEqual(measurement_errors(self.config.data, sources, engines, harness, base), [])
        self.assertIn("vllm qualification report was not verified", measurement_errors(
            self.config.data, sources, engines, harness,
        ))
        changed = copy.deepcopy(self.config.data)
        changed["hardware"]["memory_fraction"] = 0.8
        self.assertIn(
            "vllm engine settings differ from the qualified settings",
            measurement_errors(changed, sources, engines, harness, base),
        )
        self.assertIn(
            "sglang measurement harness differs from the qualified harness",
            measurement_errors(self.config.data, sources, engines, {**harness, "source_sha256": "d" * 64}, base),
        )
        path = base / "qualification/vllm.json"
        report = read_json(path)
        for change, message in (
            (lambda r: r["checks"].pop(), "did not pass every required check"),
            (lambda r: r["bindings"].update(image="other@sha256:" + "e" * 64), "different image"),
            (lambda r: r["bindings"].update(model_revision="f" * 40), "different model revision"),
            (lambda r: r.update(evidence_kind="synthetic"), "real pilot evidence"),
        ):
            with self.subTest(message=message):
                edited = copy.deepcopy(report)
                change(edited)
                write_json(path, edited)
                stale = qualification_errors(sources, engines, base)
                self.assertIn("vllm qualification report differs from its locked hash", stale)
                relocked = copy.deepcopy(engines)
                relocked["engines"]["vllm"]["qualification"]["report_sha256"] = digest_file(path)
                self.assertTrue(any(message in e for e in qualification_errors(sources, relocked, base)))
        path.unlink()
        self.assertIn("vllm qualification report is missing", qualification_errors(sources, engines, base))

    def test_formal_campaign_carries_and_rechecks_qualification_reports(self):
        from inference_bench.analysis import load_campaign
        from inference_bench.artifacts import create_campaign, seal
        from inference_bench.provenance import source_tree_hash
        base = self.root / "configs"
        sources, engines = qualified_fixture(base, self.config, source_tree_hash())
        harness = {"commit": "c" * 40, "dirty": False}
        with self.assertRaisesRegex(ValueError, "qualification reports"):
            create_campaign(
                self.root / "missing-base", self.config, sources, engines, campaign_id="formal",
                evidence_kind="measurement", harness=harness,
            )
        root = self.root / "formal"
        create_campaign(
            root, self.config, sources, engines, campaign_id="formal",
            evidence_kind="measurement", harness=harness, qualification_base=base,
        )
        self.assertEqual((root / "qualification/vllm.json").read_bytes(), (base / "qualification/vllm.json").read_bytes())
        seal(root)
        _, _, cells = load_campaign(root)
        self.assertEqual({c["status"] for c in cells}, {"missing"})
        copied = root / "qualification/sglang.json"
        copied.write_bytes(copied.read_bytes().replace(b'"passed"', b'"failed"', 1))
        (root / "COMPLETE.json").unlink()
        (root / "CHECKSUMS.json").unlink()
        seal(root)
        with self.assertRaisesRegex(ValueError, "Unqualified measurement bundle"):
            load_campaign(root)

    def test_cli_writes_reports_but_returns_nonzero_until_qualification_passes(self):
        from inference_bench.__main__ import main
        output = self.root / "reports/vllm.json"
        arguments = ["qualify", "--input", str(self.pilot), "--engine", "vllm", "--allow-synthetic"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main([*arguments, "--output", str(output)]), 3)
            self.assertEqual(read_json(output)["status"], "review_required")
            self.assertEqual(main([*arguments, "--output", str(output)]), 2)
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(main([
                *arguments, "--output", str(self.root / "reports/accepted.json"),
                "--accept", "cpu_preflight=Synthetic fixture exercise",
            ]), 0)
        self.assertEqual(parse_json(printed.getvalue())["evidence_kind"], "synthetic")


class CLITests(unittest.TestCase):
    def test_plan_and_validation_are_offline_and_readiness_fails_closed(self):
        from inference_bench.artifacts import write_json
        from inference_bench.__main__ import main
        from unittest.mock import patch
        with patch("socket.create_connection", side_effect=AssertionError("No network")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["plan", "--config", str(ROOT / "configs/benchmark.toml")]), 0)
            self.assertIn("72 cells", output.getvalue())
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main([
                    "plan", "--phase", "pilot", "--config", str(ROOT / "configs/benchmark.toml"),
                ]), 0)
            self.assertIn("24 cells; 768 pilot requests", output.getvalue())
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "benchmark.toml").write_bytes((ROOT / "configs/benchmark.toml").read_bytes())
                sources, engines = unqualified_locks()
                write_json(root / "sources.lock.json", sources)
                write_json(root / "engines.lock.json", engines)
                with contextlib.redirect_stderr(io.StringIO()) as errors:
                    self.assertEqual(main([
                        "validate", "--config", str(root / "benchmark.toml"), "--require-ready",
                    ]), 2)
            self.assertIn("Not ready", errors.getvalue())
            self.assertIn("source preparation is incomplete", errors.getvalue())
