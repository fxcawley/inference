import copy
import contextlib
import io
import re
import tempfile
import unittest
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
        tokens, _ = tokenize_corpus([" Public prose. " * 1000], self.tokenizer)
        prompts, manifest = prepare_prompts(self.config, tokens, self.tokenizer)
        again, other_manifest = prepare_prompts(self.config, tokens, self.tokenizer)
        self.assertEqual(prompts, again)
        self.assertEqual(manifest, other_manifest)
        for prompt in prompts:
            self.assertEqual(len(prompt["prompt_ids"]), 128)
            self.assertEqual(prompt["prompt_ids"][-2:], [2, 4])
            self.assertEqual(prompt["prompt_ids"].count(1), 1)
        measured = {p["request_id"] for p in prompts if p["phase"] == "measurement"}
        warmed = {p["request_id"] for p in prompts if p["phase"] == "warmup"}
        self.assertFalse(measured & warmed)
        self.assertEqual(len(measured), 4)

    def test_normalization_and_template_overflow(self):
        self.assertIsNone(normalize_row("  = Heading = "))
        self.assertEqual(normalize_row(" cafe\u0301 \n text "), "café text")
        data = copy.deepcopy(self.config.data)
        data["profiles"]["short"]["input_tokens"] = 2
        with self.assertRaises(ValueError):
            prepare_prompts(Config(data), [123] * 100, self.tokenizer)

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
            self.assertEqual(manifest["count"], 8)


class CLITests(unittest.TestCase):
    def test_plan_and_validation_are_offline_and_readiness_fails_closed(self):
        from inference_bench.artifacts import write_json
        from inference_bench.__main__ import main
        from unittest.mock import patch
        with patch("socket.create_connection", side_effect=AssertionError("No network")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["plan", "--config", str(ROOT / "configs/benchmark.toml")]), 0)
            self.assertIn("72 cells", output.getvalue())
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
