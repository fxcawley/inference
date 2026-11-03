"""Local commands do not contact Modal or Hugging Face."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .config import load_config
from .provenance import git_identity, measurement_errors, qualification_errors, read_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="inference-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "validate", "commands", "example", "prepare-prompts"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/benchmark.toml"))
        if name == "plan":
            command.add_argument("--json", action="store_true")
            command.add_argument("--phase", choices=("measurement", "pilot"), default="measurement")
        elif name == "validate":
            command.add_argument("--require-ready", action="store_true")
        elif name == "commands":
            command.add_argument("--engine", choices=("vllm", "sglang"), required=True)
            command.add_argument("--model-path", required=True)
        elif name == "example":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--phase", choices=("measurement", "pilot"), default="measurement")
        elif name == "prepare-prompts":
            command.add_argument("--model-dir", type=Path, required=True)
            command.add_argument("--corpus-jsonl", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
    verify_parser = commands.add_parser("verify-artifacts")
    verify_parser.add_argument("--input", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--input", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--allow-synthetic", action="store_true")
    inspect_parser = commands.add_parser("inspect-model")
    inspect_parser.add_argument("--model-dir", type=Path, required=True)
    bundle_parser = commands.add_parser("bundle")
    bundle_parser.add_argument("--input", type=Path, required=True)
    bundle_parser.add_argument("--output", type=Path, required=True)
    bundle_parser.add_argument("--allow-synthetic", action="store_true")
    for name in ("forecast", "resume-plan"):
        planning_parser = commands.add_parser(name)
        planning_parser.add_argument("--input", type=Path, required=True)
        planning_parser.add_argument("--ledger", type=Path, required=True)
        planning_parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-artifacts":
            from .artifacts import verify
            result = verify(args.input)
            print(f'Verified {len(result["files"])} artifact files.')
            return 0
        if args.command == "analyze":
            from .analysis import analyze
            result = analyze(args.input, args.output, allow_synthetic=args.allow_synthetic)
            print(f'Report written to {args.output}; evidence kind: {result["evidence_kind"]}.')
            return 0
        if args.command == "inspect-model":
            from .preparation import inspect_model
            print(json.dumps(inspect_model(args.model_dir), indent=2))
            return 0
        if args.command == "bundle":
            from .artifacts import bundle
            result = bundle(args.input, args.output, allow_synthetic=args.allow_synthetic)
            print(f'Wrote {result["file"]}; SHA-256 {result["sha256"]}')
            return 0
        if args.command in ("forecast", "resume-plan"):
            from .planning import forecast_campaign, resume_plan
            plan = forecast_campaign if args.command == "forecast" else resume_plan
            result = plan(args.input, args.ledger, allow_synthetic=args.allow_synthetic)
            print(json.dumps(result, indent=2))
            return 0
        config = load_config(args.config)
        if args.command == "prepare-prompts":
            from .preparation import prepare_local
            result = prepare_local(config, args.model_dir, args.corpus_jsonl, args.output)
            print(f'Prepared {result["count"]} prompts. Source/license qualification is separate.')
            return 0
        sources = read_json(args.config.parent / "sources.lock.json")
        engines = read_json(args.config.parent / "engines.lock.json")
        problems = qualification_errors(sources, engines)
        if args.command == "plan":
            result = config.describe(phase=args.phase)
            result["qualification_pending"] = problems
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(
                    f'{result["cells"]} cells; {result["measured_requests"]:,} {args.phase} requests\n'
                    f'{result["warmup_requests"]:,} separate warmup requests\n'
                    f'{result["input_tokens"]:,} input tokens; {result["output_tokens"]:,} output tokens\n'
                    f'Nominal container rate: ${float(result["container_usd_per_hour"]):.6f}/hour\n'
                    f'Project limit: ${config.data["budget"]["total_usd"]:.2f} gross\n'
                    f'Config SHA-256: {config.fingerprint}\n'
                    'Runtime forecast: pending pilot; no cloud calls were made.'
                )
                for problem in problems:
                    print(f"  Pending: {problem}")
        elif args.command == "validate":
            if args.require_ready:
                problems = measurement_errors(
                    config.data, sources, engines, git_identity(args.config.parent.parent),
                )
            if args.require_ready and problems:
                raise ValueError("Not ready for formal runs: " + "; ".join(problems))
            print(f"Configuration is valid; {len(problems)} qualification prerequisites remain.")
        elif args.command == "commands":
            from .engines import command
            print(shlex.join(command(args.engine, args.model_path, config)))
            print("Candidate argv only; installed flags and runtime behavior remain unqualified.", file=sys.stderr)
        elif args.command == "example":
            from .examples import make_example
            make_example(args.output, config, sources, engines, phase=args.phase)
            print(f"SYNTHETIC fixture written to {args.output}. No model or GPU was used.")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
