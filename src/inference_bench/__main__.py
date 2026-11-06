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
    qualify_parser = commands.add_parser("qualify")
    qualify_parser.add_argument("--input", type=Path, required=True)
    qualify_parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    qualify_parser.add_argument("--output", type=Path, required=True)
    qualify_parser.add_argument("--preflight", type=Path)
    qualify_parser.add_argument("--accept", action="append", default=[], metavar="CHECK=REASON")
    qualify_parser.add_argument("--allow-synthetic", action="store_true")
    for name in ("ledger", "record-billing"):
        ledger_parser = commands.add_parser(name)
        ledger_parser.add_argument("--config", type=Path, default=Path("configs/benchmark.toml"))
        ledger_parser.add_argument("--ledger", type=Path, required=True)
        if name == "record-billing":
            ledger_parser.add_argument("--through", required=True, help="UTC time, e.g. 2026-11-05T18:00:00Z")
            ledger_parser.add_argument("--gpu-usd", required=True)
            ledger_parser.add_argument("--other-usd", required=True)
            ledger_parser.add_argument("--source", type=Path, required=True)
            ledger_parser.add_argument("--note", default="")
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
        if args.command == "qualify":
            from .artifacts import write_json
            from .provenance import digest_file
            from .qualification import qualify
            accepted = {}
            for item in args.accept:
                name, separator, reason = item.partition("=")
                if not separator or name in accepted:
                    raise ValueError("Use --accept CHECK=REASON once per check")
                accepted[name] = reason
            report = qualify(
                args.input, args.engine, preflight=args.preflight, accepted=accepted,
                allow_synthetic=args.allow_synthetic,
            )
            if args.output.exists():
                raise ValueError("Refusing to replace an existing qualification report")
            write_json(args.output, report)
            print(json.dumps({
                "report": str(args.output), "status": report["status"],
                "evidence_kind": report["evidence_kind"], "report_sha256": digest_file(args.output),
                "unresolved": [c["name"] for c in report["checks"] if c["status"] not in ("passed", "accepted")],
            }, indent=2))
            if report["status"] != "passed":
                print("Qualification did not pass; the engine lock must remain pending.", file=sys.stderr)
                return 3
            return 0
        config = load_config(args.config)
        if args.command in ("ledger", "record-billing"):
            from .budget import Ledger
            ledger = Ledger(args.ledger, config)
            if args.command == "record-billing":
                from datetime import datetime
                from .provenance import digest_file
                through = datetime.fromisoformat(args.through)
                if through.tzinfo is None:
                    raise ValueError("Billing times need an explicit time zone")
                if not args.ledger.is_file():
                    raise ValueError("An existing project ledger is required")
                ledger.record_billing(
                    through.timestamp(), args.gpu_usd, args.other_usd, digest_file(args.source),
                    note=args.note,
                )
            elif not args.ledger.is_file():
                raise ValueError("An existing project ledger is required; a missing ledger is not zero usage")
            print(json.dumps(ledger.summary(), indent=2))
            return 0
        if args.command == "prepare-prompts":
            from .preparation import prepare_local
            result = prepare_local(config, args.model_dir, args.corpus_jsonl, args.output)
            print(f'Prepared {result["count"]} prompts. Source/license qualification is separate.')
            return 0
        sources = read_json(args.config.parent / "sources.lock.json")
        engines = read_json(args.config.parent / "engines.lock.json")
        problems = qualification_errors(sources, engines, args.config.parent)
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
                    args.config.parent,
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
