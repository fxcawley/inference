"""Offline validation, run-level aggregation, CSV tables, and standalone SVGs."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from xml.sax.saxutils import escape

from .artifacts import IDENTITY_FIELDS, atomic_write, read_jsonl, verify, write_json
from .config import Config
from .metrics import AGGREGATE_METRICS, aggregate, summarize
from .provenance import digest_json, is_hash, measurement_errors, read_json


def _cell_key(row: dict) -> tuple:
    return (row["engine"], row["profile"], row["concurrency"], row["repetition"])


def load_campaign(root: Path, *, allow_synthetic: bool = False) -> tuple[dict, Config, list[dict]]:
    verify(root)
    manifest = read_json(root / "campaign.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported campaign schema")
    kind = manifest.get("evidence_kind")
    if kind not in ("measurement", "pilot", "synthetic"):
        raise ValueError("Unknown campaign evidence kind")
    if kind == "synthetic" and not allow_synthetic:
        raise ValueError("Synthetic fixtures require --allow-synthetic; they are not measurements")
    config = Config(read_json(root / "protocol.json"))
    sources = read_json(root / "sources.lock.json")
    engines = read_json(root / "engines.lock.json")
    if not is_hash(manifest.get("harness", {}).get("source_sha256")):
        raise ValueError("Campaign lacks its frozen harness source identity")
    if kind == "measurement":
        problems = measurement_errors(config.data, sources, engines, manifest.get("harness", {}))
        if problems:
            raise ValueError("Unqualified measurement bundle: " + "; ".join(problems))
    identities = {
        "config_sha256": config.fingerprint,
        "sources_sha256": digest_json(sources),
        "engines_sha256": digest_json(engines),
        "harness_sha256": manifest["harness"]["source_sha256"],
    }
    if any(manifest.get(key) != value for key, value in identities.items()):
        raise ValueError("Campaign source, image, or protocol identity mismatch")
    if manifest.get("protocol_sha256") != digest_json(identities):
        raise ValueError("Frozen protocol identity mismatch")
    schedule = manifest["schedule"]
    planned = {_cell_key(c): c for c in schedule}
    if len(planned) != len(schedule):
        raise ValueError("Duplicate scheduled cell")
    if planned != {_cell_key(c.to_dict()): c.to_dict() for c in config.cells()}:
        raise ValueError("Campaign schedule differs from its frozen configuration")
    cells = []
    found = set()
    phase = "pilot" if kind == "pilot" else "measurement"
    for path in sorted(root.glob("runs/**/cells/*/manifest.json")):
        metadata = read_json(path)
        if any(metadata.get(key) != manifest[key] for key in IDENTITY_FIELDS):
            raise ValueError(f"Mixed campaign provenance in {path.relative_to(root)}")
        key = _cell_key(metadata)
        if key not in planned:
            raise ValueError("Unscheduled cell in result bundle")
        if type(metadata.get("attempt")) is not int or metadata["attempt"] not in (1, 2):
            raise ValueError("Unexpected replacement-attempt number")
        found.add(key)
        summary = dict.fromkeys(AGGREGATE_METRICS)
        summary.update(valid=False, expected_requests=config.experiment["requests_per_cell"])
        status = read_json(path.parent / "status.json").get("status")
        if (path.parent / "COMPLETE.json").is_file() and (path.parent / "requests.jsonl.gz").is_file():
            verify(path.parent)
            records = read_jsonl(path.parent / "requests.jsonl.gz")
            shape = config.profiles[metadata["profile"]]
            for row in records:
                if any(row.get(k) != metadata[k] for k in (*IDENTITY_FIELDS, *planned[key], "attempt")):
                    raise ValueError("Request identity differs from its cell")
                if row["phase"] != phase:
                    raise ValueError("Mixed warmup/pilot/measured records")
                if (
                    row["expected_input_tokens"] != shape["input_tokens"]
                    or row["requested_output_tokens"] != shape["output_tokens"]
                ):
                    raise ValueError("Request shape differs from the frozen profile")
            summary = summarize(records, expected_requests=config.experiment["requests_per_cell"])
            if read_json(path.parent / "summary.json") != summary:
                raise ValueError("Stored summary disagrees with raw request records")
            if summary["valid"] != (status == "complete"):
                raise ValueError("Cell status disagrees with measured validity")
        elif status == "complete":
            raise ValueError("A completed cell lacks committed request evidence")
        cells.append({
            **planned[key], "attempt": metadata["attempt"], "status": status,
            **summary, "artifact": path.parent.relative_to(root).as_posix(),
        })
    for key in sorted(set(planned) - found):
        cells.append({
            **planned[key], "attempt": 0, "status": "missing", "valid": False,
            **dict.fromkeys(AGGREGATE_METRICS),
            "expected_requests": config.experiment["requests_per_cell"], "artifact": None,
        })
    return manifest, config, cells


def _csv(path: Path, rows: list[dict]) -> None:
    stream = io.StringIO(newline="")
    fields = sorted({key for row in rows for key in row}) or [
        "engine", "profile", "concurrency", "repetition", "attempt", "status", "valid", "artifact",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, stream.getvalue().encode("utf-8"))


def _svg(
    groups: list[dict], profiles: list[str], *, title: str, x_metric: str,
    y_metrics: tuple[str, ...], y_label: str, annotation: str,
) -> str:
    width, height = max(1, len(profiles)) * 360, 440
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(annotation)}. Points are medians; bars show run minimum and maximum.</desc>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        '<g font-family="Arial, sans-serif" font-size="12" fill="#222">',
        f'<text x="20" y="26" font-size="18">{escape(title)}</text>',
        f'<text x="20" y="47">{escape(annotation)}</text>',
        '<text x="20" y="68">Blue: vLLM · Red: SGLang · bars: run min–max'
        + (" · solid: p50 · dashed: p95" if len(y_metrics) == 2 else "") + "</text>",
    ]
    colors = {"vllm": "#1769aa", "sglang": "#b72d43"}
    for index, profile in enumerate(profiles):
        left, top, plot_w, plot_h = index * 360 + 70, 112, 260, 240
        rows = [g for g in groups if g["profile"] == profile and g["complete"]]
        points = [
            (row[x_metric] if x_metric == "concurrency" else row[x_metric + "_max"], row[metric + "_max"])
            for row in rows for metric in y_metrics
            if row[x_metric] is not None and row[metric + "_max"] is not None
        ]
        xmax = max((p[0] for p in points), default=1) * 1.1 or 1
        ymax = max((p[1] for p in points), default=1) * 1.15 or 1

        def sx(value):
            return left + value / xmax * plot_w

        def sy(value):
            return top + plot_h - value / ymax * plot_h

        parts.append(f'<text x="{left}" y="{top - 17}" font-size="15">{escape(profile)}</text>')
        for step in range(5):
            x, y = left + step * plot_w / 4, top + plot_h - step * plot_h / 4
            parts.extend([
                f'<path d="M {left} {y:.2f} H {left + plot_w}" stroke="#ddd" fill="none"/>',
                f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{step * ymax / 4:.3g}</text>',
                f'<text x="{x:.2f}" y="{top + plot_h + 20}" text-anchor="middle">{step * xmax / 4:.3g}</text>',
            ])
        parts.append(
            f'<path d="M {left} {top} V {top + plot_h} H {left + plot_w}" stroke="#444" fill="none"/>'
        )
        if not points:
            parts.append(f'<text x="{left + 10}" y="{top + 120}">No complete measurement groups</text>')
        for engine, color in colors.items():
            series = sorted((r for r in rows if r["engine"] == engine), key=lambda r: r["concurrency"])
            for metric_index, metric in enumerate(y_metrics):
                valid = [r for r in series if r[metric] is not None]
                coordinates = " ".join(f"{sx(r[x_metric]):.2f},{sy(r[metric]):.2f}" for r in valid)
                dash = ' stroke-dasharray="5 4"' if metric_index else ""
                parts.append(
                    f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"{dash}/>'
                )
                for row in valid:
                    x, y = sx(row[x_metric]), sy(row[metric])
                    lo, hi = sy(row[metric + "_min"]), sy(row[metric + "_max"])
                    parts.extend([
                        f'<path d="M {x:.2f} {lo:.2f} V {hi:.2f} M {x - 3:.2f} {lo:.2f} H {x + 3:.2f}'
                        f' M {x - 3:.2f} {hi:.2f} H {x + 3:.2f}" stroke="{color}" fill="none"/>',
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>',
                    ])
                    if x_metric != "concurrency":
                        xlo, xhi = sx(row[x_metric + "_min"]), sx(row[x_metric + "_max"])
                        parts.append(
                            f'<path d="M {xlo:.2f} {y:.2f} H {xhi:.2f}" stroke="{color}" fill="none"/>'
                        )
                        parts.append(
                            f'<text x="{x + 5:.2f}" y="{y + (17 if engine == "vllm" else -8):.2f}"'
                            f' fill="{color}">C={row["concurrency"]}</text>'
                        )
        x_label = "Concurrency" if x_metric == "concurrency" else "Output tokens / second"
        parts.extend([
            f'<text x="{left + plot_w / 2}" y="{top + plot_h + 42}" text-anchor="middle">{x_label}</text>',
            f'<text transform="translate({left - 51} {top + plot_h / 2}) rotate(-90)" text-anchor="middle">'
            f'{escape(y_label)}</text>',
        ])
    parts.extend([
        '<text x="20" y="426">Only complete repetition groups are plotted. Missing/invalid attempts remain in failures.csv.</text>',
        "</g></svg>\n",
    ])
    return "".join(parts)


def analyze(root: Path, output: Path, *, allow_synthetic: bool = False) -> dict:
    root, output = Path(root), Path(output)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Analysis output must be outside the sealed input bundle")
    manifest, config, cells = load_campaign(root, allow_synthetic=allow_synthetic)
    groups = aggregate(cells, expected_repetitions=config.experiment["repetitions"])
    for group in groups:
        throughput = group["output_tokens_per_s"]
        group["gpu_usd_per_million_output_tokens"] = (
            1_000_000 * config.data["pricing"]["gpu_per_second"] / throughput if throughput else None
        )
        group["container_usd_per_million_output_tokens"] = (
            1_000_000 * float(config.rate_per_second) / throughput if throughput else None
        )
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "cells.csv", cells)
    _csv(output / "aggregate.csv", groups)
    failures = [c for c in cells if not c["valid"]]
    _csv(output / "failures.csv", failures)
    kind = manifest["evidence_kind"]
    label = {
        "synthetic": "SYNTHETIC FIXTURE — no model was run",
        "pilot": "PILOT — not the formal comparison",
        "measurement": "Gemma 4 · H200 · BF16",
    }[kind]
    annotation = (
        f'{label} · {config.experiment["requests_per_cell"]} requests/cell'
        f' · {config.experiment["repetitions"]} repetitions'
    )
    figures = (
        ("throughput", "Output throughput", "concurrency", ("output_tokens_per_s",), "Output tokens / second"),
        ("ttft", "Time to first streamed text", "concurrency", ("ttft_ms_p50", "ttft_ms_p95"), "TTFT (ms)"),
        ("latency-throughput", "Completion latency vs throughput", "output_tokens_per_s", ("e2e_ms_p95",), "p95 end-to-end latency (ms)"),
    )
    for filename, title, x, ys, ylabel in figures:
        atomic_write(output / "figures" / f"{filename}.svg", _svg(
            groups, config.profile_names, title=title, x_metric=x, y_metrics=ys,
            y_label=ylabel, annotation=annotation,
        ).encode("utf-8"))
    claims = {
        "schema_version": 1, "campaign_id": manifest["campaign_id"], "evidence_kind": kind,
        "protocol_sha256": manifest["protocol_sha256"],
        "publishable_measurements": kind == "measurement" and all(g["complete"] for g in groups),
        "failed_or_missing_attempts": len(failures), "groups": groups,
        "uncertainty": "median and minimum–maximum of repetition-level metrics; not a confidence interval",
        "observed_project_cost_usd": None,
    }
    write_json(output / "claims.json", claims)
    table = [
        f"# {label}", "",
        "| Engine | Profile | C | Valid repetitions | Output tok/s | p95 TTFT (ms) | p95 E2E (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        def display(key):
            return f"{group[key]:.3f}" if group[key] is not None else "—"
        table.append(
            f'| {group["engine"]} | {group["profile"]} | {group["concurrency"]} | '
            f'{group["valid_repetitions"]} | {display("output_tokens_per_s")} | '
            f'{display("ttft_ms_p95")} | {display("e2e_ms_p95")} |'
        )
    table.extend(["", "Costs in aggregate.csv are derived list-price estimates. Actual billed project cost remains unreconciled.", ""])
    atomic_write(output / "tables.md", "\n".join(table).encode("utf-8"))
    return claims
