# Gemma 4 serving benchmark

A planned public comparison of **vLLM and SGLang serving Gemma 4 26B-A4B on Modal**, with a short write-up for [cawley.dev](https://cawley.dev).

**Status: the local harness, pilot planning, cost forecasting, recovery planning, and offline reports are implemented.** CPU tests cover streaming/token accounting, deterministic prompts, process cleanup, persistent budgets, and artifact validation. Engine images, model access, and real GPU behavior remain unqualified; there are no measured engine results yet.

Read the **[full megaplan](docs/megaplan.md)** for the experiment design, implementation work packages, file map, measurement definitions, budget controls, and acceptance criteria.

## Continue on another machine

```bash
git clone https://github.com/fxcawley/inference.git
cd inference
```

Use Python 3.12 on Linux or WSL. Planning and analysis need only the standard library. Streaming HTTP and its tests use HTTPX; cloud and source-preparation dependencies are separate extras.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[client]'
python -m inference_bench plan
python -m unittest discover -s tests -v
```

To inspect the plan without installing anything:

```bash
PYTHONPATH=src python3 -m inference_bench plan
```

Read [AGENTS.md](AGENTS.md), the [current runbook](docs/runbook.md), and the [measurement definitions](docs/methodology.md). The megaplan describes the full project; the runbook distinguishes working commands from integration work still ahead.

## Exercise the complete offline report path

```bash
python -m inference_bench example --output artifacts/example
python -m inference_bench verify-artifacts --input artifacts/example
python -m inference_bench analyze \
  --input artifacts/example --output artifacts/example-report --allow-synthetic
python -m inference_bench bundle \
  --input artifacts/example --output artifacts/example.tar.gz --allow-synthetic
```

This produces a checksummed **synthetic fixture**, three labelled SVGs, CSV tables, and `claims.json`. It does not predict either engine's performance. Analysis requires explicit opt-in for synthetic evidence, and synthetic bundles cannot pass the formal measurement gate.

The core includes:

- Strict configuration and deterministic, paired schedules.
- A bounded pilot with separate inputs: 24 cells, 768 timed requests, and 672 warmup requests.
- Cost forecasts from verified pilot artifacts, including the four agreed full/compact choices.
- Exact-length prompt assembly around a supplied tokenizer's chat template.
- One streaming client with fixed concurrency, absolute request deadlines, authoritative usage checks, and no request retries.
- Pure latency/throughput calculations and aggregation across independent repetitions.
- Durable checksums, completion markers, reproducible bundles, and offline SVG reports.
- Persistent budget reservations, serial-work checks, absolute leases, bounded logs, process-group cleanup, and GPU identity checks.
- Optional source-preparation functions and a Modal **CPU image inspection** entrypoint.

## Continue without network access

Use the existing Python environment; these commands need no package downloads:

```bash
PYTHONPATH=src python -m inference_bench plan --phase pilot
PYTHONPATH=src:tests python -m unittest \
  test_protocol test_metrics test_runner test_client.DecoderTests -v
```

This test selection uses local files, in-memory client doubles, and short-lived owned processes. The full suite above also includes six loopback HTTP tests.

The new planning commands operate on local artifacts and an existing ledger:

```bash
python -m inference_bench forecast --input artifacts/pilot --ledger artifacts/budget-ledger.json
python -m inference_bench resume-plan --input artifacts/campaign --ledger artifacts/budget-ledger.json
```

`forecast` compares the agreed protocol sizes against the remaining budget. `resume-plan` accepts an interrupted campaign, preserves valid cells, and proposes only missing work or the one permitted replacement. Both print advisory JSON; they do not reserve money or launch work. See the [runbook](docs/runbook.md#pilot-planning-and-cost-forecasting) for a synthetic example and the required evidence.

## Agreed first study

| Item | Plan |
|---|---|
| Question | How do output throughput, first-response latency, and completion latency change with concurrency? |
| Model | `google/gemma-4-26B-A4B-it`, official standard BF16 checkpoint, immutable revision to resolve during preparation |
| Candidate engines | vLLM 0.32.0 and SGLang 0.5.22, with official image digests recorded in the plan |
| Hardware | One verified NVIDIA H200 SXM, TP=1 |
| Inputs | A pinned public-text corpus; identical prepared prompt IDs for both engines |
| Load | Three input/output shapes; concurrency 1, 4, 16, and 32; three repetitions per engine |
| Measurement | One common streaming client colocated with the server; prefix reuse and speculation disabled |
| Budget | **$75 total gross Modal usage**, including preparation, failed attempts, and reruns |
| Article | Approximately 1,200–1,800 words and three figures, with detailed methods and raw data linked |

The full matrix has 72 cells and 256 measured requests per cell. A smaller pilot must establish compatibility and forecast cost before freezing the full or compact matrix.

Prices checked on October 30, 2026 imply approximately **$5.94/hour** for the proposed H200 container including eight CPU cores and 128 GiB of host RAM. Refresh prices before execution; no runtime or performance result is implied by this estimate.

## Next integration tasks

1. Run CPU inspection of the pinned images and establish the shared, isolated Python/client environment with complete dependency locks.
2. Resolve and download the official model and public dataset inside Modal; inspect the actual tokenizer/template, corpus format, and licenses.
3. Finish the Modal Volume and GPU orchestration adapter using the tested local runner, ledger, and recovery planner. Preserve reservation/call state across failures and machine changes.
4. Perform real-model conformance checks and the bounded pilot, then freeze the formal protocol and cost forecast.

`python -m inference_bench validate --require-ready` currently fails with the unresolved prerequisites. `modal_app.py` exposes CPU inspection only; remote GPU dispatch is not wired yet. Direct dependency candidates are recorded in [requirements/](requirements/README.md), and the source/client locks intentionally remain unresolved.

Suggested continuation instruction:

> Read AGENTS.md, docs/runbook.md, and docs/megaplan.md. Continue at the external qualification boundary: inspect the pinned engine images on CPUs, resolve source and client locks, then complete the Modal Volume/GPU adapter around the existing runner and budget ledger. Preserve the agreed protocol and $75 gross budget. Qualify the real request semantics before formal measurements.

The site repository is [fxcawley/site](https://github.com/fxcawley/site). Keep the article draft in this repository until measured results are ready: the existing site content loader has no draft filter.

## Reproducibility and licensing

The planned release will include the source, frozen configuration, provenance manifests, summary tables, figures, and a checksummed raw-data bundle. GPU packages, credentials, downloaded weights, caches, and raw runs do not belong in Git.

Original project code and documentation are licensed under [MIT](LICENSE). Model weights and source datasets retain their own licenses and attribution requirements.
