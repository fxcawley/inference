# Handoff: continue development on another computer

This is the starting point for continuing the project on a new machine. It summarizes the current state, sets up the environment, and lists the remaining work in order. The [runbook](runbook.md) has the detailed command reference, the [methodology](methodology.md) defines the measurements, and the [megaplan](megaplan.md) records the full design.

## 1. Current state

The repository is [fxcawley/inference](https://github.com/fxcawley/inference). Everything is on `main`:

- PR #1 merged the offline harness, conformance probes, qualification gate, budget ledger, pilot forecast, recovery planning, and offline reports.
- PR #2 (`f281888`) merged the backend-neutral campaign controller and remote worker entry.

The branches `build/offline-benchmark` and `build/campaign-controller` are fully merged and can be deleted on GitHub.

**Built and verified on CPUs:**

- Strict configuration, paired schedules, and deterministic prompt preparation, including pilot, warmup, and conformance inputs.
- One HTTPX streaming client with authoritative token accounting and no request retries.
- A process-group runner with absolute leases, GPU identity checks, input validation before startup, and five conformance probes before warmup.
- Checksummed artifacts, offline analysis with three SVG figures, paired SGLang ÷ vLLM ratios, and reproducible bundles.
- Qualification reports that the formal gate verifies against the model revision, image digest, engine settings, and harness source.
- One persistent project ledger with conservative billing reconciliation.
- A pilot cost forecast, read-only recovery planning, and a serial campaign controller that the Modal adapter will drive.

All 94 tests pass locally and in GitHub Actions. They use loopback HTTP, short-lived local processes, and simulated cloud backends; nothing has exercised a real engine, model, or GPU.

**Not done yet:**

- The CPU preflight of the pinned engine images has never run.
- Dependency locks, model and dataset revisions, prompt preparation with the real tokenizer, and the dataset license check are unresolved.
- The Modal adapter does not exist: backend, GPU functions, Volumes, `pilot`, `benchmark`, `resolve`, and `export` entrypoints.
- No pilot, qualification, formal measurement, article, or site integration has happened.

**Spend so far: none.** No Modal job has run and no model has been downloaded. No project ledger exists yet; the first GPU reservation will create it on the new machine.

`python -m inference_bench plan` currently lists 11 pending prerequisites. They clear in the steps below.

## 2. Fixed decisions

| Item | Decision |
|---|---|
| Question | Throughput versus first-response and completion latency as concurrency changes, for two released serving stacks on the same GPU model. |
| Model | `google/gemma-4-26B-A4B-it`, official BF16 checkpoint, downloaded inside Modal at an immutable revision. |
| Engines | vLLM 0.32.0 and SGLang 0.5.22, official images pinned by digest in `configs/engines.lock.json`. |
| Hardware | One H200 SXM, TP=1, 8 CPU cores, 128 GiB RAM; no region pinning. |
| Workload | WikiText-2 raw train text; 512/128, 4,096/128, and 512/512 input/output tokens; C=1, 4, 16, 32; 256 requests per cell; three repetitions per engine. |
| Controls | Greedy decoding, fixed output length with EOS ignored, no prefix caching, no speculation, non-thinking template. |
| Fallback | If the forecast says the full matrix does not fit, drop C=32, then the decode shape, then reduce to 200 requests per cell. The maintainer reviewed and kept this order. |
| Budget | $75 gross, including preparation, failures, and reruns. |
| Article | About 1,200–1,800 neutral words and three figures for cawley.dev, written only from measured results. |

## 3. Set up the new machine

### Windows

Use WSL2 with Ubuntu 24.04, which provides Python 3.12. The process-group tests, the CI environment, and the remote containers are all Linux. Inside the Ubuntu shell:

```bash
sudo apt update && sudo apt install -y git python3-venv
```

If you also use Git from PowerShell and it reports that `git` is not recognized, add Git for Windows' `cmd` directory to your user `PATH`.

### Linux or macOS

Install Git and Python 3.12. On macOS, the Linux-only process cleanup tests are skipped.

### Clone and verify

```bash
git clone https://github.com/fxcawley/inference.git
cd inference
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[client]'
python -m unittest discover -s tests         # expect: Ran 94 tests ... OK
python -m inference_bench plan               # expect: 72 cells and 11 "Pending" lines
python -m inference_bench validate --require-ready   # expected to fail until qualification
```

For commits, use your GitHub noreply address in this repository and plain commit messages without generated-by or co-author trailers. Use `gh auth login` if you want to open PRs from the command line.

Nothing needs to be copied from the previous machine. The code and documents are on GitHub, and no ledger, billing export, or real artifact exists yet. Its local Git bundle is superseded by `main`.

## 4. Rules that protect the budget and the result

- **One ledger, forever.** The first GPU reservation creates the project ledger. Keep it at `private/budget-ledger.json`; `private/` is ignored by Git. Back it up after every stage. Never delete, reset, or edit it. Never start a second one, even on another machine or in a new billing month.
- **Freeze prices before the ledger exists.** The ledger is bound to the `[budget]`, `[pricing]`, and `[hardware]` sections. Refresh and commit the prices before the first reservation; after that, those sections cannot change.
- **Freeze the package before the qualifying pilot.** Qualification reports bind the hash of `src/inference_bench/*.py`. Any change to those files after the pilot invalidates the reports, and the formal gate will require a new pilot. Put Modal-only glue in `modal_app.py`, which is outside that hash. If a log pattern does not match real output, accept the review item with the value read from the log instead of changing code.
- **One GPU call at a time**, each with a ledger reservation. CI never launches paid work.
- **No formal measurements** until both engines have passed qualification reports recorded in `configs/engines.lock.json`. Never accept a failed check or edit a report.
- **Synthetic fixtures are never results.** Anything made with `--allow-synthetic` stays out of the article.
- **Secrets** live only in Modal Secrets or your credential store. Keep billing exports in `private/`.
- **Emergency stop:** `modal app stop <app-id>`. Then confirm that the work ended before settling its reservation.

## 5. Remaining work, in order

### Step 0: Modal account and budget backstop (no cost)

1. Install the Modal client and authenticate: `python -m pip install -e '.[modal]'`, then `modal setup`.
2. Set the Workspace's native gross usage budget to $75, or to existing monthly usage plus $75.
3. Run `modal billing rates --json`. If H200, CPU, or memory prices changed, update `[pricing]` and `checked_on` in `configs/benchmark.toml`, run the tests, and commit.

### Step 1: CPU preflight of the pinned images (small CPU cost)

```bash
modal run modal_app.py::preflight --config configs/benchmark.toml
```

Inspect `artifacts/preflight/<id>/vllm.json` and `sglang.json`. Each needs an `engine_executable`, `cli_help.returncode` of 0, and an empty `flags_not_listed_in_help`. The most likely problems are vLLM's `--disable-log-requests` flag and the `sglang serve` entry point. Fix `engines.command()` and its tests, then rerun until both are clean. Keep the final preflight directory; qualification uses it.

Also record each image's Python version and installed engine, Torch, and Transformers versions.

### Step 2: Common client environment and locks

1. Decide how the shared client runs in both images: a separate Python 3.12 virtual environment, without changing either engine's interpreter.
2. Generate hashed locks from `requirements/*.in`, for example with `uv pip compile --generate-hashes --python-version 3.12 requirements/client.in -o requirements/client.lock.txt`.
3. In `configs/engines.lock.json`, set `client.python_version` to the patch version actually used and `client.requirements_sha256` to the lock file's SHA-256.

### Step 3: Source preparation in Modal (CPU; code to write)

Add a CPU `prepare` function to `modal_app.py` that uses an upstream-access Secret and a version-2 Volume:

1. `preparation.resolve_sources(config)` for immutable model and dataset commits.
2. `preparation.download_model(source, directory)` for the snapshot and its file hashes.
3. Export the WikiText-2 raw v1 train split at the pinned commit into `{"text": ...}` JSONL. This adapter does not exist yet; record its conversion.
4. `preparation.prepare_local(config, model_dir, corpus_jsonl, output)` in an image that has Transformers. It writes measured, warmup, pilot, and conformance prompts with the snapshot's stop token IDs.
5. Commit the Volume.

Then complete `configs/sources.lock.json`: `status: "complete"`, both revisions and file hashes, the dataset license and attribution from its card, the SHA-256 of the prepared `manifest.json`, and generator version 3. Check the real template: exactly one BOS, no thinking block, and exact prompt lengths. The tests use a fake tokenizer only.

### Step 4: Modal GPU adapter (code to write)

The contract is in `controller.Backend` and the runbook's [campaign controller](runbook.md#campaign-controller) section:

- A backend with `spawn`, `poll`, and `cancel`. `poll` must work from a new process using only the call ID.
- `run_vllm_block` and `run_sglang_block` using the pinned image and `planned_gpu_options()`, with the model and inputs mounted read-only and the results Volume mounted. Each calls `runner.run_worker(...)`, commits the results Volume, and returns the result.
- Local entrypoints:
  - `pilot` and `benchmark` create the campaign locally with `artifacts.create_campaign` and run `controller.plan_blocks` plus `controller.execute`.
  - `resolve` calls `controller.resolve_outstanding`.
  - `export` copies committed `runs/` directories and prompts into the local campaign root, then seals it.
- Decide the compile-cache policy and whether the first pilot needs a readiness limit above 900 seconds. Record cache state in the artifacts.

Before any GPU use, check the Modal SDK calls against the installed version and exercise the backend with a trivial CPU function.

### Step 5: Pilot (first GPU spend; at most 6,000 seconds, about $9.90)

```bash
python -m inference_bench blocks --phase pilot --lease-seconds 2100 --queue-seconds 300
modal run modal_app.py::pilot            # to be written
python -m inference_bench verify-artifacts --input artifacts/<pilot>
python -m inference_bench analyze --input artifacts/<pilot> --output artifacts/<pilot>-report
python -m inference_bench qualify --input artifacts/<pilot> --engine vllm \
  --preflight artifacts/preflight/<id> --output configs/qualification/vllm.json
python -m inference_bench qualify --input artifacts/<pilot> --engine sglang \
  --preflight artifacts/preflight/<id> --output configs/qualification/sglang.json
```

Size the pilot lease from the expected startup and pilot work. Settled charges, not reservations, accumulate against the 6,000-second pilot allowance, but each new reservation must still fit what remains; leave room for at least one failed start. Review every non-passing check. Accept a review item only with a written reason. Record each passed report's path and SHA-256 in `configs/engines.lock.json`.

Then save the billing export and record it:

```bash
python -m inference_bench record-billing --ledger private/budget-ledger.json \
  --through <UTC time with zone> --gpu-usd <X> --other-usd <Y> --source private/billing-<date>.json
```

### Step 6: Forecast and freeze

```bash
python -m inference_bench forecast --input artifacts/<pilot> --ledger private/budget-ledger.json
```

Adopt `recommended_protocol`. For a compact protocol, edit `configs/benchmark.toml`:

- `without-c32`: set `concurrency = [1, 4, 16]`.
- `two-profiles`: also remove `[profiles.decode]`.
- `two-profiles-200`: also set `requests_per_cell = 200`.

Leave `[budget]`, `[pricing]`, and `[hardware]` untouched. Commit, then require `python -m inference_bench validate --require-ready` to pass with a clean tree. If nothing fits, stop and report the pilot findings.

### Step 7: Formal measurements (at most 24,000 seconds formal and 6,000 seconds reruns)

```bash
python -m inference_bench blocks --phase measurement --lease-seconds <forecast block_seconds>
modal run modal_app.py::benchmark        # to be written
```

After any stop, resolve outstanding calls, then replan only missing or replaceable cells:

```bash
python -m inference_bench resume-plan --input artifacts/<campaign> --ledger private/budget-ledger.json
python -m inference_bench blocks --phase measurement --lease-seconds <block_seconds> \
  --resume-input artifacts/<campaign> --ledger private/budget-ledger.json
```

Export, seal, verify, analyze, and bundle the campaign. Record billing after the stage.

### Step 8: Article and publication

1. Draft `report/article.md` from `claims.json`, the three figures, and the paired-ratio table. Disclose the limits in the methodology, including synchronized waves at C≥4 and KV capacity.
2. Publish a GitHub release with the raw bundle and its SHA-256 receipt.
3. Integrate the post into [fxcawley/site](https://github.com/fxcawley/site) on a separate branch, following that repository's instructions. Build it and review it before merging.

## 6. Budget at a glance

| Allocation | Limit |
|---|---:|
| Pilot and compatibility work | 6,000 GPU-container seconds, about $9.90 |
| Formal measurements | 24,000 seconds, about $39.60 |
| Replacement attempts | 6,000 seconds, about $9.90 |
| CPU preparation, image builds, storage, export | $5.00 |
| Reserve | $10.00 |
| **Total** | **$75.00 gross** |

The planned container costs about $5.94/hour at the October 30, 2026 list prices. Useful commands:

- `python -m inference_bench ledger --ledger private/budget-ledger.json` shows exposure, billing, and remaining allowances.
- `python -m inference_bench settle ... --confirmed-stopped` settles a reservation manually after you confirm in Modal that its work ended.

## 7. Code map

| Path | Purpose |
|---|---|
| `configs/benchmark.toml` | Protocol, resources, budget, prices, and project `ledger_id`. |
| `configs/sources.lock.json`, `configs/engines.lock.json` | Model, dataset, prompt, image, client, and qualification identities. |
| `src/inference_bench/config.py` | Validation and paired schedules. |
| `workloads.py`, `preparation.py` | Prompt assembly, conformance probes, and source preparation. |
| `client.py`, `metrics.py` | Shared streaming client and metric definitions. |
| `runner.py` | Server process lifecycle, probes, cells, and `run_worker`. |
| `engines.py`, `preflight.py` | Engine commands, hardware checks, log parsing, and CPU image inspection. |
| `qualification.py` | Probe definitions, qualification reports, and gate verification. |
| `budget.py` | Persistent ledger, reservations, settlement, and billing. |
| `controller.py` | Block planning, serial dispatch, and recovery. |
| `planning.py` | Pilot forecast and resume plan. |
| `artifacts.py`, `analysis.py` | Checksums, campaigns, bundles, tables, figures, and claims. |
| `examples.py` | Labelled synthetic fixtures. |
| `modal_app.py` | Modal CPU preflight and the GPU resource contract. |
| `tests/` | CPU tests; `test_runner.py` holds process, ledger, controller, and worker tests. |

## 8. Known risks and open decisions

- The full matrix fits the formal allowance only if single-stream decode is fast enough. A rough planning model put the threshold near 150 tokens/s; the pilot forecast decides.
- Fresh containers start with empty compiler and JIT caches. Cold starts may approach the 900-second readiness limit and use much of the pilot allowance.
- vLLM's `--generation-config vllm` likely drops the checkpoint's extra EOS IDs. Timed requests ignore EOS; probes send pinned stop IDs. Confirm during the pilot.
- At C=32 the 4,096-token shape needs 135,168 concurrent tokens. Each engine's KV accounting may make capacity bind differently. Report it.
- The KV and context patterns were taken from earlier engine releases and are unconfirmed.
- WikiText is reported to carry CC BY-SA terms. Verify them before publishing prepared prompt text.

The full list is in the runbook's [secondary assessment](runbook.md#secondary-assessment).

## 9. Prompt for an AI agent

> Read docs/handoff.md, AGENTS.md, docs/runbook.md, and docs/megaplan.md. The CPU harness, qualification gate, project ledger, and serial campaign controller are implemented and tested; nothing has run on Modal and no ledger exists. Continue from the first incomplete step in section 5 of the handoff. Ask before any step that costs money, and never start GPU work without my explicit approval. Keep the $75 gross budget and a single project ledger, freeze `src/inference_bench` before the qualifying pilot, and record actual validation outcomes without overstating them.
