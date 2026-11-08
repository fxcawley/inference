# Current implementation runbook

The local benchmark core, conformance probes, qualification reports, pilot forecast, budget reconciliation, recovery planner, and backend-neutral campaign controller are implemented and tested on Python 3.12 in Linux/WSL. The real engine images, model/tokenizer, dataset source, and Modal integration have not been exercised. There are no measured vLLM/SGLang results and no passed engine qualification.

## Local setup and verification

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[client]'
python -m unittest discover -s tests -v
python -m inference_bench plan
python -m inference_bench validate
```

The package has no mandatory dependencies. The `client` extra adds HTTPX 0.28.1 for actual streaming HTTP and loopback tests. The core, fixture generator, analysis, and artifact checks also run directly from a checkout with `PYTHONPATH=src`.

CPU tests cover actual fragmented HTTP streams, UTF-8 boundaries, concurrency, terminal usage, failures, total request timeouts, exact prompt lengths, probe semantics, metric units, first-valid replacement selection, paired ratios, checksums, synthetic-data labelling, qualification-report verification, process groups, persistent budgets, and billing reconciliation. The Linux process tests start and terminate short-lived dummy processes.

CI runs the same CPU tests, regenerates a synthetic report, and exercises the synthetic pilot forecast and qualification report. It has no Modal credentials, model access, or GPU jobs.

For a pass that opens no network sockets, including loopback connections, use the already installed environment:

```bash
PYTHONPATH=src:tests python -m unittest \
  test_protocol test_metrics test_runner test_client.DecoderTests test_client.RequestContractTests -v
```

The offline development passes also installed a Python audit hook that rejected socket connections, binds, datagram sends, and DNS resolution. The six HTTP integration tests were excluded from those passes. No packages or remote services were needed.

## Working commands

| Command | Current behavior |
|---|---|
| `plan [--json]` | Validate configuration; print matrix, schedule, token totals, cost bounds, and unresolved qualification items. No cloud calls. |
| `plan --phase pilot [--json]` | Print the paired, single-repetition pilot and its per-cell request/warmup counts. |
| `validate` | Validate local configuration structure and numeric constraints. |
| `validate --require-ready` | Also require immutable sources, client identity, verified engine qualification reports, the formal matrix, and a clean committed harness. Currently fails as intended. |
| `commands --engine vllm --model-path /models/snapshot` | Print candidate argv as a safely quoted command; never execute it. |
| `inspect-model --model-dir PATH` | Inspect local checkpoint metadata and shard inventory and compute hashes. This is not runtime qualification. |
| `prepare-prompts --model-dir PATH --corpus-jsonl FILE --output DIR` | Use the local snapshot's tokenizer to prepare exact-length prompts and conformance probes. Transformers is imported only for this command. |
| `example --output DIR [--phase pilot]` | Create a deterministic synthetic campaign for exercising reports, forecasts, and qualification. Refuse an existing directory. |
| `forecast --input PILOT --ledger FILE` | Verify a complete pilot and existing ledger; print estimates for the full matrix and three ordered reductions. |
| `qualify --input PILOT --engine NAME --output FILE [--preflight DIR] [--accept CHECK=REASON]` | Evaluate one engine's sealed pilot block and write a qualification report. Return 3 unless every required check passed or was explicitly accepted. |
| `resume-plan --input CAMPAIGN --ledger FILE` | Inspect complete or interrupted formal artifacts and the existing ledger; print remaining work and blockers without writing either input. |
| `ledger --ledger FILE` | Print reservations, billing reconciliation, remaining allowances, and warnings. Requires an existing ledger. |
| `record-billing --ledger FILE --through TIME --gpu-usd X --other-usd Y --source EXPORT` | Append cumulative gross charges from a saved billing export. |
| `blocks --phase NAME --lease-seconds N [--queue-seconds M] [--resume-input DIR --ledger FILE]` | Preview the serial fresh-process blocks, each reservation, and worst-case stage use. Dispatches nothing. |
| `settle --ledger FILE --reservation ID --charged-seconds N --confirmed-stopped` | Settle one reservation after its termination was confirmed outside the controller. Without the flag it refuses. |
| `verify-artifacts --input DIR` | Require the completion marker, exact file set, sizes, and SHA-256 hashes. Reject symlinks and escaping paths. |
| `analyze --input DIR --output REPORT` | Recompute metrics from raw records and generate CSVs, paired ratios, three SVGs, tables, and claims. Output must be outside the sealed input. |
| `bundle --input DIR --output FILE.tar.gz` | Create a deterministic archive and SHA-256 receipt. Refuse to replace an existing archive. |

Synthetic analysis, bundling, forecasts, qualification, and recovery plans require `--allow-synthetic`. Every synthetic figure carries a visible label, and its claims are marked as nonpublishable measurements. A synthetic qualification report can never satisfy the formal gate. Fixtures are useful for testing the pipeline; they are not model performance estimates.

## Offline example

```bash
python -m inference_bench example --output artifacts/example
python -m inference_bench verify-artifacts --input artifacts/example
python -m inference_bench analyze \
  --input artifacts/example --output artifacts/example-report --allow-synthetic
python -m inference_bench bundle \
  --input artifacts/example --output artifacts/example.tar.gz --allow-synthetic
```

Open `artifacts/example-report/figures/throughput.svg`, `ttft.svg`, and `latency-throughput.svg` in a browser. `aggregate.csv` includes run medians and ranges and derived compute-price estimates. `paired-ratios.csv` gives SGLang ÷ vLLM for each matched repetition. Actual project cost remains null until billing reconciliation.

Use another output directory for a second example, or deliberately remove the old synthetic artifacts first. Generated data live under the ignored `artifacts/` directory.

## Pilot planning and cost forecasting

The pilot uses one fresh process per engine, all three profiles and all four concurrency levels, `max(16, 2*C)` timed requests per cell, and the common `max(8, 2*C)` warmup rule. C=4 and C=16 use the same small pilot rule so their timings are observed rather than interpolated. The default totals are 24 cells, 768 pilot requests, and 672 warmup requests. Five untimed conformance probes per block are additional.

Prompt generator version 3 writes `measured.jsonl.gz`, `warmup.jsonl.gz`, `pilot.jsonl.gz`, `pilot-warmup.jsonl.gz`, and `conformance.jsonl.gz`. Pilot input IDs and sampling seeds are separate from formal inputs. Both engines still use the same ordered inputs at each concurrency. Regenerate preparation artifacts made by earlier generator versions.

Exercise the forecast entirely offline:

```bash
python -m inference_bench example --phase pilot --output artifacts/example-pilot
python -m inference_bench forecast \
  --input artifacts/example-pilot \
  --ledger artifacts/example-pilot/budget-ledger.json --allow-synthetic
python -m inference_bench analyze \
  --input artifacts/example-pilot --output artifacts/example-pilot-report --allow-synthetic
```

The fixture includes arbitrary block timings and a zero-usage ledger. Its recommendations are synthetic pipeline checks, not estimates of Gemma performance.

For an actual forecast, supply a verified pilot export and the current project ledger. Required evidence includes both complete engine blocks, every scheduled pilot cell, valid warmup records, and monotonic `timing.json` block timestamps. Every actual block/cell must identify its ledger reservation. Missing cells, wrong phases, corrupt checksums, inconsistent timings, and missing budget history stop the forecast.

The calculation uses the slower engine's seconds per request and warmup duration for each profile/concurrency pair. It adds the maximum observed startup duration. Observed noncell overhead, including conformance probes, serialization, and artifact writes, is scaled up by the ratio of formal to pilot request counts, including warmup. It never scales that overhead down. Each new block also receives 210 seconds for container initialization and final shutdown/commit work that the local runner cannot time. The entire estimate then receives 25% timing headroom, rounded up to whole seconds per block. If reconciled bills exceed the ledger's list-price charges, candidate costs are multiplied by the observed billing ratio.

The recommendation considers these choices in order, keeping both engines and three repetitions:

1. Full matrix: 72 cells, 256 requests per cell.
2. Drop C=32: 54 cells, 256 requests per cell.
3. Keep `short` and `prefill`: 36 cells, 256 requests per cell.
4. Use 200 requests per retained cell: 36 cells.

The maintainer reviewed this order during the secondary assessment and kept it. Expect the first step to save relatively little time; see the assessment below.

Each candidate includes its complete configuration and hash, estimated block/total duration, nominal and projected cost, and budget failures. The forecast checks the block limit, formal-stage allowance, remaining campaign time, and remaining dollar allocation. Unresolved calls, overruns, a different harness, or any already-started formal/rerun reservation prevent a recommendation. If no candidate fits, stop after the pilot.

These are timing estimates, not billing guarantees or runtime qualification. Prices are read from the frozen configuration; forecasting does not refresh them. The remote controller must still freeze the selected protocol, enforce absolute leases, reserve each call, and reconcile actual usage. The smaller protocols must be selected before formal work starts.

## Engine qualification

Every pilot and formal block runs five untimed C=1 probes after readiness and before warmup. The probe prompts are prepared with the pinned tokenizer and template:

| Probe | Request | Pass condition |
|---|---|---|
| `eos_honored` | "Reply with exactly one word: yes"; EOS honored; 64-token limit | Finish reason `stop` before the limit. |
| `eos_ignored` | Same prompt; EOS ignored; 64 tokens | Finish reason `length` with exactly 64 output tokens. |
| `arithmetic` | "What is 17 + 25?"; EOS honored; 32-token limit | Natural stop and `42` in the text. |
| `capital` | "What is the capital of France?"; EOS honored; 32-token limit | Natural stop and `paris` in the text. |
| `passage_summary` | Standard instruction and corpus excerpt; EOS honored; 256-token limit | At least 40 printable characters and 8 words, no replacement characters, not dominated by repeated words. |

Probes that honor EOS send `stop_token_ids` from the pinned snapshot: `generation_config.json` `eos_token_id` plus the tokenizer EOS. They do not rely on either server's generation defaults. Measured, pilot, and warmup requests still ignore EOS and never send stop IDs. A failed probe stops the block before warmup or measured work; its records and evaluations remain in `conformance/`.

These probes are behavioral sanity checks. They do not compare model quality.

After a pilot export, qualify each engine:

```bash
python -m inference_bench qualify --input artifacts/pilot --engine vllm \
  --preflight artifacts/preflight/<id> --output configs/qualification/vllm.json
```

The report evaluates the engine's sealed block and binds the result to the pilot protocol, harness source, model revision, image digest, normalized engine argv, and preflight file. Required checks are:

| Check | Requirement |
|---|---|
| `pilot_cells` | Every pilot cell is a valid first attempt. |
| `warmup_records` | Every warmup has complete, exact usage. |
| `warmup_coverage` | At C=1, the first measured request is no more than 1.5× the cell's median E2E latency. Otherwise review. |
| `client_capacity` | Every cell's p95 client event-loop lag is at most 2 ms. |
| `hardware` | One full H200 SXM with MIG disabled. |
| `served_model` | `/v1/models` lists `gemma4-bench` with the configured context limit. Missing context metadata requires review. |
| `engine_settings` | The launched argv equals the configured candidate, ignoring model path and port. |
| `conformance` | All five probes are present, have campaign provenance, and pass when re-evaluated. |
| `kv_capacity` | KV-cache capacity parsed from the server log covers the largest cell's tokens. Missing or smaller capacity requires review. |
| `cpu_preflight` | The pinned image's CLI completed, lists every candidate flag, and inspected the same image and argv. Missing preflight requires review. |

The KV and context checks use log/endpoint formats from earlier releases. They have not been confirmed against the pinned images. An unmatched pattern is reported as unobserved, never inferred.

A review item can pass only with `--accept CHECK=REASON`; the reason is stored in the report. Failed checks cannot be accepted. The command writes non-passing reports for evidence and returns status 3.

Record a passed report in `configs/engines.lock.json`:

```json
"qualification": {
  "status": "passed",
  "report": "qualification/vllm.json",
  "report_sha256": "<SHA-256 printed by qualify>"
}
```

`plan`, `validate --require-ready`, formal campaign creation, and formal analysis verify the report file and its hash. They also check that every required check passed or was accepted, the evidence is a real pilot, and the image and model revision match the locks. Formal readiness also requires matching engine settings and harness source. Formal campaigns copy the reports into their bundle so offline analysis can verify them again.

## Budget ledger and billing reconciliation

`budget.ledger_id` and the budget/pricing/hardware policy identify one persistent project ledger. Pilot and formal campaigns share it even when their campaign IDs differ. Changing the identifier, budget, prices, or resources cannot reuse or reset an existing ledger.

Reservations still use the frozen list price. After each stage, or whenever no lease is active, save the Modal billing export for the campaign. Then record cumulative gross charges since the project began:

```bash
modal billing report --for <period> --resolution h --show-resources --json > private/billing-<date>.json
python -m inference_bench record-billing --ledger artifacts/budget-ledger.json \
  --through 2026-11-05T18:00:00Z --gpu-usd 12.34 --other-usd 0.56 \
  --source private/billing-<date>.json --note "pilot reconciliation"
```

Derive the amounts from the saved export, filtered to this project's GPU functions and its CPU/storage/egress work. Keep billing exports private; the ledger stores their SHA-256, not their contents. The Modal billing JSON format has not been inspected offline, so the amounts are entered explicitly.

Billing records are append-only and cumulative: time and charges cannot decrease. A billing time cannot fall inside a reservation lease, and every lease that ended by then must be settled. New leases cannot start inside an already reconciled period.

Reconciliation can only make the ledger more conservative:

- Billed GPU-container charges above the settled list-price charges add the difference to exposure.
- The ratio of billed to settled GPU-container charges, when above one, scales later reservations and forecasts.
- Non-GPU charges reserve at least the configured $5 allocation; any excess consumes the reserve.
- A lower bill never releases allowance.

`ledger --ledger FILE` reports these values, the remaining dollar and time allowances, and warnings.

An overrun (charged seconds above the reservation) blocks further reservations until billing is recorded through the end of the overrun's charged interval. The billing time cannot fall inside that interval, so the recorded bill includes the overrun's actual usage.

## Campaign controller

`controller.py` turns a frozen schedule into fresh-process blocks and runs them one at a time. It contains no Modal code; the remote adapter supplies a small backend.

`plan_blocks` creates one block per repetition, engine, and attempt in frozen schedule order: two pilot blocks, or six formal blocks with the alternating engine order. With `resume-plan` output, it groups the eligible cells. Unstarted cells form attempt 1 in the formal stage; replacements form attempt 2 in the rerun stage, each linked to its original artifact.

For every block, `execute`:

1. Refuses to start while any reservation is unresolved or an overrun is unreconciled.
2. Reserves `lease + queue` seconds in the block's stage. A budget refusal stops before submission.
3. Submits one payload and records the returned call ID in the ledger.
4. Waits until the lease deadline plus 60 seconds, then cancels a call that is still running.
5. Settles only after the backend reports that the call ended. Otherwise the reservation remains unresolved and the controller stops.
6. Stops after any cancelled, overrun, failed, or incomplete block. Inspect it and replan with `resume-plan`.

The charge runs from the earliest plausible container start to the observed end, rounded up. When the worker reports its start time, queue time before that start minus 180 seconds is excluded; otherwise the whole time since reservation counts. A platform restart can make this low, so record billing after each stage.

The worker receives its absolute deadline and a minimum working time equal to the lease. If GPU queueing leaves less than that, it fails before inspecting hardware or creating artifacts. It never starts a block whose results the lease could not cover.

`resolve_outstanding` handles calls left by an earlier controller process. It leaves calls running within their lease alone. It cancels calls past their deadline, then settles only after termination is confirmed. When the end time is unknown, it charges through the lease deadline. Reservations without a call ID need manual confirmation and the `settle` command.

Preview a plan without dispatching:

```bash
python -m inference_bench blocks --phase pilot --lease-seconds 2400 --queue-seconds 600
python -m inference_bench blocks --phase measurement --lease-seconds <forecast block_seconds> \
  --resume-input artifacts/campaign --ledger artifacts/budget-ledger.json
```

Use the forecast's `block_seconds` as the formal lease. The pilot has no forecast; choose a lease from the startup and cell estimates, and keep two pilot reservations within the 6,000-second pilot allowance.

### Remaining Modal adapter

The adapter must provide `spawn`, `poll`, and `cancel` as described in `controller.Backend`:

- `spawn` submits one GPU invocation without waiting and returns its durable call ID. Do not enable retries.
- `poll(call_id, timeout_s)` waits up to the timeout. It returns None while the call can still run and a terminal record afterwards. It must work from a new process with only the call ID, so `resolve_outstanding` can recover a disconnected session.
- `cancel` terminates every container serving the call.

Each GPU function should mount the verified model snapshot and prepared inputs read-only, mount the results Volume, call `runner.run_worker(payload, results_root=..., model_path=..., prompts=...)`, commit the results Volume, and return the worker result. Use one function per engine with the pinned image and `planned_gpu_options()`. Keep the campaign manifest, locks, and ledger with the controller; after a stage, export the committed block directories into the local campaign root, seal it, and analyze it. None of this Modal glue exists yet, and the SDK calls have not been checked against Modal 1.6.3.

## Implemented boundaries

`workloads.py` accepts a tokenizer object, preserves its role/template prefix and assistant suffix, and fills only the excerpt span to achieve the total input-token budget. It records deterministic source offsets, separates warmup/pilot/measured IDs, and renders the conformance probes with the same template. The real Gemma tokenizer and non-thinking template still need inspection.

`preparation.py` can resolve immutable Hub identities, download a model at an explicit commit, inventory shards, and prepare a supplied public corpus JSONL file. It reads probe stop IDs from the snapshot. It does not infer dataset licensing or guess the actual WikiText file layout. The remaining corpus adapter should export the verified upstream format into `{"text": "..."}` JSONL rows and record its conversion/version.

`client.py` implements the common completions request and parser. A server must emit a finish reason, final authoritative usage, and DONE. Only conformance probes may honor EOS; they may stop early but cannot exceed the requested limit. The common token-ID/EOS/streaming contract has not yet been tested against either real engine.

`runner.py` validates every probe and prompt set before inspecting hardware or starting a server. It owns a fresh server process group, validates the GPU, polls health/model identity, records `/v1/models`, runs conformance probes, runs phase-specific warmup and cells, and writes checksummed outcomes plus monotonic startup/completion timings. It rejects cells outside the selected pilot/formal schedule. An independent timer stops the process group on lease expiry, even if the asyncio loop is blocked. It reserves the final 30 seconds of the supplied lease for shutdown/artifacts. The remote adapter must still enforce the entire container lifetime and Volume commits.

`qualification.py` holds the canonical probes, the report builder, and the gate's report verification. Probe expectations live in source, so changing them changes the frozen harness identity.

`budget.py` provides a persistent single-controller ledger. An unresolved call retains its full reservation and blocks another GPU submission. Settlement requires explicit confirmation that work stopped; an overrun is charged and blocks further work until billing covers it. The lock file is local process coordination, not a distributed Volume lock.

`controller.py` plans and serially executes blocks through an injected backend, and resolves calls left by an earlier controller. `runner.run_worker` validates a block payload, enforces the minimum working time, links replacement cells, runs the block, and returns its start and finish times.

Ledger reads validate reservation and billing identities, amounts, durations, statuses, deadlines, and cumulative order. Malformed or duplicate records cannot silently lower exposure. Planning requires an existing ledger file; a missing file never means zero historical usage.

## Secondary assessment

The proposal and implementation were reassessed offline after the pilot and recovery work. No engine, model, or GPU measurement informed this review. Estimates below come from a deliberately rough closed-loop model with assumed speeds and are planning context only.

The core design remains sound: one shared client, colocated loopback measurement, authoritative token accounting, fixed request schedules, preserved failures, first-valid replacement, and conservative serial leases. The main risks are budget feasibility, interpretation at high concurrency, cold-start cost, and checks that previously existed only on paper.

| Finding | Disposition |
|---|---|
| Full-matrix feasibility depends strongly on single-stream decode speed. In the rough model, the full matrix exceeded the 24,000-second formal allowance when batch-1 decode was below about 150 tokens/s, mostly because sequential C=1 work dominates. | The pilot forecast decides. No result is implied. |
| The agreed first fallback, dropping C=32, saved only about 4–12% of modelled cell time, while C=1 accounted for about 54–72%. It rarely rescues an over-budget forecast and removes the highest-load comparison. | Reviewed with the maintainer, who kept the agreed order. Every candidate's estimate remains visible. |
| The formal gate accepted any `passed` status with a 64-character hash. | Fixed. The gate verifies the report file, hash, checks, evidence kind, and bindings. |
| The planned conformance checks were not executable, and the client could only send fixed-length requests. | Fixed. Every block runs five probes and stops before measured work if one fails. |
| Pilot and formal campaigns could share a ledger only by reusing one campaign ID. | Fixed with a configured project `ledger_id`. |
| Billing reconciliation and non-GPU costs were not represented. | Fixed with conservative, append-only cumulative billing records. |
| Paired engine ratios from the megaplan were not computed. | Fixed in `paired-ratios.csv`, `claims.json`, and `tables.md`. |
| Missing prompts could be discovered only after paid model startup. | Fixed. All inputs are resolved before hardware inspection. |
| `--generation-config vllm` likely omits the checkpoint's generation-config EOS list in vLLM, while SGLang reads model stop tokens separately. | Timed requests ignore EOS, so they are unaffected. Probes send pinned stop IDs. Confirm actual behavior during qualification. |
| Fixed input/output lengths and simultaneous worker starts make C≥4 cells run as synchronized waves. High-concurrency TTFT then largely measures how each engine prefills a full wave, including its default chunked-prefill budget. | Disclose in the methods and article. No protocol change. |
| At C=32, the prefill shape needs 135,168 concurrent tokens. Engine KV accounting, sliding-window support, and memory reservations may make capacity bind differently. | Qualification records parsed capacity and requires review if unobserved or insufficient. Report it beside the results. |
| Fresh single-use containers start with empty compiler/JIT caches. With eight CPU cores, cold starts could approach the 900-second readiness limit and consume much of the pilot allowance. | Adapter decision: consider a per-engine writable compile-cache Volume and a longer first-pilot readiness limit, and record cache state. Observed startup enters the forecast. |
| A reservation's lease starts when it is recorded, so GPU queueing consumes the lease. | Addressed in the controller: reservations add an explicit queue allowance, workers refuse to start without their full working lease, and reported worker start times exclude queueing from the charge. |
| The harness identity covers every package module, including analysis and planning. A post-freeze code change blocks resumption and invalidates qualification. | Finish analysis/planning changes before the formal freeze. |
| The WikiText dataset is reported as CC BY-SA licensed. Prepared prompt text in a public bundle would carry attribution and share-alike terms. | Verify the pinned dataset card during preparation. Keep project code under MIT and data under its own terms. |

## Next external steps

1. Authenticate the personal Modal profile and check the actual rates/account usage. Establish the gross Workspace usage-budget backstop described in the megaplan before running cloud preparation.
2. Install the optional Modal client and run CPU image inspection:

   ```bash
   python -m pip install -e '.[modal]'
   modal setup
   modal run modal_app.py::preflight --config configs/benchmark.toml
   ```

   This builds/inspects two pinned images serially on CPUs. It costs CPU/build resources, records executable/package/help information, and leaves the qualification locks unchanged. The SDK/image combination has not been exercised here. Fix any flag reported missing from CLI help before GPU work; for example, the candidate vLLM `--disable-log-requests` flag may have changed in recent releases.

3. Inspect each engine interpreter and CLI. Build the same isolated Python 3.12 client environment in both images, resolve full transitive dependency locks with hashes, and record the derived Modal image IDs. Keep native engine dependencies intact.
4. Set up the upstream access Secret in Modal. Resolve model/dataset commits, inspect licensing/configuration, download and hash the snapshot in CPU preparation, and verify the real tokenizer, template, stop IDs, prepared prompts, and conformance probes.
5. Finish the Modal adapter around the tested controller: the backend's `spawn`/`poll`/`cancel`, one GPU function per engine calling `run_worker`, version-2 Volumes, read-only model/input mounts, the shared client interpreter, results commits, and exports. The controller already passes reservation IDs, deadlines, and replacement links through to the block, cell, and request records. Decide the compile-cache policy from the assessment. The resource contract is in `planned_gpu_options()`; no GPU functions are currently registered.
6. Count every allocation in the ledger, including retries, preemptions, startup, idle time, and failed attempts. Do not settle from one successful attempt's measurement duration. Record billing after each stage.
7. Run the bounded conformance pilot. Export it, qualify both engines, confirm the log/endpoint patterns, update the engine lock with passed reports, and use the local forecast to select the full or compact protocol before formal measurements.

The controller, worker entry, and ledger are integration components, not yet a one-command paid campaign launcher. The megaplan's `pilot`, `benchmark`, and `export` Modal entrypoints remain to be connected to them after these checks.

## Recovery and laptop handoff

Keep the committed source and locks on GitHub. Preserve the project ledger, billing exports, and artifacts separately; a new machine or billing month does not create a fresh budget allowance.

If the work moves without network access, create and verify a Git bundle:

```bash
git bundle create artifacts/handoff/inference.bundle main build/offline-benchmark
git bundle verify artifacts/handoff/inference.bundle
git clone --branch build/offline-benchmark artifacts/handoff/inference.bundle inference
```

Compare the bundle's SHA-256 after transfer. Pushing the branch remains the normal handoff when a connection is available.

Before resuming, resolve outstanding call IDs in Modal. An expired lease or disconnected laptop does not prove a container has stopped. Keep its conservative reservation until termination is confirmed. A stale local ledger `.lock` requires checking that the previous controller has stopped before removing it.

An interrupted cell cannot become valid merely because some request files exist. Incomplete or failed attempts remain visible in reports. A replacement is a new attempt, with at most one replacement per formal cell; aggregation selects the first valid attempt.

Inspect the copied artifacts and ledger without contacting Modal:

```bash
python -m inference_bench resume-plan \
  --input artifacts/campaign --ledger artifacts/budget-ledger.json
```

The campaign's top-level completion marker may be absent after an interruption. Every completed cell is still verified individually; corrupt sealed evidence stops recovery. A status file that says `complete` without the cell's completion marker remains interrupted. Missing attempt identities, duplicate attempts, a third attempt, and replacements without their original history are rejected.

The JSON distinguishes complete, pending, blocked, budget-stopped, and exhausted cells in the frozen schedule order. It retains the first valid result, assigns attempt 1 to unstarted cells and attempt 2 to failed/interrupted cells, and links each proposed replacement to its original artifact directory. Exhausted replacement budgets do not borrow from the formal stage.

Eligible cells are candidates for the controller, not submitted work. Each needs a new reservation and fresh process in a unique directory. The reported reservation maximum is the remaining ceiling, not a prediction that a cell will finish within it. Check that ceiling against the required block estimate and reserve against the latest ledger before each submission. The planner does not change files, settle calls, remove stale locks, or reset the budget. An expired lease remains unresolved until termination is confirmed externally.

The eventual emergency stop remains `modal app stop <app-id>`, followed by checking that active work ended and reconciling usage.
