# Current implementation runbook

The local benchmark core, pilot forecast, and recovery planner are implemented and tested on Python 3.12 in Linux/WSL. The real engine images, model/tokenizer, dataset source, and Modal integration have not been exercised. There are no measured vLLM/SGLang results.

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

CPU tests cover actual fragmented HTTP streams, UTF-8 boundaries, concurrency, terminal usage, failures, total request timeouts, exact prompt lengths, metric units, first-valid replacement selection, checksums, synthetic-data labelling, process groups, and persistent budgets. The Linux process tests start and terminate short-lived dummy processes.

CI runs the same CPU tests and regenerates a synthetic report. It has no Modal credentials, model access, or GPU jobs.

For a pass that opens no network sockets, including loopback connections, use the already installed environment:

```bash
PYTHONPATH=src:tests python -m unittest \
  test_protocol test_metrics test_runner test_client.DecoderTests -v
```

The November 2 offline development pass also installed a Python audit hook that rejected socket connections, binds, datagram sends, and DNS resolution. The six existing HTTP integration tests were excluded from that pass. No packages or remote services were needed.

## Working commands

| Command | Current behavior |
|---|---|
| `plan [--json]` | Validate configuration; print matrix, schedule, token totals, cost bounds, and unresolved qualification items. No cloud calls. |
| `plan --phase pilot [--json]` | Print the paired, single-repetition pilot and its per-cell request/warmup counts. |
| `validate` | Validate local configuration structure and numeric constraints. |
| `validate --require-ready` | Also require immutable sources, client identity, qualified engines, the formal matrix, and a clean committed harness. Currently fails as intended. |
| `commands --engine vllm --model-path /models/snapshot` | Print candidate argv as a safely quoted command; never execute it. |
| `inspect-model --model-dir PATH` | Inspect local checkpoint metadata and shard inventory and compute hashes. This is not runtime qualification. |
| `prepare-prompts --model-dir PATH --corpus-jsonl FILE --output DIR` | Use the local snapshot's tokenizer to prepare exact-length prompts. Transformers is imported only for this command. |
| `example --output DIR [--phase pilot]` | Create a deterministic synthetic campaign for exercising reports or forecasts. Refuse an existing directory. |
| `forecast --input PILOT --ledger FILE` | Verify a complete pilot and existing ledger; print estimates for the full matrix and three ordered reductions. |
| `resume-plan --input CAMPAIGN --ledger FILE` | Inspect complete or interrupted formal artifacts and the existing ledger; print remaining work and blockers without writing either input. |
| `verify-artifacts --input DIR` | Require the completion marker, exact file set, sizes, and SHA-256 hashes. Reject symlinks and escaping paths. |
| `analyze --input DIR --output REPORT` | Recompute metrics from raw records and generate CSVs, three SVGs, tables, and claims. Output must be outside the sealed input. |
| `bundle --input DIR --output FILE.tar.gz` | Create a deterministic archive and SHA-256 receipt. Refuse to replace an existing archive. |

Synthetic analysis, bundling, forecasts, and recovery plans require `--allow-synthetic`. Every synthetic figure carries a visible label, and its claims are marked as nonpublishable measurements. Fixtures are useful for testing the pipeline; they are not model performance estimates.

## Offline example

```bash
python -m inference_bench example --output artifacts/example
python -m inference_bench verify-artifacts --input artifacts/example
python -m inference_bench analyze \
  --input artifacts/example --output artifacts/example-report --allow-synthetic
python -m inference_bench bundle \
  --input artifacts/example --output artifacts/example.tar.gz --allow-synthetic
```

Open `artifacts/example-report/figures/throughput.svg`, `ttft.svg`, and `latency-throughput.svg` in a browser. `aggregate.csv` includes run medians and ranges and derived compute-price estimates. Actual project cost remains null until billing reconciliation.

Use another output directory for a second example, or deliberately remove the old synthetic artifacts first. Generated data live under the ignored `artifacts/` directory.

## Pilot planning and cost forecasting

The pilot uses one fresh process per engine, all three profiles and all four concurrency levels, `max(16, 2*C)` timed requests per cell, and the common `max(8, 2*C)` warmup rule. C=4 and C=16 use the same small pilot rule so their timings are observed rather than interpolated. The default totals are 24 cells, 768 pilot requests, and 672 warmup requests. Separate conformance probes are additional work within the pilot budget.

Prompt generator version 2 writes `measured.jsonl.gz`, `warmup.jsonl.gz`, `pilot.jsonl.gz`, and `pilot-warmup.jsonl.gz`. Pilot input IDs and sampling seeds are separate from formal inputs. Both engines still use the same ordered inputs at each concurrency. Regenerate preparation artifacts when moving from generator version 1.

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

For an actual forecast, supply a verified pilot export and the current campaign ledger. Required evidence includes both complete engine blocks, every scheduled pilot cell, valid warmup records, and monotonic `timing.json` block timestamps. Every actual block/cell must identify its ledger reservation. Missing cells, wrong phases, corrupt checksums, inconsistent timings, and missing budget history stop the forecast.

The calculation uses the slower engine's seconds per request and warmup duration for each profile/concurrency pair. It adds the maximum observed startup duration. Observed noncell overhead, including serialization and artifact writes, is scaled up by the ratio of formal to pilot request counts, including warmup. It never scales that overhead down. Each new block also receives 210 seconds for container initialization and final shutdown/commit work that the local runner cannot time. The entire estimate then receives 25% timing headroom, rounded up to whole seconds per block.

The recommendation considers these choices in order, keeping both engines and three repetitions:

1. Full matrix: 72 cells, 256 requests per cell.
2. Drop C=32: 54 cells, 256 requests per cell.
3. Keep `short` and `prefill`: 36 cells, 256 requests per cell.
4. Use 200 requests per retained cell: 36 cells.

Each candidate includes its complete configuration and hash, estimated block/total duration, nominal cost, and budget failures. The forecast checks the block limit, formal-stage allowance, remaining campaign time, and remaining dollar allocation. Unresolved calls, overruns, a different harness, or any already-started formal/rerun reservation prevent a recommendation. If no candidate fits, stop after the pilot.

These are timing estimates, not billing guarantees or runtime qualification. Prices are read from the frozen configuration; forecasting does not refresh them. The remote controller must still freeze the selected protocol, enforce absolute leases, reserve each call, and reconcile actual usage. The smaller protocols must be selected before formal work starts.

## Implemented boundaries

`workloads.py` accepts a tokenizer object, preserves its role/template prefix and assistant suffix, and fills only the excerpt span to achieve the total input-token budget. It records deterministic source offsets and separates warmup and measured IDs. The real Gemma tokenizer and non-thinking template still need inspection.

`preparation.py` can resolve immutable Hub identities, download a model at an explicit commit, inventory shards, and prepare a supplied public corpus JSONL file. It does not infer dataset licensing or guess the actual WikiText file layout. The remaining corpus adapter should export the verified upstream format into `{"text": "..."}` JSONL rows and record its conversion/version.

`client.py` implements the common completions request and parser. A server must emit a finish reason, final authoritative usage, and DONE. The common token-ID/EOS/streaming contract has not yet been tested against either real engine.

`runner.py` owns a fresh server process group, validates the GPU, polls health/model identity, runs phase-specific warmup and cells, and writes checksummed outcomes plus monotonic startup/completion timings. It rejects cells outside the selected pilot/formal schedule. An independent timer stops the process group on lease expiry, even if the asyncio loop is blocked. It reserves the final 30 seconds of the supplied lease for shutdown/artifacts. The remote adapter must still enforce the entire container lifetime and Volume commits.

`budget.py` provides a persistent single-controller ledger. An unresolved call retains its full reservation and blocks another GPU submission. Settlement requires explicit confirmation that work stopped; an overrun is charged and blocks further work. Changing the campaign or budget policy cannot silently reuse/reset an existing ledger. The lock file is local process coordination, not a distributed Volume lock.

Ledger reads validate reservation identities, amounts, durations, statuses, and deadlines. Malformed or duplicate records cannot silently lower exposure. Planning requires an existing ledger file; a missing file never means zero historical usage.

## Next external steps

1. Authenticate the personal Modal profile and check the actual rates/account usage. Establish the gross Workspace usage-budget backstop described in the megaplan before running cloud preparation.
2. Install the optional Modal client and run CPU image inspection:

   ```bash
   python -m pip install -e '.[modal]'
   modal setup
   modal run modal_app.py::preflight --config configs/benchmark.toml
   ```

   This builds/inspects two pinned images serially on CPUs. It costs CPU/build resources, records executable/package/help information, and leaves the qualification locks unchanged. The SDK/image combination has not been exercised here.

3. Inspect each engine interpreter and CLI. Build the same isolated Python 3.12 client environment in both images, resolve full transitive dependency locks with hashes, and record the derived Modal image IDs. Keep native engine dependencies intact.
4. Set up the upstream access Secret in Modal. Resolve model/dataset commits, inspect licensing/configuration, download and hash the snapshot in CPU preparation, and verify the real tokenizer and prepared prompts.
5. Finish the Modal adapter: version-2 Volumes, read-only model/input mounts, the shared client interpreter, serial GPU calls, durable call IDs, absolute leases, per-cell commits, complete attempt history, and exports. Include `reservation_id` in each block's context so it reaches its block, cell, and request records. Preserve `replacement_for` links for replacement cells. The resource contract is in `planned_gpu_options()`; no GPU functions are currently registered.
6. Wire the native billing report to ledger reconciliation. Count every allocation, including retries, preemptions, startup, idle time, and failed attempts. Do not settle from one successful attempt's measurement duration.
7. Run the bounded conformance pilot: model identity, actual hardware/limits, BF16 execution, EOS behavior, token usage, natural-language sanity, warmup coverage, and client capacity. Export the complete pilot and use the local forecast to select the full or symmetric compact protocol before formal measurements.

The existing `run_block` and ledger are integration components, not a one-command paid campaign launcher. The megaplan's `pilot`, `benchmark`, and `export` Modal entrypoints remain to be connected after these checks.

## Recovery and laptop handoff

Keep the committed source and locks on GitHub. Preserve the campaign's ledger and artifacts separately; a new machine or billing month does not create a fresh budget allowance.

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
