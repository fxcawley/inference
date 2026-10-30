# Current implementation runbook

The local benchmark core is implemented and tested on Python 3.12 in Linux/WSL. The real engine images, model/tokenizer, dataset source, and Modal integration have not been exercised. There are no measured vLLM/SGLang results.

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

## Working commands

| Command | Current behavior |
|---|---|
| `plan [--json]` | Validate configuration; print matrix, schedule, token totals, cost bounds, and unresolved qualification items. No cloud calls. |
| `validate` | Validate local configuration structure and numeric constraints. |
| `validate --require-ready` | Also require immutable sources, client identity, qualified engines, the formal matrix, and a clean committed harness. Currently fails as intended. |
| `commands --engine vllm --model-path /models/snapshot` | Print candidate argv as a safely quoted command; never execute it. |
| `inspect-model --model-dir PATH` | Inspect local checkpoint metadata and shard inventory and compute hashes. This is not runtime qualification. |
| `prepare-prompts --model-dir PATH --corpus-jsonl FILE --output DIR` | Use the local snapshot's tokenizer to prepare exact-length prompts. Transformers is imported only for this command. |
| `example --output DIR` | Create a small, deterministic synthetic campaign for exercising reports. Refuse an existing directory. |
| `verify-artifacts --input DIR` | Require the completion marker, exact file set, sizes, and SHA-256 hashes. Reject symlinks and escaping paths. |
| `analyze --input DIR --output REPORT` | Recompute metrics from raw records and generate CSVs, three SVGs, tables, and claims. Output must be outside the sealed input. |
| `bundle --input DIR --output FILE.tar.gz` | Create a deterministic archive and SHA-256 receipt. Refuse to replace an existing archive. |

Synthetic analysis and bundling require `--allow-synthetic`. Every synthetic figure carries a visible label, and its claims are marked as nonpublishable measurements. Fixtures are useful for testing the pipeline; they are not model performance estimates.

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

## Implemented boundaries

`workloads.py` accepts a tokenizer object, preserves its role/template prefix and assistant suffix, and fills only the excerpt span to achieve the total input-token budget. It records deterministic source offsets and separates warmup and measured IDs. The real Gemma tokenizer and non-thinking template still need inspection.

`preparation.py` can resolve immutable Hub identities, download a model at an explicit commit, inventory shards, and prepare a supplied public corpus JSONL file. It does not infer dataset licensing or guess the actual WikiText file layout. The remaining corpus adapter should export the verified upstream format into `{"text": "..."}` JSONL rows and record its conversion/version.

`client.py` implements the common completions request and parser. A server must emit a finish reason, final authoritative usage, and DONE. The common token-ID/EOS/streaming contract has not yet been tested against either real engine.

`runner.py` owns a fresh server process group, validates the GPU, polls health/model identity, runs warmup and cells, and writes checksummed outcomes. An independent timer stops the process group on lease expiry, even if the asyncio loop is blocked. It reserves the final 30 seconds of the supplied lease for shutdown/artifacts. The remote adapter must still enforce the entire container lifetime and Volume commits.

`budget.py` provides a persistent single-controller ledger. An unresolved call retains its full reservation and blocks another GPU submission. Settlement requires explicit confirmation that work stopped; an overrun is charged and blocks further work. Changing the campaign or budget policy cannot silently reuse/reset an existing ledger. The lock file is local process coordination, not a distributed Volume lock.

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
5. Finish the Modal adapter: version-2 Volumes, read-only model/input mounts, the shared client interpreter, serial GPU calls, durable call IDs, absolute leases, per-cell commits, complete attempt history, and exports. The resource contract is in `planned_gpu_options()`; no GPU functions are currently registered.
6. Wire the native billing report to ledger reconciliation. Count every allocation, including retries, preemptions, startup, idle time, and failed attempts. Do not settle from one successful attempt's measurement duration.
7. Run the bounded conformance pilot: model identity, actual hardware/limits, BF16 execution, EOS behavior, token usage, natural-language sanity, warmup coverage, client capacity, and runtime/cost forecast. Freeze the full or symmetric compact protocol before formal measurements.

The existing `run_block` and ledger are integration components, not a one-command paid campaign launcher. The megaplan's `pilot`, `benchmark`, and `export` Modal entrypoints remain to be connected after these checks.

## Recovery and laptop handoff

Keep the committed source and locks on GitHub. Preserve the campaign's ledger and artifacts separately; a new machine or billing month does not create a fresh budget allowance.

Before resuming, resolve outstanding call IDs in Modal. An expired lease or disconnected laptop does not prove a container has stopped. Keep its conservative reservation until termination is confirmed. A stale local ledger `.lock` requires checking that the previous controller has stopped before removing it.

An interrupted cell cannot become valid merely because some request files exist. Incomplete or failed attempts remain visible in reports. A replacement is a new attempt, with at most one replacement per formal cell; aggregation selects the first valid attempt.

The eventual emergency stop remains `modal app stop <app-id>`, followed by checking that active work ended and reconciling usage.
