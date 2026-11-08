# Implemented measurement definitions

This document describes the local implementation. The [megaplan](megaplan.md) supplies the complete study design; the [runbook](runbook.md) records what remains unqualified.

## Inputs and control

The baseline configuration contains three input/output shapes (512/128, 4096/128, and 512/512), concurrency 1/4/16/32, 256 measured requests per cell, three fresh-process repetitions, and two engines. That is 72 cells and 18,432 requests, excluding warmup, pilot, and failures.

Each repetition uses a deterministic cell permutation shared by both engines. Engine order alternates vLLM/SGLang, SGLang/vLLM, vLLM/SGLang. Prompt IDs are identical across engines and concurrency levels for a given profile/repetition.

Corpus normalization applies Unicode NFC, collapses whitespace, and omits empty or heading-only rows. Tokenization happens once. A supplied chat template is rendered with thinking disabled and the assistant generation prefix retained. The generator inserts corpus token spans between the fixed template prefix/suffix, rejects special tokens in the excerpt, and checks the exact total input length. It does not decode and re-tokenize to estimate length.

The request uses integer prompt IDs, fixed output length, temperature 0, top-p 1, no penalties, one output, streaming, ignored EOS, and visible special tokens. Prefix reuse and speculative decoding are disabled by the candidate engine commands. Actual server behavior still requires the pilot.

Every input/output pair has the same length within a cell, and all workers start together. At C≥4 requests therefore run as synchronized waves: a wave prefills together, decodes together, and finishes together before the next wave starts. The results describe this burst-like closed-loop load, not staggered production arrivals. High-concurrency TTFT largely reflects how each engine prefills a full wave, including its released chunked-prefill and batch-token defaults. Report those resolved settings and this limitation beside the results.

The sampling-seed base is the first seven hexadecimal digits of a canonical SHA-256 hash of the global seed, profile, repetition, and phase. The client adds the request ordinal. Engine and concurrency do not enter that derivation.

## Conformance probes

Each block runs five untimed C=1 probes after readiness and before warmup: EOS honored, EOS ignored at a fixed 64 tokens, a simple arithmetic answer, a simple factual answer, and a plausible passage summary. Probes use the common client and request body. Only the EOS-honoring probes set `ignore_eos` false; they send stop IDs from the pinned model snapshot instead of relying on either server's generation defaults. A failed probe stops the block before warmup or measurement. The probes check basic serving behavior and token accounting; they are not an accuracy comparison.

## Pilot and protocol selection

The pilot has one repetition per engine and all twelve profile/concurrency pairs. Each cell uses `max(16, 2*C)` timed requests and the configured common warmup. Its prompt IDs and seeds are separate from formal measurement and formal warmup. The default pilot contains 768 timed and 672 warmup requests; conformance probes are additional.

Forecasting requires complete, valid pilot and warmup records for both engines, checked block timings, and the existing project ledger. For every retained profile/concurrency pair, use the slower engine's seconds per request and warmup duration. Scale measured time by the formal request count. Use the maximum observed startup and noncell overhead across engines; scale noncell overhead up with the total request volume, including warmup, and never down. Add 210 seconds per fresh block for container initialization and final shutdown/commit work outside the runner's timing. Apply 25% headroom to the whole block and round up to seconds.

Evaluate the full matrix, dropping C=32, retaining only `short`/`prefill`, and finally reducing to 200 requests per cell, in that order. Both engines and three repetitions remain mandatory. Check block, stage, campaign-time, and dollar limits before recommending a configuration. Formal measurements cannot trigger a later protocol reduction. If the smallest agreed study does not fit, stop after the pilot. Forecasts use finite-batch pilot observations and are estimates; absolute deadlines and the persistent ledger supply the execution bounds.

## Client and timing

HTTPX 0.28.1 supplies one common HTTP transport. The client connects directly to a loopback origin, disables environment proxies and redirects, pre-serializes request payloads, and permits at least the configured concurrent connections.

Exactly C workers draw from the ordered request list. Each sends another request after its previous request terminates. Per-request timing starts immediately before HTTP dispatch, after the worker has a slot. A total request deadline applies even if a server keeps sending partial data.

The SSE parser handles fragmented UTF-8, LF/CRLF framing, comments, multiline data, empty content, usage-only events, finish events, and DONE. Received chunks/events are never assumed to be individual model tokens.

Elapsed timestamps use `perf_counter_ns()`. Headers, first nonempty content, last content, DONE, and errors have separate timestamps. Multiple events delivered together may have identical receipt times.

The client consumes the HTTP response through completion so keep-alive connections return to the pool. E2E still ends at the recorded DONE event; the total request deadline also bounds any remaining transport cleanup.

## Metrics

| Field | Definition |
|---|---|
| `wall_time_s` | First measured dispatch to the last terminal response/error; includes ramp-up and drain. |
| `output_tokens_per_s` | Successful authoritative output tokens divided by the complete cell window. |
| `requests_per_s` | Successful requests divided by the complete cell window. |
| `ttft_ms` | First nonempty streamed content minus dispatch, in milliseconds. |
| `e2e_ms` | DONE minus dispatch, in milliseconds. |
| `tpot_ms_per_token` | Last minus first content, divided by actual output tokens minus one. Null for one token or a multi-token response delivered in one content event. |
| `failure_rate` | Failed/cancelled requests divided by attempted requests. Unattempted requests are separate. |

Server usage must be integer, authoritative, and final. Input counts must equal prepared lengths; output counts must equal the fixed target. Requested lengths, chunk counts, characters, and re-tokenized text never substitute for usage.

The first request failure stops further assignment; already active requests drain. There are no automatic request retries. Diagnostic throughput for an invalid cell uses the entire attempted window, but the cell is excluded from primary aggregation. Warmup, pilot, and measured records cannot be pooled.

Latency p50/p95 use linear interpolation within each repetition. Reports then show the median and minimum–maximum of the three repetition-level metrics. This range is run-to-run spread, not a confidence interval. No token-level ITL or p99 claims are produced.

Engine ratios are SGLang ÷ vLLM within each matched profile, concurrency, and repetition, using each side's first valid attempt. `paired-ratios.csv` lists every repetition's ratio and their median and range for output throughput, p50/p95 TTFT, and p95 E2E latency. Above 1 means higher SGLang throughput or higher SGLang latency. A repetition pair ran adjacently in separate allocations, not on one GPU, so the pairing reduces drift but does not remove host variation.

Equal memory-fraction settings do not imply equal KV capacity. At C=32, the 4,096/128 shape needs 135,168 concurrent tokens. Qualification records each engine's reported KV capacity when its log exposes it; cells where capacity may bind should be described as such.

## Artifact and analysis validity

Atomic files, SHA-256 manifests, and completion markers distinguish durable bundles from partial writes. A completion marker proves file integrity, not experimental success; each cell separately records valid/failed/interrupted status.

Analysis verifies the exact file set, protocol/source/image/harness identities, request phases and shapes, and stored summaries against recomputed raw metrics. The protocol includes a path-independent hash of the harness's Python source files; the worker checks its actual source before hardware inspection or model startup. Analysis can use a later implementation, but all input cells must identify the same frozen measurement harness. It rejects mixed campaigns, mismatched summaries, unqualified formal evidence, and missing completion evidence for purportedly successful cells.

All planned cells remain represented. Missing cells and failed attempts appear in `failures.csv`. Aggregation selects the first valid attempt, not the fastest, and the figures plot only groups with the complete repetition set. The third figure shows concurrency-labelled throughput/latency points with both horizontal and vertical run ranges.

Recovery can inspect an unsealed campaign, but it still validates every completed cell and rejects corrupt checksums. Unsealed cells remain invalid even when their raw files look complete. Each replacement must retain and link to attempt 1; no third attempt is permitted. Recovery proposals preserve the frozen schedule, require the same harness and existing budget history, and keep outstanding reservations until external termination is confirmed.

Synthetic fixtures require explicit analysis/export opt-in, carry visible labels in every figure, and cannot be promoted to measured results by changing a campaign label. Their arbitrary timings test the pipeline; they contain no information about engine speed.

## Cost interpretation

The proposed H200, eight-core, 128-GiB container costs $5.939856/hour at the dated planning rates. `aggregate.csv` derives GPU-only and requested-container dollars per million output tokens from those rates and observed throughput.

Those estimates are distinct from actual project expenditure. The latter remains null until reconciled with Modal, including preparation, startup, warmup, retries, failures, storage, and export. Ledger reconciliation records cumulative billed charges; billed amounts above list-price charges increase exposure, and lower bills never release allowance.

The current report path is CPU-only and uses the standard library for quantiles, CSV, and SVG output. It requires neither engine packages nor credentials.
