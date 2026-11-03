# Gemma 4 serving benchmark megaplan

Build a reproducible public comparison of vLLM and SGLang serving Gemma 4 26B-A4B on Modal, keep the first study within $75, and explain its measured throughput and latency tradeoffs in a short article on cawley.dev.

## 1. Status and decisions

**Planning date:** October 30, 2026. This file records the full project plan. The local harness, CPU tests, and offline report path have since been implemented; see [the current runbook](runbook.md) for working commands and remaining integration work. No real-model or GPU qualification has been performed.

Implementation choices for the offline stage: HTTPX 0.28.1 supplies the common streaming transport, standard-library `unittest` supplies CPU tests, and the analysis emits SVG directly. These replace the proposed aiohttp/pytest/Matplotlib dependencies while preserving the experimental contract. The hash-locked remote environments and Modal GPU orchestration remain external qualification work.

The November 2 offline continuation implemented the bounded pilot schedule and separate inputs, artifact-based cost forecasts for the four agreed protocol sizes, and read-only recovery planning with persistent budget/attempt checks. `planning.py` provides these local plans; it does not dispatch remote calls. See the runbook for commands, evidence requirements, and the forecast's overhead assumptions.

The project decisions are:

| Question | Decision |
|---|---|
| Initial milestone | Finish the megaplan and publish the handoff; implement from it later. |
| Main objective | Serving tradeoffs: throughput, first-token latency, response latency, and cost as concurrency changes. |
| Comparison style | A controlled baseline; defer optimization and tuning studies. |
| Total first-study budget | Up to **$75**, including setup, failures, and reruns. |
| Model source | Official upstream weights downloaded inside Modal. |
| Accounts | The maintainer has a personal Modal account and upstream model access. Reproducers need their own access. |
| Timed inputs | Public text with controlled token lengths. |
| Article | Approximately **1,200–1,800 words**, three main figures, and detailed methods in the repository. |
| Development machine | A small client environment on Linux/WSL; use another machine if local installation or ordinary connectivity is restricted. |

Implementation recommendations established by the research:

- Model: `google/gemma-4-26B-A4B-it`, standard unquantized checkpoint, pinned to an immutable upstream revision before running.
- Hardware: **one NVIDIA H200 SXM, 141 GB**, tensor parallelism 1. Verify the actual allocation for every invocation.
- Candidate engine releases: **vLLM 0.32.0** and **SGLang 0.5.22**, using their official CUDA 13 images by digest.
- Workload: a fixed, public WikiText corpus snapshot, pretokenized once with the model's tokenizer, sent through one common streaming completions client.
- Timed comparison: three input/output shapes × four concurrency levels × three repetitions × two engines.
- Measurement location: the client and server run in the same Modal GPU container and communicate over loopback.
- Output: a small Python repository, complete raw measurements and provenance, deterministic analysis, a methods document, and a reviewable article.

The study should answer:

> On a single H200, how do these two released serving stacks trade output throughput against client-observed first-response and completion latency when serving the same Gemma checkpoint under a controlled, warmed, fixed-concurrency workload?

This wording matters. The experiment compares two complete serving stacks on the same GPU **model** and resource allocation. Separate Modal allocations do not establish that both ran on the same physical GPU or host.

## 2. Scope and completion standard

### 2.1 Included

1. The public repository at [fxcawley/inference](https://github.com/fxcawley/inference).
2. A portable controller that requires no local CUDA, PyTorch, vLLM, SGLang, Docker daemon, or model weights.
3. Separate remote engine images, a shared measurement client, and CPU-only preparation and analysis paths.
4. An immutable model/dataset/image manifest and a frozen experiment specification.
5. A pilot that establishes compatibility, correct request accounting, adequate client capacity, and an affordable run schedule.
6. A bounded benchmark campaign, including failures and actual resource use.
7. Public results that allow regeneration of tables and figures without a GPU or model access.
8. A short article and an eventual content-only integration into the existing site.

### 2.2 Deferred

- Quantization, FP8 KV caches, speculative decoding/MTP, LoRA, expert parallelism, multiple GPUs, and multiple model families.
- A prefix-cache comparison or workload with deliberate cross-request prefix reuse.
- A search over kernels, schedulers, batch sizes, or engine-specific optimizations.
- Multimodal requests, tool calls, structured outputs, multi-turn agents, or RAG retrieval.
- An accuracy leaderboard, a claim of equal model quality, or a full evaluation suite.
- Internet-facing endpoints, autoscaling comparisons, production arrival traces, open-loop SLO capacity, or cold-start rankings.
- Profiler-based explanations of why one runtime is faster.
- A dashboard, database service, distributed experiment scheduler, or general benchmarking framework.

The short article can mention these limits. They do not need implementations in the first repository.

### 2.3 Intended acceptance criteria

The full implementation is complete when:

1. A fresh WSL/Linux checkout can install the small controller, validate a configuration, and print a run plan without allocating a GPU.
2. Both pinned engine images pass the same real-model conformance and sanity checks on the chosen hardware.
3. Every scheduled measurement has a durable status and its relevant raw records; incomplete or failed cells remain visible.
4. The primary comparison uses the same model revision, input IDs, output-token target, precision, hardware SKU, request parameters, and measurement implementation.
5. Each reported primary cell has three valid repetitions and 256 measured requests per repetition, or uses the explicitly frozen compact protocol described below.
6. Analysis can regenerate the published numbers and three figures from the downloaded artifact bundle, offline and without GPU packages.
7. The article identifies the experimental limits and links to the code, protocol, configurations, and raw data.
8. Gross project usage stays within $75, with both the estimated allocation and reconciled Modal charges recorded.
9. The site integration builds using its existing workflow and is ready for review before publication.

The planning deliverable is complete when this file specifies that implementation, identifies the exact existing site integration, and distinguishes verified facts from runtime checks still to perform.

## 3. What the research established

### 3.1 Implementation starting point

| Finding | Consequence |
|---|---|
| The project started with an empty directory and no existing benchmark harness. | There is no earlier benchmark code or environment to preserve. Git initialization and the documentation handoff are now complete. |
| Python 3.12 on Ubuntu WSL was available during planning. | Linux/WSL with Python 3.12 is the first controller environment to support. |
| The inspected environment did not have Modal or uv installed. | Include installation of the small client; do not assume a prepared environment. |
| Direct Hugging Face access was unavailable on the original development network. | Resolve and download model/dataset content inside the personal Modal workspace. Do not make local Hub access a prerequisite. |
| Official Google, Modal, engine documentation, GitHub release metadata, and Docker Hub image metadata were reachable through ordinary HTTPS requests. | Documentation and candidate image identities could be researched without installing packages. |
| Some upstream Python source URLs were unavailable. | Exact installed CLI and OpenAPI behavior must be verified in the pinned containers before measurement. |
| The engines have different dependency requirements. | Use independent official engine images rather than borrowing an existing experimental environment. |

The available development environment is evidence of a possible controller path, not proof that Modal authentication and its full RPC connection will work on every network. If ordinary installation or connection fails, move the public project checkout to another machine. No network-policy workarounds are part of this project.

### 3.2 Site data flow and exact integration

The site repository is [fxcawley/site](https://github.com/fxcawley/site). Its inspected baseline was commit `47cb60f2f7164680f3e7c8f672416220b95603f8`.

The existing flow is:

```text
content/posts/<slug>/index.md
    -> src/lib/content.ts
       reads frontmatter and content; derives update dates from Git
    -> src/app/posts/[slug]/page.tsx
       generates static routes and metadata
    -> src/components/MdxContent.tsx
       renders Markdown/MDX, GFM tables, code, and math
    -> next build
    -> out/
    -> .github/workflows/deploy.yml
       publishes GitHub Pages when master is pushed
```

Relevant inspected files:

- [Content discovery and date handling](https://github.com/fxcawley/site/blob/47cb60f2f7164680f3e7c8f672416220b95603f8/src/lib/content.ts#L71-L129).
- [Markdown renderer](https://github.com/fxcawley/site/blob/47cb60f2f7164680f3e7c8f672416220b95603f8/src/components/MdxContent.tsx).
- [Post route](https://github.com/fxcawley/site/blob/47cb60f2f7164680f3e7c8f672416220b95603f8/src/app/posts/%5Bslug%5D/page.tsx).
- [Static export](https://github.com/fxcawley/site/blob/47cb60f2f7164680f3e7c8f672416220b95603f8/next.config.js).
- [Deployment and full-history checkout](https://github.com/fxcawley/site/blob/47cb60f2f7164680f3e7c8f672416220b95603f8/.github/workflows/deploy.yml).
- [Writing and repository conventions](https://github.com/fxcawley/site/blob/47cb60f2f7164680f3e7c8f672416220b95603f8/AGENTS.md).

There is **no draft filter** in the inspected content loader. A `draft: true` field would not prevent publication. Keep unfinished prose in this repository's `report/article.md`.

For eventual integration, use a separate site branch/worktree and add only the post and its generated assets. Recheck the current site instructions and content pipeline at that point.

### 3.3 Model and engine support

- Google's Gemma overview identifies the 26B-A4B variant as an MoE model. Four billion active parameters do not imply four-billion-parameter weight residency.
- Google gives an approximate BF16 loading requirement of **57.7 GB**, with additional serving/KV-cache requirements. This is planning context, not measured memory use.
- vLLM's supported-model documentation lists Gemma 4 architectures.
- The SGLang 0.5.22 cookbook explicitly lists `google/gemma-4-26B-A4B-it` and a **single-H200, TP=1** configuration. Its H200 command generator uses `--mem-fraction-static 0.85`.
- The same cookbook describes thinking as opt-in and documents `enable_thinking`. The benchmark will explicitly prepare a non-thinking prompt format and omit reasoning/tool parsers.
- Both current stable release pages identify official CUDA 13 images. Modal currently documents a CUDA 13.3 driver API, so the proposed runtime generation is compatible in principle.
- SGLang's cookbook also contains historical instructions to install from `main`. Do not copy those unpinned instructions; qualify the released image itself.

**Unresolved before execution:** the exact Hugging Face commit, the checkpoint's downloaded configuration and license, tokenizer/template hashes, actual packages inside the images, and real GPU loading/streaming behavior. Qualification will use the official upstream snapshot.

### 3.4 Candidate immutable image identities

These identities were read from Docker Hub on October 30, 2026. Use the Linux `amd64` manifest digest for the proposed Modal images.

| Engine | Release tag | Linux amd64 digest |
|---|---|---|
| vLLM | `vllm/vllm-openai:v0.32.0` | `sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9` |
| SGLang | `lmsysorg/sglang:v0.5.22` | `sha256:5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e` |

The corresponding multi-architecture index digests were:

```text
vLLM:  sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
SGLang: sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae
```

Record both the upstream digest and the final Modal image ID after adding the harness. The image tag, a Git release tag, and the contents actually installed in a container are separate evidence. Inspect installed versions during the CPU preflight.

The Modal Python package available during planning was **1.6.3** and supports Python 3.12. Start the controller lock with that version. Resolve and commit the remaining CPU/client dependency locks during implementation.

### 3.5 Modal details that affect validity

1. `gpu="H100"` may be upgraded to an H200 without changing the price. `gpu="H100!"` opts out. The proposed experiment requests H200 explicitly and still validates the returned GPU.
2. Region selection currently adds **1.15×** for broad regions or **1.75×** for narrow regions. Do not set `region` for the baseline; record the actual region instead.
3. Loading and idle container time are billable. The default idle window is 60 seconds.
4. CPU and RAM are charged at the larger of the request and usage. Explicit limits and fixed allocations make the estimate more useful.
5. `single_use_containers=True` is available and terminates a container after one invocation. Use it for experiment blocks.
6. `retries=0` does **not** eliminate platform crash/preemption rescheduling. Functions can restart with the same input.
7. A function's timeout applies to an execution attempt and can restart on retry. An absolute deadline in the input is needed to bound a rescheduled block.
8. Volumes require explicit commit/reload handling. Concurrent writes to the same file are unsafe; Volumes are not a distributed lock service.
9. Volumes v2 are recommended, but v1 is still the default unless version 2 is requested.
10. Modal's native Workspace usage budget is a gross-usage cap; its spend limit is a separate after-credit cap. Environment budgets require a paid plan.

## 4. Experimental protocol

### 4.1 Primary matrix

| Profile | Total input tokens | Generated output tokens | Purpose |
|---|---:|---:|---|
| `short` | 512 | 128 | A compact request with relatively little prefill work. |
| `prefill` | 4,096 | 128 | More prefill work with the same decode target. |
| `decode` | 512 | 512 | More decode work with the same input target as `short`. |

For each profile:

- Client concurrency: **1, 4, 16, 32**.
- Measured requests: **256 per cell**.
- Independent engine-process repetitions: **3**.
- Engines: **2**.
- Maximum model context: **8,192 tokens**.

This creates:

- **72 measured cells**.
- **18,432 measured requests**.
- **31,457,280 input tokens**.
- **4,718,592 requested output tokens**.

Warmups, sanity checks, pilot requests, and failed attempts are additional and belong in the cost ledger.

A cell is one `(engine, profile, concurrency, repetition, protocol_hash)` tuple. A repetition is a fresh server process, preferably in a fresh single-use Modal container.

### 4.2 Sampling and serving controls

| Dimension | Required setting |
|---|---|
| Checkpoint | One official model repository and immutable commit, same files for both engines. |
| Weight/activation precision | Explicit BF16; reject a quantized checkpoint or silent quantization. |
| KV precision | BF16, or `auto` only after confirming that it resolves to BF16. |
| GPUs / parallelism | One H200; TP=1, no CPU weight offload. |
| CPU / host RAM | 8 physical cores and 128 GiB, with the same explicit limits. |
| Temperature / top-p | 0 and 1. |
| Other penalties | Presence/frequency 0; repetition 1. |
| Number of outputs | `n=1`; no beam search or best-of expansion. |
| Output length | Fixed target, `ignore_eos=true`, no stop strings. |
| Streaming | Enabled; stream interval 1 where exposed. |
| Prefix reuse | Disabled in both engines. |
| Speculation | Disabled; no assistant/draft model. |
| Thinking/tool processing | Non-thinking input template; no reasoning/tool parsers. |
| Model-provided sampling defaults | Disable them; send explicit common sampling parameters. |
| Kernels and scheduling | Released engine defaults except the documented controls above. |
| CUDA graphs | Keep normal engine behavior; do not use an eager-mode startup shortcut. |
| Maximum active sequences | 32 in both engines, matching the largest client concurrency. |

The two memory-fraction options have different implementations. Giving each the numeric value 0.85 is a resource configuration choice, **not proof of equal KV capacity or workspace reservation**. Record each engine's resolved cache allocation and memory behavior.

Likewise, a vLLM batch-token budget and an SGLang prefill chunk size are not interchangeable knobs. Leave them at their released, model-resolved settings and record them. This is a controlled deployment baseline, not a claim of fully optimized engines or byte-for-byte identical internal work.

### 4.3 Public corpus and prompt preparation

Use the proposed source:

```text
Hugging Face dataset: Salesforce/wikitext
Configuration:        wikitext-2-raw-v1
Split:                train
```

Its current immutable revision and dataset-card licensing were not retrievable from the original development environment. Resolve them from the ordinary upstream service inside the CPU preparation function. The source lock must contain the actual revision, file hashes, attribution, and license before any GPU run. Do not label an uninspected license as verified.

Preparation:

1. Resolve the model and dataset to immutable commits; download from those commits.
2. Save the original dataset card and the selected source file hashes.
3. Read rows in their original order; retain nonempty prose and record a deterministic normalization rule. Exclude heading-only rows from the sampled prose pool.
4. Tokenize the prose once with the exact model tokenizer. Keep source row/article offsets so each excerpt can be reconstructed.
5. For each repetition and profile, draw 256 deterministic windows from the prose using a recorded seed. Overlapping source windows are allowed and disclosed.
6. Put the excerpt inside a simple, fixed instruction, for example: “Summarize and explain the following passage in detail.”
7. Render the pinned Gemma template once with `add_generation_prompt=True` and `enable_thinking=False`.
8. Enforce the total **input** budget including BOS, role delimiters, the instruction, the excerpt, and the assistant prefix. Adjust only the excerpt token span; never truncate the template suffix or assistant prefix.
9. Save the final integer prompt IDs and counts. Send those IDs directly to both completion endpoints.
10. Use separate IDs/windows for warmup, conformance, and measurement. The measured 256 prompts for a profile/repetition are reused at every concurrency and in both engines.

Implementation detail: token-level assembly around a verified template prefix/suffix is acceptable for exact lengths, but the assembled sequence must be checked for valid role boundaries and a single intended BOS. Do not decode and re-encode a fixed-length token list and assume its length survived.

Preparation must be deterministic from the source lock, generator version, and seed. Store the actual prepared prompts as evidence as well; a seed by itself is not a durable data format.

This choice measures generation over public-language inputs with controlled lengths. It excludes input tokenization and chat-template rendering from the timed path. It also forces output length beyond any natural EOS. State both limits in the article.

### 4.4 Load model and timing window

Use a **closed-loop concurrency** test:

- Create exactly `C` client workers.
- Workers take the next request from a prebuilt ordered list.
- Each worker sends its next request after its previous request terminates.
- There is no artificial think time or arrival-rate controller.
- A request timer starts after the worker has acquired its slot, immediately before the HTTP operation.
- Do not include waiting for a client worker in per-request latency.
- Include server queueing, API handling, prefill, decode, and loopback transport after dispatch.

The client serializes payloads and loads the prompt list before the measurement window. The HTTP connection pool must support at least `C` simultaneous connections without introducing a hidden lower concurrency limit.

For a successful cell, wall time runs from the first measured dispatch through the last complete measured response. It therefore includes batch ramp-up and drain. Call the result warmed **finite-batch throughput**; do not silently substitute a best interior interval.

The maximum-concurrency case has eight request waves with 256 requests. Publish the request count and timing definition so that this finite-window effect is apparent.

### 4.5 Warmup and trial ordering

- Start the server, wait for health plus a successful generation, and capture its resolved settings.
- Before each cell, send `max(8, 2*C)` warmup requests of the same input/output shape. Exclude them from measured records and retain their status/timing separately.
- Use the pilot to confirm that this warmup covers compilation and graph capture for the final matrix.
- If the pilot shows that more warmup is needed, change the common rule before freezing the protocol.
- Do not repeatedly warm or rerun until a favorable number appears.
- Drain all requests before moving to another cell.

Run the engines serially, with alternating order by repetition:

```text
repetition 1: vLLM, SGLang
repetition 2: SGLang, vLLM
repetition 3: vLLM, SGLang
```

Choose this schedule before measurement and retain it even if early results are surprising. Within a repetition, use a seeded permutation of the 12 profile/concurrency cells and give both engines the same cell order.

Do not run both engines simultaneously on separate GPUs. The experiment needs one active GPU allocation at a time, and serial execution simplifies the cost bound.

### 4.6 Pilot, freeze, and compact protocol

The pilot is a qualification and time-estimation step, not part of the reported primary data:

1. Qualify both engines on the same checkpoint and hardware.
2. Exercise all three profiles at C=1 and C=32.
3. Use `max(16, 2*C)` measured pilot requests, plus the defined warmup.
4. Exercise C=4 and C=16 briefly to check graph/shape transitions.
5. Measure startup, warmup, cell durations, failures, and client event-loop lag.
6. Forecast the full matrix from the slower observed engine, adding 25% timing headroom and all startup/warmup allowances.
7. Freeze the configuration, prompt files, engine settings, prices, schedule, and protocol hash before formal runs.

If the full protocol does not fit the formal-run allocation, reduce scope symmetrically in this order:

1. Drop C=32 for both engines and all profiles.
2. If still necessary, retain only `short` and `prefill`, both engines, C=1/4/16.
3. If needed, reduce the measured request count to **200** for every retained cell.

Keep three repetitions. Do not reduce only the slower engine's work, select workloads based on which engine wins, or lower the sample count halfway through a published cell.

The smallest publishable planned study is therefore two profiles × three concurrency levels × three repetitions × two engines, with at least 200 requests per cell. If even that does not fit, stop with the pilot results and an explanation rather than quietly changing the claim.

## 5. Common client and metric definitions

### 5.1 One measurement implementation

The implemented engine-independent client uses HTTPX 0.28.1 and a streaming SSE parser. Use the same client source, Python version, package lock, connection settings, request bodies, and timing code in both images.

Do not take vLLM's benchmark summary for one engine and SGLang's summary for the other. Their official benchmark tools are useful references, but combining their independently calculated metrics would introduce an avoidable difference.

Keep the new client narrow: `/v1/completions`, one output, streaming, fixed concurrency, supplied prompt IDs, per-request traces, and deterministic aggregation. It does not need arbitrary backend plugins or rate schedules.

### 5.2 Common request contract

The intended HTTP body is:

```json
{
  "model": "gemma4-bench",
  "prompt": [2, 123, 456],
  "max_tokens": 128,
  "temperature": 0.0,
  "top_p": 1.0,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "repetition_penalty": 1.0,
  "seed": 20261030,
  "n": 1,
  "echo": false,
  "stop": [],
  "ignore_eos": true,
  "skip_special_tokens": false,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

The three example prompt IDs are illustrative, not a valid Gemma prompt or a test fixture. Real IDs come from the prepared dataset.

Important behavior:

- Extra fields are top-level JSON fields when using HTTP directly; `extra_body` is an OpenAI SDK convenience, not a field to send literally.
- `max_tokens` changes by profile.
- The seed changes by the frozen repetition/request convention.
- Raw completions do not perform chat-template rendering. The saved IDs already contain the intended prompt format.
- The protocol requests visible special tokens so token work is not hidden merely by special-token filtering. This does not guarantee one token per SSE event.
- No log probabilities, expert-routing traces, or hidden states are requested during timing.

Validate support against each installed server's actual schema and behavior. A server accepting a field without applying it is not sufficient.

### 5.3 Conformance gate

Before a timed campaign, prove for both engines:

1. `/v1/models` identifies the intended served model.
2. Integer prompt IDs are accepted, and reported input usage equals the prepared count.
3. A stream terminates correctly and supplies authoritative completion-token usage.
4. A short-output probe demonstrates early stopping when EOS is honored and reaches the fixed target when EOS is ignored.
5. The test client records the first actual content event, rather than headers, an empty delta, or a usage-only event.
6. Output length equals the requested count for every profile in the pilot.
7. Normal natural-language sanity prompts produce plausible responses with EOS enabled.
8. Prefix caching, speculative decoding, precision, context limit, and active-sequence settings match the frozen configuration.

If a field is unsupported, change and document the adapter/protocol before formal measurement. Do not substitute the number of received chunks for token usage or silently change one engine to a native endpoint.

### 5.4 SSE parsing and timing

The parser must handle:

- Arbitrary TCP fragmentation, including splits inside UTF-8 characters and JSON strings.
- Multiple events arriving in one read.
- LF and CRLF framing, blank separators, comments/keepalives, and multi-line `data:` payloads.
- Initial empty content, usage-only events, finish events, and `[DONE]`.
- Non-200 responses, malformed JSON, incomplete streams, client cancellation, and timeouts.

Use `time.perf_counter_ns()` for elapsed times. Record a UTC start timestamp for provenance, but do not calculate latencies by subtracting wall clocks on different machines.

Record at least:

```text
t_dispatch        before starting the HTTP operation
t_headers         response headers received
t_first_content   first nonempty completion text received
t_last_content    last nonempty completion text received
t_done            valid terminal stream marker received
t_error           terminal failure, if any
```

Record event receipt times at the read/parser boundary, without synchronous console logging or compression in that path. Events delivered together may share a receipt timestamp; do not invent sub-event timings.

An SSE event is not necessarily one model token. Preserve event timing, but do not publish an ITL percentile derived by treating chunks as tokens.

### 5.5 Metrics

Let `O_i` be the server-reported generated-token count for a successful request and `D` the measured cell wall time.

Keep raw timestamps as integer nanoseconds. Convert `D` to seconds for throughput and elapsed request times to milliseconds for published latency columns. Use unit-bearing names such as `output_tokens_per_s`, `ttft_ms`, `e2e_ms`, and `tpot_ms_per_token`; do not leave unit conversion implicit.

| Metric | Definition |
|---|---|
| Output throughput | `sum(O_i) / D`, using successful requests and the complete cell window. |
| Request throughput | Successful requests divided by `D`. |
| TTFT | `t_first_content - t_dispatch`; explicitly client-observed time to first streamed text. |
| End-to-end latency | `t_done - t_dispatch`. |
| TPOT estimate | `(t_last_content - t_first_content) / (O_i - 1)` when `O_i > 1` and timing exists. |
| Failure rate | Failed measured requests divided by attempted measured requests; report unattempted requests separately. |
| GPU memory | Sampled allocated/device-used memory, with sampler and interval identified. |
| GPU utilization | Sampled utilization; diagnostic context rather than a standalone engine score. |

The TPOT estimate is secondary because detokenization and stream coalescing affect it. It is not a measured distribution of individual decoder-step durations.

If a multi-token response arrives in only one content event, TPOT is unobservable from this trace: store null with an explanation rather than reporting zero. TTFT and end-to-end latency can still describe that response's delivery.

For invalid cells, retain diagnostic successful-token throughput over the entire attempted window, but label the cell invalid and exclude it from primary speed ratios. Display its failure status beside valid cells.

Token accounting rules:

- Use final server usage for the throughput numerator.
- Require input usage to equal the prepared input length and output usage to equal the fixed target.
- Record actual counts even when they disagree.
- Do not use requested length, character count, whitespace words, chunk count, or re-tokenized output text as a replacement for missing authoritative usage.
- Optional returned token IDs can cross-check usage in the conformance run; do not add expensive per-token instrumentation to only one engine's timed run.

### 5.6 Repetitions and uncertainty

- Compute request-level p50 and p95 separately within each repetition, using a documented quantile method such as NumPy's `method="linear"`.
- At 256 requests, p95 is supported by only about 13 observations in the upper 5%; it is still a modest tail sample.
- Plot the median of the three repetition-level measurements and show their minimum–maximum range.
- Label that range as run-to-run spread, not a confidence interval.
- Do not pool all requests into a fictitious set of independent host-level trials.
- Do not headline p99, significance tests, or precise claims from three cloud allocations.
- Compute speed ratios per matched profile/concurrency/repetition and report all three plus their median when useful.
- Never require a statistically significant winner as an acceptance criterion.

## 6. Engine configuration and image isolation

### 6.1 Candidate launch commands

These are intended baseline commands. Check every flag against `--help` in the pinned image, capture the final argv, and qualify the resolved behavior before freezing.

vLLM:

```bash
vllm serve /models/<immutable-model-directory> \
  --served-model-name gemma4-bench \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --generation-config vllm \
  --disable-log-requests
```

SGLang:

```bash
sglang serve \
  --model-path /models/<immutable-model-directory> \
  --served-model-name gemma4-bench \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --tp 1 \
  --context-length 8192 \
  --max-running-requests 32 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --sampling-defaults openai \
  --stream-interval 1
```

If vLLM exposes a stream-interval setting in this image, set it to 1 as well; otherwise record and verify its default streaming behavior. Preserve useful startup warnings and error logs.

Use the complete official checkpoint with text-only requests. Do not extract its text tower or selectively disable different model components in the two engines. Record which components each native loader actually allocates.

### 6.2 Image construction

For each engine:

1. Use `modal.Image.from_registry("<repository>@<amd64-digest>")`.
2. Clear the serving image's default entrypoint with `.entrypoint([])` so the Modal worker owns startup.
3. Inspect the image's Python and engine executable paths and package versions.
4. Preserve the engine's installed dependency stack. Do not upgrade Torch/Transformers to make it match the other engine.
5. Install the common client in a separate virtual environment, using a pinned client lock and the same CPython 3.12 patch version in both images.
6. If the images do not already provide a common client interpreter, install a pinned standalone interpreter at an explicit client-only path. Do not change the engine executable's interpreter or package search path.
7. Add only this project's source, configuration, and dependency manifests to the image. Record their hashes.
8. Run CPU checks for imports, package identity, CLI arguments, and client startup before requesting a GPU.

Avoid blindly using `add_python` on an image that already contains an engine environment: changing the default interpreter can make its preinstalled packages disappear from the command's environment.

The controller may be Python 3.12 on WSL; the remote engine interpreter is whatever the pinned official image supplies and is recorded independently.

### 6.3 Model preparation

Use a CPU-only Modal function to:

- Resolve the official model commit and download the snapshot at that commit.
- Inspect `config.json`, `generation_config.json`, tokenizer files, chat template, model-card/license files, shard index, and quantization metadata.
- Confirm that the checkpoint is the intended standard BF16 model rather than a GGUF, quantized, QAT variant, or assistant model.
- Hash every downloaded model shard once during preparation; save file sizes and the shard-index mapping.
- Prepare the dataset and prompt IDs.
- Write a completion manifest only after every required file is present and verified.
- Commit the Volume before any GPU function starts.

Mount completed model/prompt data read-only in GPU functions using the Volume mount options. Put compiler caches and temporary files elsewhere. Set Hub/Transformers offline behavior in the serving process so an incomplete local snapshot fails instead of silently downloading another revision.

The Hugging Face credential belongs only in a Modal Secret used by preparation. Public artifacts contain model and dataset identities, not credentials or private artifact locations.

## 7. Modal execution design

### 7.1 Data flow

```text
Linux/WSL controller machine
  small Python controller + frozen config + budget ledger
       |
       | CPU calls
       v
Modal preparation function
  official model + public dataset -> verified read-only Volume
       |
       | one GPU invocation at a time
       v
Modal single-use GPU container
  engine subprocess <---- loopback HTTP ----> common client subprocess
         |                                     |
         +----------- logs / metrics ---------+
                            |
                       local scratch
                            |
                   close files, commit
                            v
                      results Volume
                            |
                  CPU analysis / export
                            v
  raw bundle + summaries + static figures + article draft
                            |
                   existing site content path
```

No public web endpoint or Modal HTTP gateway is involved in timing. This measures a local serving path inside the allocated cloud container, not end-user Internet latency.

### 7.2 Modal application layout

Use one `modal_app.py` with explicit functions/entrypoints for:

- `preflight`: CPU image checks and metadata inspection.
- `prepare`: CPU model/dataset download and prompt generation.
- `run_vllm_block`: one fresh vLLM server process and a supplied list of cells.
- `run_sglang_block`: equivalent SGLang execution.
- `analyze_remote`: optional CPU analysis for a machine that cannot install plotting dependencies.
- Local entrypoints `pilot`, `benchmark`, and `export` that coordinate the above serially.

Common GPU function options:

```python
gpu="H200"
cpu=(8.0, 8.0)
memory=(131072, 131072)  # MiB, 128 GiB
min_containers=0
max_containers=1
buffer_containers=0
single_use_containers=True
retries=0
timeout=7200           # per block, further bounded by its absolute deadline
startup_timeout=180
```

Do not set `region`, automatic GPU fallback lists, memory snapshots, or a persistent minimum replica count.

`max_containers=1` is per function, not an application-wide GPU mutex. The controller must still serialize both engines and refuse to start a new GPU call while an earlier call has an unresolved state.

### 7.3 Runner lifecycle

Implement the runner with explicit state transitions:

```text
reserved -> started -> hardware_checked -> server_starting
    -> ready -> conformance_passed -> warming -> measuring
    -> draining -> artifacts_committed -> complete
```

Any state can end in `failed`, `interrupted`, or `budget_stopped`, with a reason and whatever evidence was safely written.

For each invocation:

1. Read the absolute deadline and reservation ID before starting the engine. Exit promptly if the lease is already expired.
2. Write a start receipt with the Modal call/container identifiers and attempt ID.
3. Capture actual GPU, driver, CPU, memory limits, region, package versions, and source/config hashes.
4. Require one full H200, the expected memory class, and the intended resource settings. Reject a different GPU or a partitioned device.
5. Start the engine as an argv list using its recorded executable, with a new process group. Keep the parent lightweight.
6. Stream server stdout/stderr to bounded logs. Do not leave pipes unread.
7. Poll health and process liveness, with a 15-minute readiness limit inside the function execution budget.
8. Run the conformance/sanity subset appropriate to the phase.
9. Run the specified warmup and measurement cells.
10. Sample GPU/host/client diagnostics at a fixed interval, initially one second.
11. Close and checkpoint each cell's artifacts outside its timing window.
12. On completion or exception, terminate the entire engine process group; allow a short grace period, then kill remaining children.
13. Close all artifact writers, write final status and checksums, and explicitly commit the results Volume.
14. Return a small manifest reference to the controller.

All heavyweight engine imports and model loading occur after the deadline/hardware checks, in the child process. This reduces the chance of an import crash repeatedly allocating GPUs before the guard runs.

### 7.4 Request and block failures

- No automatic per-request retry during measurement.
- A malformed stream, wrong token count, engine exit, or timeout invalidates the cell.
- Stop assigning new work after the first such failure; drain or cancel the already active requests, recording both attempted and unattempted requests.
- Give each request a common frozen timeout, initially 300 seconds, and a short connection timeout.
- A block deadline terminates the child processes even if HTTP requests or engine initialization hang.
- The remaining valid cells may be run later under the frozen protocol if the budget allows.
- A corrected engine setting creates a new protocol version; never pool its timings with the previous configuration.

A maximum of one documented replacement attempt per invalid formal cell is the initial rerun policy. Keep the original failure, the reason for rerun, and the replacement link. Select the first valid attempt, not the fastest.

### 7.5 Preemption and resumption

Platform rescheduling must not turn one reservation into unlimited paid work:

- Every invocation receives a fixed UTC deadline plus a conservative maximum-dollar/time reservation.
- A restarted invocation receives the same deadline; it does not get a fresh allowance.
- Use unique attempt directories. A complete-cell marker is written only after its files are closed, hashed, and committed.
- A partial cell is never treated as complete.
- On controller restart, reload the ledger and resolve outstanding Modal call IDs before launching anything.
- If call state is uncertain, stop new submissions until the existing call is confirmed finished or explicitly cancelled.
- Do not rely on a Volume file as a cross-process lock. This project supports one active controller.
- A repeat that resumes in a different container is not a continuation of the same independent repetition. Start a new attempt/repetition record and apply the predefined replacement policy.

Use fresh engine processes for formal repetitions. If a block must be split because its estimated duration exceeds the block allowance, specify the split before launching; record the allocation identity for each resulting cell.

### 7.6 Artifact writes without measurement interference

Record timing events to memory or local scratch during a cell, with bounded buffering. Perform compression, checksumming, Volume copying, and commits after the cell finishes.

Server errors and a lightweight journal can be written continuously, but avoid committing or compressing the full trace while measuring another request. If a hard crash destroys an incomplete local trace, the cell is marked interrupted; preserving a partial trace is less important than pretending it is a valid measurement.

Use separate directories for every engine/block/attempt. All metadata shared across the campaign is written only by the controller.

## 8. Reproducibility and artifact contracts

### 8.1 Frozen source identities

`configs/sources.lock.json` must include:

- Model repository, immutable commit, file paths, sizes, and SHA-256 hashes.
- Tokenizer, tokenizer configuration, chat template, generation configuration, and model configuration hashes.
- Dataset repository/configuration/split, immutable commit, downloaded file hashes, attribution/license, and normalization rules.
- Prompt generator version, seed derivation, prompt-manifest hash, and counts.

`configs/engines.lock.json` must include:

- Engine names/releases, upstream image/index digests, platform, and final Modal image IDs.
- Installed engine, Torch, Transformers, tokenizer, Triton, FlashInfer and related package versions when present.
- CUDA runtime and actual driver information from qualification.
- Engine and client interpreter identities.
- Final launch argv and relevant non-secret environment variables.

No placeholder revision, `latest` tag, or unresolved lock is permitted in a formal-run configuration.

### 8.2 Run artifact tree

```text
artifacts/<campaign-id>/
  campaign.json
  protocol.toml
  sources.lock.json
  engines.lock.json
  pricing.json
  schedule.json
  budget-ledger.jsonl
  prompts/
    manifest.json
    measured.jsonl.gz
    warmup.jsonl.gz
    attribution.md
  runs/<engine>/<repetition>/<attempt-id>/
    manifest.json
    status.json
    environment.json
    server-command.json
    server.log
    gpu-host-samples.jsonl.gz
    cells/<profile>-c<concurrency>/
      status.json
      warmup.jsonl.gz
      requests.jsonl.gz
      stream-events.jsonl.gz
      summary.json
      checksums.json
  analysis/
    cells.csv
    aggregate.csv
    failures.csv
    claims.json
    figures/
      throughput.svg
      ttft.svg
      latency-throughput.svg
    tables.md
  checksums.json
```

The exact compression format can be gzip throughout; do not add a compression dependency unless size warrants it.

### 8.3 Minimum request record

```text
schema_version
campaign_id, protocol_hash
engine, profile, concurrency, repetition, attempt_id, cell_id
request_id, sequence_ordinal, prompt_hash, phase
expected_input_tokens, requested_output_tokens
dispatch_ns, headers_ns, first_content_ns, last_content_ns, done_ns, error_ns
http_status, finish_reason
actual_input_tokens, actual_output_tokens
content_event_count
status, error_category, error_message
response_text or a reference to its persisted stream
```

Null timing/usage values remain null with an error reason. Do not replace missing latency with zero.

### 8.4 Minimum environment record

- Harness Git commit and tracked-source/config checksum; formal runs require a clean source state.
- Base and derived image identity; installed package versions.
- Actual GPU name, memory capacity, compute capability, device count, driver, and MIG/partition status when available.
- A stable anonymized device identifier to notice repeated allocations without relying on it as a hardware guarantee.
- CPU model, requested/available cores, memory request/limit, and relevant cgroup data.
- Modal region/container/call identifiers and timestamps; do not include authentication material.
- Resolved engine arguments, attention/MoE backends, cache precision/capacity, graph settings, batching/chunking settings, and any model-specific overrides exposed by logs/configuration.
- Client interpreter/dependency versions, connection limits, and telemetry settings.

### 8.5 Public artifact strategy

- Track source, dependency locks, frozen small manifests, the final summary tables, figures, and methods in Git.
- Keep raw run directories, caches, and model weights out of Git.
- Publish a versioned compressed raw-data bundle with SHA-256 checksums, preferably attached to the public repository's release.
- Include prepared prompt data and its dataset attribution/license, or a complete reconstruction manifest if redistribution requirements favor reconstruction. Offline **analysis** must not depend on fetching the corpus.
- Include failed and interrupted attempt metadata. Remove secrets and irrelevant machine-specific local paths from exported logs through a narrow allowlist/export step.
- Keep private operational records, such as account billing identifiers, separate from the public cost summary.
- Model weights are downloaded by each reproducer under the model's own terms; they are not part of the result bundle.

The public repository is `https://github.com/fxcawley/inference`. The measured-results release tag and raw-data asset links will be established after execution. Do not put a guessed live release URL into the article.

## 9. Budget and runtime controls

### 9.1 Prices checked October 30, 2026

| Resource | List price |
|---|---:|
| H200 SXM | $0.001261 / GPU-second = **$4.5396/hour** |
| H100 SXM5 | $0.001097 / GPU-second = $3.9492/hour |
| CPU | $0.0000131 / physical-core-second |
| RAM | $0.00000222 / GiB-second |
| Volume storage | $0.09 / GiB-month, with a listed 1 TiB monthly free allowance |
| Network egress | $0.04 / GiB, with plan-specific included usage |

With the proposed eight cores and 128 GiB:

```text
GPU per hour:                  $4.539600
8 CPU cores per hour:          $0.377280
128 GiB RAM per hour:          $1.022976
Nominal GPU-container total:   $5.939856/hour
```

Do not count on free credits or free storage already being available. Treat $75 as the gross project usage budget. Show any credits separately from the experiment's resource cost.

### 9.2 Allocation

| Stage | GPU-container time ceiling | Nominal ceiling |
|---|---:|---:|
| Compatibility checks and pilot | 6,000 seconds total | $9.90 |
| Formal measurements, including startup/warmup | 24,000 seconds total | $39.60 |
| Documented replacement attempts | 6,000 seconds total | $9.90 |
| CPU preparation, image build overhead, analysis, storage, export | Separate allowance | $5.00 |
| Unused reserve for billing/timing variance | Reserve | $10.60 |
| **Total** | **10 GPU-container hours maximum** | **$75.00** |

These ceilings are project-wide totals across both engines and all attempts. They are not per-engine allowances.

The full matrix has not been timed. The pilot determines whether it fits the formal allocation; the allocation is not a runtime prediction.

### 9.3 Enforcement

Implement a simple single-controller budget ledger:

1. Capture current rates, account billing-cycle context, and current project usage before launch.
2. Before each remote call, reserve its worst allowed duration at the current configured rate, plus its fixed overhead allowance.
3. Reject submission if completed cost plus outstanding reservations plus the new reservation exceeds the stage or project limit.
4. Pass the reservation ID and absolute deadline into the GPU invocation.
5. Count startup, warmup, serialization/commit time, failed attempts, and rescheduled attempts.
6. Release unused reservation only after the call is confirmed stopped; a missing result retains the full conservative reservation.
7. Stop new work at the $60 GPU-container soft allocation or ten-hour ceiling. Keep the remaining money for CPU work and uncertainty.
8. Reconcile the ledger against tagged Modal billing reports after each stage and at campaign completion.

Tag the app/functions with a stable project/campaign identifier if the current SDK supports the documented billing attribution. Verify which tags appear in the billing report during the pilot.

Use the personal workspace's native **usage budget** as the outer backstop. For an otherwise unused workspace, $75 gross usage is the intended cap; for a workspace with existing monthly usage, the cap must reflect its already-used amount. Do not silently change a shared workspace-wide limit or assume an after-credit spend cap means the same thing.

The campaign ledger persists across midnight and month boundaries; a new billing month must not reset the experiment allowance. Platform usage reports may lag, so the controller relies on conservative reservations rather than live billing alone.

### 9.4 Reporting cost

Report three distinct quantities:

1. **Observed project cost:** preparation, all GPU attempts, analysis, and any billed storage/egress, reconciled with Modal.
2. **GPU-only dollars per million output tokens at the measured load:**
   `1_000_000 * GPU_price_per_second / output_tokens_per_second`.
3. **Nominal requested-container dollars per million output tokens:** the same formula with the stated GPU+CPU+RAM rate.

The latter two are derived compute-price estimates at the measured utilization. They do not include idle production capacity, networking to users, an API provider's margin, or a promise about an application's invoice.

## 10. Repository scaffold and responsibilities

The local package, tests, configuration, analysis, and handoff documentation are implemented. The tree below remains the blueprint for the complete study, including dependency locks and report files that still depend on external qualification and measured data. The current runbook records that boundary. Keep the module boundaries practical; do not introduce a plugin system.

```text
inference/
  README.md
  AGENTS.md
  LICENSE
  .gitignore
  .gitattributes
  pyproject.toml
  modal_app.py
  configs/
    benchmark.toml
    sources.lock.json
    engines.lock.json
  requirements/
    controller.lock.txt
    client.lock.txt
    prepare.lock.txt
    analysis.lock.txt
    dev.lock.txt
  src/inference_bench/
    __init__.py
    __main__.py
    config.py
    provenance.py
    workloads.py
    engines.py
    client.py
    metrics.py
    runner.py
    artifacts.py
    budget.py
    analysis.py
  tests/
    test_protocol.py
    test_client.py
    test_metrics.py
    test_runner.py
    fixtures/
      streams.json
      example_campaign/
  docs/
    megaplan.md
    methodology.md
    runbook.md
  report/
    article.md
    figures/
    tables/
  .github/workflows/ci.yml
```

| File/module | Responsibility |
|---|---|
| `README.md` | Research question, current result status, quickstart, reproduction paths, and eventual result links. |
| `AGENTS.md` | The benchmark invariants, artifact rules, explicit distinction between mocked tests and measured results, and the site voice reference. |
| `pyproject.toml` | Python 3.12 package, `inference-bench` CLI, CPU-oriented optional dependency groups, and optional Ruff configuration. |
| `requirements/*.lock.txt` | Exact transitive CPU dependency sets with hashes; GPU engines are pinned by images instead. |
| `configs/benchmark.toml` | Matrix, controls, request/warmup counts, seed, schedule, resource allocations, budget, and timeout policy. |
| `sources.lock.json` | Immutable model, dataset, template, and prepared-input identities. |
| `engines.lock.json` | Image and dependency identities plus qualified launch settings. |
| `modal_app.py` | Modal images, Volumes, functions, and serial stage orchestration; minimal local imports. |
| `config.py` | Strict parsing, validation, canonical serialization, and protocol hashing. |
| `provenance.py` | Source and environment manifests; hash and allowlist helpers. |
| `workloads.py` | Corpus normalization, exact-length prompts, seed handling, and saved prompt manifests. |
| `engines.py` | Two command builders and collection of resolved runtime settings. |
| `client.py` | Common HTTP/SSE client, bounded concurrency, event timestamps, and request outcomes. |
| `metrics.py` | Pure aggregation and metric definitions, independent of Modal and GPU libraries. |
| `runner.py` | Process lifecycle, readiness, deadlines, warmup, cell execution, and cleanup. |
| `artifacts.py` | Atomic local writes, checksums, committed completion markers, bundle validation/export. |
| `budget.py` | Price model, conservative reservations, ledger, and campaign/stage limits. |
| `analysis.py` | Validated CSV/JSON summaries, plots, tables, and machine-readable article claims. |
| `docs/megaplan.md` | The implementation contract and record of planning decisions. |
| `docs/methodology.md` | The frozen experimental method and metric definitions. |
| `docs/runbook.md` | Setup, machine handoff, cloud execution, failure recovery, costs, export, and cleanup. |
| `report/article.md` | Working draft outside the site content directory until measured claims and site integration are ready. |
| CI | CPU-only tests/lint and fixture-based analysis regeneration; no secrets or paid runs. |

The implementation uses standard-library `argparse`, `tomllib`, dataclasses, JSON, hashing, subprocess utilities, and `unittest`. HTTPX supplies the common HTTP client; Modal and Hub/tokenizer libraries remain separate integration dependencies. Analysis computes the documented linear quantiles and emits CSV/SVG without NumPy or Matplotlib.

Do not make Pandas, a notebook server, Docker, or a GPU engine a local requirement unless an implementation need justifies it. A standalone analysis CLI is enough.

Original project code and documentation use the MIT license. Dataset and model terms remain separate and retain their own provenance.

## 11. Ordered implementation work packages

### WP0 — Finish the project setup and thin controller

**Files:** repository metadata, `README.md`, `AGENTS.md`, `.gitignore`, `.gitattributes`, `pyproject.toml`, dependency lock files, initial `configs/benchmark.toml`.

The public repository, Python package, offline controller, and CPU tests are implemented. Complete dependency locks and actual Modal connectivity remain part of external qualification.

Actions:

1. Start an implementation branch from this repository.
2. Create a Linux/WSL Python 3.12 virtual environment.
3. Install only the locked controller/dev dependencies needed for this phase.
4. Verify the personal Modal profile and ordinary service connectivity without printing tokens.
5. Document the identical setup path on another Linux/WSL machine.
6. Implement configuration validation and `inference-bench plan`, which prints the matrix, bounds, and unresolved locks without making cloud calls.
7. Maintain ignore rules for environments, local credentials, caches, downloaded weights, raw run output, and temporary files.

**Done when:** a dry plan works locally; no GPU package is imported or installed by it; the controller machine either reaches Modal normally or has a precise handoff path to another machine.

### WP1 — Resolve sources and qualify container construction on CPUs

**Files:** `modal_app.py`, `provenance.py`, `engines.py`, both source/image locks, preparation dependencies.

Actions:

1. Build wrapper images from the recorded official digests.
2. Inspect engine versions, interpreter paths, CUDA libraries, CLI flags, and the common client environment.
3. Verify the available Modal driver generation against the image requirements.
4. Resolve official model and dataset revisions using upstream access in Modal.
5. Inspect the small model/tokenizer/dataset metadata before downloading the weight shards.
6. Download and hash the complete standard checkpoint in a CPU function.
7. Create version-2 model/input and result Volumes; complete and commit the preparation manifest.
8. Ensure only intended project files are uploaded as source.

**Done when:** both images have reproducible identities and usable commands; the read-only model snapshot is complete; model/dataset attribution is recorded; all later GPU work can operate without Hub downloads.

**Stop condition:** do not spend GPU time resolving package installation or model-access errors.

### WP2 — Build the shared workload and measurement path

**Files:** `workloads.py`, `client.py`, `metrics.py`, `artifacts.py`, protocol/client/metric tests and fixtures.

Actions:

1. Implement corpus preparation and exact total prompt lengths.
2. Save deterministic prompt manifests shared by both engines.
3. Implement the common request body and bounded-concurrency client.
4. Implement event-safe SSE parsing and raw timing records.
5. Implement pure metric aggregation and strict usage validation.
6. Exercise the path against a local fake streaming server using known timings and token counts.
7. Implement artifact checksums and explicit incomplete/complete status.

**Done when:** CPU tests demonstrate correct timing/counting on split/coalesced streams and failure cases; valid fixtures regenerate known metrics; missing usage never becomes a guessed token count.

### WP3 — Implement bounded Modal lifecycle and cost handling

**Files:** `runner.py`, `budget.py`, `modal_app.py`, runner/budget tests, runbook.

Actions:

1. Implement serial submissions with budget reservations and absolute deadlines.
2. Implement hardware validation, process-group ownership, readiness, warmup, and cleanup.
3. Implement one-second telemetry and client event-loop lag measurements.
4. Implement close/hash/commit between cells, unique attempt directories, and resume validation.
5. Handle ordinary failures, timeouts, platform rescheduling, and controller interruption.
6. Document native Workspace usage-budget setup and the emergency stop command.

**Done when:** CPU fake-process tests prove that deadlines, invalid hardware, server failure, and cancellation all prevent a false successful run and clean up the owned child processes.

### WP4 — Run the real-model compatibility pilot and freeze the protocol

**Files/artifacts:** qualification manifests, pilot artifacts, final configuration/locks, methodology and runbook updates.

Actions:

1. Capture rates and billing context and establish the $75 outer budget.
2. Launch one engine at a time on verified H200 hardware.
3. Execute the common conformance probes and natural-language sanity checks.
4. Run the small pilot and inspect actual token counts, streaming, dtypes, and model-specific settings.
5. Check that the client is not limiting throughput or delaying content events.
6. Forecast duration/cost using both engines and select the full or compact matrix.
7. Freeze the protocol, source inputs, request schedule, command lines, and client environment.
8. Commit the frozen configuration before formal measurements.

**Done when:** both engines qualify, the chosen matrix fits the remaining allocation with headroom, and there are no unresolved measurement semantics.

**If blocked:** produce a specific compatibility finding and a smaller next action. Do not silently switch models, precision, engines, or hardware.

### WP5 — Execute the formal campaign

**Artifacts:** all planned run/cell records, environment manifests, logs, telemetry, and ledger.

Actions:

1. Execute the predefined alternating engine order and cell schedule.
2. Keep one active GPU call at a time.
3. Validate and commit every cell before proceeding.
4. Stop on budget exhaustion, unresolved call state, or a protocol-breaking issue.
5. Use only the predefined replacement policy for invalid cells.
6. Reconcile the stage's billing and export a verified local result copy.

**Done when:** every planned cell has a final status, all primary cells selected for reporting meet the validity/repetition standard, and project usage remains inside the allowance.

Do not conceal a capacity failure at high concurrency. Report that limit and compare only the common valid region.

### WP6 — Analyze and package the evidence

**Files/artifacts:** `analysis.py`, analysis tests in existing test files, output tables/figures/claims, a versioned raw bundle.

Actions:

1. Validate source/protocol/image identities across the campaign.
2. Generate per-cell summaries and a complete failure table.
3. Aggregate repetition-level metrics using the predefined rules.
4. Produce three static figures and a compact cost table.
5. Generate `claims.json` so the article's numeric assertions can be checked against data.
6. Re-run analysis from the exported bundle without Modal credentials or network access.
7. Create the public artifact manifest and checksums.

**Done when:** every published number is traceable to raw records and reproducible locally with CPU dependencies alone.

### WP7 — Write the short article

**Files:** `report/article.md`, `docs/methodology.md`, final README result section.

Actions:

1. Write from the observed outcome, including a tie or workload-dependent result if that is what occurred.
2. Use the agreed neutral, measured site voice.
3. Explain the question, setup, three figures, cost, and main limits within approximately 1,200–1,800 words.
4. Link to detailed methodology rather than expanding the article into a setup manual.
5. Check each number against `claims.json` and avoid attributing causes without evidence.

**Done when:** the draft is readable without the repository, while the methods and data let a technical reader audit it.

### WP8 — Integrate with cawley.dev and publish the reproducible result

**Planned site additions:**

```text
content/posts/gemma4-vllm-sglang/index.md
public/posts/gemma4-vllm-sglang/throughput.svg
public/posts/gemma4-vllm-sglang/ttft.svg
public/posts/gemma4-vllm-sglang/latency-throughput.svg
public/posts/gemma4-vllm-sglang/summary.csv
```

Actions:

1. Publish the measured code/data release and verify artifact downloads/checksums.
2. Integrate the completed post and generated assets on an isolated site branch/worktree.
3. Use existing frontmatter and Markdown image/table rendering; no new React chart components are required.
4. Run the site's static build and inspect the post and assets in the exported output.
5. Review desktop/mobile readability and light/dark backgrounds.
6. Present the concrete article/site change for review before the master publication step.
7. After publication, verify the public post, code, raw-data, and figure links.

**Done when:** the public result has working reproduction links and the repository records its measured configuration and limitations.

## 12. Validation plan

### 12.1 CPU tests worth writing

This project needs tests because errors in timing, token accounting, failure handling, or cost control would invalidate the public result. Keep them in the four planned test modules rather than creating one test file per function.

| Area | Meaningful cases |
|---|---|
| Workload/configuration | Total token budgets include template overhead; malformed/unresolved locks rejected; identical seed/source gives identical prompt IDs; warmup and measurement IDs separated. |
| SSE client | Fragmented UTF-8/JSON; multiple events in one read; empty initial content; usage-only trailer; comments/CRLF; missing DONE; server error; timeout; content arriving after finish anomalies. |
| Timing/metrics | Known clock fixtures and explicit ns/ms/s conversion; TTFT ignores headers/empty chunks; finite-batch denominator includes drain; missing usage rejected; failures, one-token outputs, and a multi-token single-event response handled; p95 and repetition aggregation verified. |
| Concurrency | At most C requests active; pool does not silently serialize; no request retried; stopped work counted as unattempted. |
| Lifecycle | Child exit during readiness; hang; deadline expiry; exception during measurement; process-group cleanup; incomplete artifact cannot receive a valid completion marker. |
| Budget/resume | Reservation rejection; unresolved call retains cost; no allowance reset on retry/resume/month boundary; completed cells not duplicated; first valid replacement chosen rather than fastest. |
| Analysis | Fixture bundle regenerates expected tables; mismatched protocol/image/source identity fails validation; failed cells remain visible. |

Use a fake clock where appropriate and a small local fake streaming server for actual HTTP framing/concurrency tests. Avoid timing assertions tied to a CI machine's speed.

### 12.2 GPU qualification

The real-model pilot checks what mocks cannot:

- Correct model and tokenizer, complete checkpoint loading, BF16 execution, and appropriate hardware.
- Both endpoints' actual token-ID and streaming-usage behavior.
- EOS/length controls and non-thinking template behavior.
- Plausible generation on a small fixed set of arithmetic, short factual, and passage-summary prompts with normal EOS behavior.
- All chosen input/output shapes and concurrency transitions.
- No OOM, invalid memory access, NaNs in reported diagnostics, or repeated empty/garbled outputs.
- Warmup coverage and client capacity.

Greedy BF16 outputs need not match bitwise across engines. Preserve differences and investigate clear behavioral failures; do not claim an accuracy comparison from these sanity checks.

### 12.3 Client capacity check

Track event-loop scheduling lag and the client's CPU use during the pilot. A useful initial qualification threshold is p95 client loop lag below 2 ms, with no sustained CPU saturation or connection-pool waiting.

If that fails, profile the lightweight client first: remove synchronous output, buffering/compression, or accidental per-request setup. If necessary, increase the **common** CPU allocation or adjust the common client before the freeze and recalculate cost.

Do not assume a high server token rate proves that client-side TTFT is accurately measured.

### 12.4 Routine verification commands

After implementation:

```bash
python -m unittest discover -s tests -v
python -m inference_bench validate --config configs/benchmark.toml
python -m inference_bench plan --config configs/benchmark.toml
python -m inference_bench example --output artifacts/example
python -m inference_bench analyze --input artifacts/example --output artifacts/example-report --allow-synthetic
```

Run relevant tests while developing and the small complete CPU suite before the pilot/final release. Do not start GPU jobs from CI.

The site gate is `npm run build` on the integration worktree. Use `npm ci` there if dependencies need installation, matching the existing workflow's Node 20 setup.

## 13. Command-line runbook to implement

These commands describe the full intended study interface. Use [the current runbook](runbook.md) for the implemented commands. Hash-locked remote environments and paid campaign orchestration remain to be connected after external qualification.

### 13.1 Controller setup

On Linux or inside Ubuntu WSL:

```bash
git clone https://github.com/fxcawley/inference.git
cd inference
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements/controller.lock.txt
python -m pip install -e . --no-deps
modal setup
python -m inference_bench plan --config configs/benchmark.toml
```

Use the existing personal Modal profile. If already authenticated, validate it rather than creating another token unnecessarily.

If dependency installation or Modal's normal connection is unavailable, use another machine:

1. Clone the public project source and committed locks.
2. Create the same Python environment.
3. Authenticate to the same personal Modal workspace.
4. Continue from the committed configuration and cloud artifacts.

Neither path requires an existing local weight copy or local GPU packages.

### 13.2 Cloud phases

```bash
modal run modal_app.py::preflight --config configs/benchmark.toml
modal run modal_app.py::prepare --config configs/benchmark.toml
modal run modal_app.py::pilot --config configs/benchmark.toml
python -m inference_bench plan --config configs/benchmark.toml
modal run modal_app.py::benchmark --config configs/benchmark.toml
modal run modal_app.py::export --campaign <campaign-id>
```

Preflight/preparation still consume some CPU resources. The pilot is the first GPU phase. Keep those distinctions visible in CLI output.

The benchmark command prints the frozen matrix, protocol hash, remaining reservation allowance, and maximum duration before submission. It must refuse unresolved sources or a forecast outside the budget.

Use attached execution initially. A disconnected controller is recoverable through recorded call IDs and remote deadlines; do not assume that closing a laptop immediately cancels every remote task.

### 13.3 Billing and emergency stop

Current documented Modal commands:

```bash
modal billing rates --json
modal billing report --for today --resolution h --show-resources --json
modal billing summary --json
modal app stop <app-id>
```

The runner should print the relevant app/call IDs and the stop instruction at startup. After stopping, confirm active work has ended before releasing reservations or starting another stage.

### 13.4 Reproduce the published analysis

```bash
python -m pip install --require-hashes -r requirements/analysis.lock.txt
python -m inference_bench verify-artifacts --input <downloaded-bundle>
python -m inference_bench analyze --input <downloaded-bundle> --output report
```

Provide an equivalent `analyze_remote` path for a restricted machine, using a CPU-only Modal image. Local analysis of an already downloaded bundle remains the canonical no-cloud reproduction path.

## 14. Figures, article, and site handoff

### 14.1 Three main figures

1. **Output throughput vs concurrency**, with one panel per workload, consistent engine colors, and the three-run spread.
2. **TTFT vs concurrency**, showing p50/p95 clearly and distinguishing quantile lines from repetition spread.
3. **Completion-latency/throughput tradeoff**, with achieved output throughput on one axis and p95 end-to-end latency on the other; label the concurrency points.

Generate all figures from the same validated summary tables. Include units, the exact GPU/model/precision, the sample counts, and captions explaining the range shown. Do not use a misleading zero-suppressed bar chart or conceal failed points.

Keep cost in a compact table with both the price basis and the actual project cost. With one hardware allocation, unit compute cost is largely a transformation of throughput, so another large cost chart adds little.

Export self-contained SVGs with a legible opaque background and embedded/default-safe text styling. Check them on the site's light and dark themes and at phone width. Static files avoid changing the site's MDX component registry.

### 14.2 Article outline and word budget

Suggested working title:

> Gemma 4 on Modal: a small vLLM and SGLang serving comparison

Approximate allocation:

| Part | Words |
|---|---:|
| The question and measured outcome | 150–200 |
| Hardware, software, and workload setup | 250–300 |
| Throughput and first-response results | 300–400 |
| Completion-latency tradeoff and cost | 250–300 |
| Limits, interpretation, and reproduction | 250–350 |

Writing rules:

- Follow the site instructions: neutral, considered, skeptical of overclaims.
- Put observed results ahead of generic engine descriptions.
- Use the exact measured versions, hardware, precision, prompt lengths, output lengths, and concurrency.
- Explain that prefix caching and speculation were disabled.
- Explain pretokenized input, forced output length, closed-loop load, finite-batch timing, and colocated measurement.
- State that cloud allocations, CPU/kernel choices, and numerical output differences can affect results.
- Distinguish an empirical crossover from a general runtime recommendation.
- Do not claim one architecture caused a measured effect without supporting investigation.
- Do not imply production deployment experience from this experiment.
- Link the raw bundle, frozen configuration, methods, and a specific code release.
- If there is no clear winner, say so and report where the curves differ.

### 14.3 Site frontmatter and assets

Planned frontmatter:

```yaml
---
title: "Gemma 4 on Modal: a small vLLM and SGLang serving comparison"
tags: ["inference", "benchmarks", "vllm", "sglang"]
date: 2026-10-30
excerpt: <A measured description of the completed study, without an unverified winner.>
---
```

Keep the project inception date according to the site's convention; Git will supply the later update date. The example excerpt is a drafting placeholder and must be replaced before the file enters `content/posts`.

Use ordinary Markdown images, for example:

```markdown
![Output throughput at each concurrency for the three workloads.](/posts/gemma4-vllm-sglang/throughput.svg)
```

No thread metadata is needed for this single post. Add it only if a follow-up series actually exists.

### 14.4 Publication checklist tied to this project

- Build the static site.
- Verify `/posts/gemma4-vllm-sglang/` and all three asset paths in the exported output.
- Check figure/table agreement and numeric claims against `claims.json`.
- Verify actual public code/release/artifact links.
- Ensure no unfinished draft placeholders reached the content directory.
- Confirm the site diff contains only the intended content/assets.

## 15. Failure decisions and scope preservation

| Finding | Planned response |
|---|---|
| Modal client or ordinary network access fails on the controller machine | Continue on another machine using the same public source and locks. |
| Official model access fails inside Modal | Fix upstream access/Secret configuration or stop preparation; do not substitute an unverified artifact silently. |
| The pinned image lacks a required capability | Record the error; investigate a documented compatible release within the pilot allowance; freeze a new explicit candidate pair before running. |
| A notebook/example suggests installing `main` | Prefer the released image. A source build would require its own immutable commit and a reason, and may be outside the small study's budget. |
| One engine cannot load the standard checkpoint on H200 | Investigate the concrete error within the pilot budget; report the limitation if unresolved. Do not silently switch to quantization or another model. |
| Actual GPU differs from the declared H200 | Reject the allocation and record it; do not combine its timings with H200 results. |
| Memory usage or KV allocation differs | Record it as part of the serving-stack comparison; do not claim numeric memory fractions made the pools equal. |
| Streaming usage or exact input/output lengths are wrong | Fix the shared request/protocol or stop; no approximate token accounting. |
| Client timing is saturated | Fix the common client/allocation before freezing and rerun the pilot. |
| Compilation occurs inside formal measurement | Mark affected cells invalid, establish an adequate common warmup, and version the protocol if the rule changes. |
| High-concurrency OOM or instability | Preserve and display the failure; compare the common valid region and apply only the predefined rerun policy. |
| Formal matrix exceeds budget forecast | Use the symmetric compact protocol before formal runs begin. |
| An interruption leaves partial results | Export what exists, keep the partial cell invalid, resolve active calls, and resume only within the remaining reservation. |
| Results disagree with expectations | Investigate correctness/provenance, then publish the actual result; do not search settings for a preferred winner. |
| No publishable common matrix fits | Deliver the reproducible scaffold and pilot finding; any expanded experiment is a separate scope decision. |

## 16. Final engineering handoff

The engineer implementing this plan should be able to answer the following from committed files or produced artifacts:

1. What exact question is the comparison answering?
2. Which model files, tokenizer/template, corpus, prompts, images, package versions, and harness source produced a point?
3. What work did both engines receive, and what controls were deliberately left engine-specific?
4. What exactly starts/stops each timer, and what is counted as an output token?
5. What happened to errors, incomplete streams, early EOS, OOM, retries, warmup, startup, and preemption?
6. How many independently started repetitions support each figure, and what does its error band mean?
7. How can a reader regenerate the figures without running the model?
8. How much did the whole project cost, including failed work?
9. What changes in the site publish the result?

Expected implementation effort is approximately **two to four focused working days** for the scaffold, qualification, analysis, and article, plus GPU run time and any external package/access issues. This is a planning estimate, not a deadline commitment. The matrix and budget remain deliberately small even though the plan documents the details needed for a defensible public result.

Proceed in this order:

```text
thin controller
  -> CPU source/image preparation
  -> tested common client and runner
  -> bounded real-model pilot
  -> frozen protocol
  -> formal measurements
  -> reproducible analysis
  -> article
  -> site integration/public release
```

## 17. References and verification limits

Primary references checked during planning:

1. [Google Gemma overview, model sizes, and memory discussion](https://ai.google.dev/gemma/docs/core).
2. [vLLM 0.32.0 release and CUDA image variants](https://github.com/vllm-project/vllm/releases/tag/v0.32.0).
3. [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/).
4. [vLLM 0.32.0 OpenAI-compatible server documentation](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/serving/online_serving/openai_compatible_server.md), including model generation defaults.
5. [SGLang 0.5.22 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.22).
6. SGLang 0.5.22 [Gemma cookbook](https://github.com/sgl-project/sglang/blob/v0.5.22/docs/cookbook/autoregressive/Google/Gemma4.mdx) and [hardware command generator](https://github.com/sgl-project/sglang/blob/v0.5.22/docs/src/snippets/autoregressive/gemma4-deployment.jsx).
7. SGLang [installation](https://docs.sglang.io/docs/get-started/install), [OpenAI API](https://docs.sglang.io/docs/basic_usage/openai_api_completions), [sampling defaults](https://docs.sglang.io/docs/basic_usage/sampling_params), [server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments), and [benchmark guide](https://docs.sglang.io/docs/developer_guide/bench_serving).
8. Docker Hub image metadata for [vLLM 0.32.0](https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/v0.32.0) and [SGLang 0.5.22](https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/v0.5.22), used for digest resolution.
9. Modal [pricing](https://modal.com/pricing), [budget semantics](https://modal.com/docs/guide/budgets), and [billing CLI](https://modal.com/docs/cli/latest/billing).
10. Modal [GPU allocation](https://modal.com/docs/guide/gpu), [CUDA](https://modal.com/docs/guide/cuda), [resources](https://modal.com/docs/guide/resources), [regions](https://modal.com/docs/guide/region-selection), and [existing images](https://modal.com/docs/guide/existing-images).
11. Modal [function lifecycle](https://modal.com/docs/sdk/py/latest/App), [timeouts](https://modal.com/docs/guide/timeouts), [preemption](https://modal.com/docs/guide/preemption), [retries](https://modal.com/docs/guide/retries), [model storage](https://modal.com/docs/guide/model-weights), and [Volumes](https://modal.com/docs/guide/volumes).
12. [Modal package metadata](https://pypi.org/pypi/modal/json).

Proposed source identifiers to inspect during CPU preparation:

- https://huggingface.co/google/gemma-4-26B-A4B-it
- https://huggingface.co/datasets/Salesforce/wikitext

The model/dataset Hub metadata was unavailable from the original development environment, and no GPU or container execution was performed during planning. The evidence supports the proposed architecture and candidate versions; the CPU preflight and real-model pilot remain the required checks of executable compatibility and measurement behavior.
