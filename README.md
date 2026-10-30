# Gemma 4 serving benchmark

A planned public comparison of **vLLM and SGLang serving Gemma 4 26B-A4B on Modal**, with a short write-up for [cawley.dev](https://cawley.dev).

**Status: planning and handoff documentation.** The benchmark harness has not been implemented, and no GPU measurements have been collected. The initial repository contains the plan and conventions for continuing the work on another machine.

Read the **[full megaplan](docs/megaplan.md)** for the experiment design, implementation work packages, file map, measurement definitions, budget controls, and acceptance criteria.

## Continue on another machine

```bash
git clone https://github.com/fxcawley/inference.git
cd inference
```

Only Git is needed to read the plan. The intended implementation uses Python 3.12 on Linux or WSL for a small Modal client. Model downloads, GPU dependencies, and inference run inside Modal; local GPU hardware and a local copy of the model are not required.

Start with [AGENTS.md](AGENTS.md) and [the implementation work packages](docs/megaplan.md#11-ordered-implementation-work-packages).

Suggested instruction for resuming with a coding assistant:

> Read README.md, AGENTS.md, and docs/megaplan.md. Implement WP0 through WP3: the small controller, reproducible remote image/source preparation, shared benchmark client, artifact handling, budget controls, and CPU tests. Keep the agreed experimental protocol. Report what is implemented and verified, and leave the real-model GPU pilot for WP4. Do not present mocked results as measurements.

The commands in the megaplan describe the interface to implement. They are not runnable benchmark commands yet.

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

## First implementation tasks

1. Finish WP0's Python package, small controller environment, strict configuration validation, and dry-run planning command. Git initialization and this documentation handoff are already done.
2. Build and inspect the remote images on CPUs. Resolve the model/dataset revisions and dependency locks before GPU work.
3. Implement and test common request generation, streaming measurements, artifact validation, process cleanup, and budget reservations.
4. Run the bounded real-model pilot only after those prerequisites are ready and the GPU phase is requested.

The site repository is [fxcawley/site](https://github.com/fxcawley/site). Keep the article draft in this repository until measured results are ready: the existing site content loader has no draft filter.

## Reproducibility and licensing

The planned release will include the source, frozen configuration, provenance manifests, summary tables, figures, and a checksummed raw-data bundle. GPU packages, credentials, downloaded weights, caches, and raw runs do not belong in Git.

Original project code and documentation are licensed under [MIT](LICENSE). Model weights and source datasets retain their own licenses and attribution requirements.
