# Project instructions

## Starting point

- Read `README.md` and `docs/megaplan.md` before implementation.
- The local harness and offline analysis are implemented. Read `docs/runbook.md` for verified capabilities and remaining integration work; the complete megaplan still includes future steps.
- Follow the user's requested work package. Publishing this plan does not by itself request running the paid GPU campaign.
- Update the README's implementation status as work becomes complete. Distinguish source inspection, CPU tests, model qualification, and actual measurements.

## Experimental contract

- Compare the same official Gemma checkpoint revision, prepared prompt IDs, output-token targets, precision, GPU SKU, and resource allocation.
- Use one shared measurement client and metric implementation for both engines.
- The baseline disables prefix reuse and speculative decoding. Engine-specific optimization studies are later work.
- Freeze the protocol before formal measurements; retain the protocol hash and immutable source/image identities.
- Use authoritative server token usage. Never substitute chunk count, character count, requested length, or re-tokenized output for missing token accounting.
- Streaming events are not necessarily individual tokens. Do not label chunk intervals as token-level ITL.
- Keep warmup, pilot, measured, failed, and incomplete work distinguishable.
- Preserve failures and replacement-attempt history. Select results using the plan's validity rules, not the fastest run.
- Mock fixtures are for tests and must never appear as measured public results.

## Environment and cost

- Prefer Python 3.12 on Linux/WSL for the small local controller.
- Keep the engine stacks in separate pinned remote images. Do not make CUDA, PyTorch, vLLM, SGLang, or local model weights controller requirements.
- Download official weights and public data inside Modal, with immutable revisions and checksums.
- Use the $75 gross project budget and the plan's stage allocations, deadlines, serial execution, reservations, and billing reconciliation.
- Preserve budget state when resuming or moving machines. Do not reset the campaign allowance on retries or a new billing month.
- Use one project ledger, identified by `budget.ledger_id`, for pilot and formal stages. Record cumulative billing with `record-billing`; never edit the ledger to release allowance.
- Keep credentials in the user's credential store or Modal Secrets. Upload only intended project source files.
- Store raw artifacts and weights outside Git; publish the eventual raw-data bundle with checksums.

## Implementation and verification

- Use the planned module boundaries without building a general plugin framework.
- Prefer standard-library tools where they suffice.
- Write meaningful CPU tests for streaming/timing, token accounting, failure handling, process cleanup, and budget controls.
- CI must remain CPU-only and must not launch paid jobs or require credentials.
- Record actual validation outcomes. Do not describe documented support as a successful runtime qualification.
- Mark an engine qualified only with a passed `qualify` report from real pilot evidence, recorded by path and SHA-256 in `configs/engines.lock.json`. Never accept a failed check or hand-edit a report.
- Without network access, run `PYTHONPATH=src:tests python -m unittest test_protocol test_metrics test_runner test_client.DecoderTests test_client.RequestContractTests`; the remaining HTTP tests open loopback sockets.

## Article and site

- Keep unfinished prose in `report/article.md`; the site currently publishes every discovered post under `content/posts`.
- Write in a neutral, measured voice. State limitations and uncertainty; avoid hype, punchy one-liners, and unsupported causal explanations.
- The short article should link to detailed methods, frozen configuration, code, and raw data.
- Follow the site repository's own instructions when integrating the final article.

## Git

- Use plain, concise commit messages.
- Do not add `Generated with` or `Co-Authored-By` trailers.
- Do not commit local environments, secrets, model weights, caches, or raw experiment directories.
