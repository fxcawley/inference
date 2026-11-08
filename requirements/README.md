# Dependency boundaries

The core CLI, metric calculations, artifact validation, SVG analysis, and synthetic fixture generator use Python's standard library. Tests use `unittest`.

The only dependency for streaming HTTP and its loopback tests is HTTPX 0.28.1. The `client` extra installs it. The `modal` and `prepare` extras are separate, optional integration dependencies.

These `.in` files declare direct candidates, **not transitive hash locks**. Resolve and inspect the full dependency sets in the actual remote environments before the pilot. Then commit the planned `*.lock.txt` files with hashes, record the shared Python 3.12 patch version, and update `configs/engines.lock.json` with the client lock's SHA-256.

Keep the common client in a separate remote virtual environment. The engine interpreters retain their own official image dependencies; do not upgrade one engine's Torch or Transformers to match the other.

The model tokenizer is loaded from an already downloaded snapshot using the qualified preparation image's Transformers installation. A corpus adapter must export the pinned WikiText data to JSONL objects with a `text` field; its dataset revision, normalization, license, and file hashes must be recorded.

No dependency lock or engine qualification is inferred from the development machine's installed packages.
