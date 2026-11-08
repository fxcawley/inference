"""CPU image inspection adapter. GPU dispatch follows external qualification.

    modal run modal_app.py::preflight --config configs/benchmark.toml

The offline harness works without importing this module or installing Modal.
"""

from pathlib import Path

from inference_bench.config import load_config
from inference_bench.provenance import read_json

ROOT = Path(__file__).resolve().parent


def cpu_options() -> dict:
    return {
        "cpu": (2.0, 2.0), "memory": (8192, 8192), "timeout": 600,
        "startup_timeout": 180, "single_use_containers": True,
        "retries": 0, "min_containers": 0, "max_containers": 1,
    }


def planned_gpu_options() -> dict:
    """Resource contract for the later qualified GPU adapter; creates no resource."""
    return {
        "gpu": "H200", "cpu": (8.0, 8.0), "memory": (131072, 131072),
        "timeout": 7200, "startup_timeout": 180, "single_use_containers": True,
        "retries": 0, "min_containers": 0, "max_containers": 1,
        "buffer_containers": 0,
    }


try:
    import modal
except ImportError:
    app = None
else:
    app = modal.App("gemma-serving-benchmark-preflight")
    locks = read_json(ROOT / "configs/engines.lock.json")["engines"]
    images = {
        name: modal.Image.from_registry(record["image"]).entrypoint([]).add_local_python_source(
            "inference_bench",
        )
        for name, record in locks.items()
    }

    @app.function(image=images["vllm"], **cpu_options())
    def inspect_vllm(configuration: dict) -> dict:
        from inference_bench.preflight import inspect_runtime
        return inspect_runtime("vllm", configuration, locks["vllm"]["image"])

    @app.function(image=images["sglang"], **cpu_options())
    def inspect_sglang(configuration: dict) -> dict:
        from inference_bench.preflight import inspect_runtime
        return inspect_runtime("sglang", configuration, locks["sglang"]["image"])

    @app.local_entrypoint()
    def preflight(config: str = "configs/benchmark.toml", output: str = "artifacts/preflight"):
        from inference_bench.artifacts import seal, write_json
        from inference_bench.config import Config
        import uuid

        specification = load_config(config)
        # Each invocation gets its own directory; inspection never changes the locks.
        destination = Path(output) / uuid.uuid4().hex
        destination.mkdir(parents=True, exist_ok=False)
        for name, function in (("vllm", inspect_vllm), ("sglang", inspect_sglang)):
            Config(specification.data)
            result = function.remote(specification.data)
            write_json(destination / f"{name}.json", result)
        seal(destination)
        print(f"CPU inspection saved to {destination}; real-model qualification remains pending.")


if __name__ == "__main__":
    raise SystemExit(
        "Use the offline inference-bench CLI, or install the Modal extra and run the CPU preflight with Modal."
    )
