"""Read-only inspection inside a candidate engine image, without requesting a GPU."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

from .config import Config
from .engines import command
from .provenance import environment


def inspect_runtime(engine: str, configuration: dict, image: str) -> dict:
    config = Config(configuration)
    argv = command(engine, "/models/qualification-pending", config)
    executable = shutil.which(argv[0])
    result = {
        "engine": engine, "upstream_image": image, "environment": environment(),
        "engine_executable": executable, "worker_interpreter": sys.executable,
        "candidate_argv": argv, "qualification": "pending_real_model",
    }
    if executable is None:
        result["cli_error"] = "Engine executable was not found in the pinned image"
        return result
    help_argv = [executable, "serve", "--help=all" if engine == "vllm" else "--help"]
    try:
        help_result = subprocess.run(
            help_argv, capture_output=True, text=True, timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result["cli_error"] = f"{type(exc).__name__}: {exc}"
        return result
    text = help_result.stdout + "\n" + help_result.stderr
    result["cli_help"] = {
        "returncode": help_result.returncode, "text": text[:262144],
        "truncated": len(text) > 262144,
    }
    flags = [value for value in argv if value.startswith("--")]
    result["flags_not_listed_in_help"] = [
        flag for flag in flags if not re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text)
    ]
    # Help output cannot establish token accounting, image driver compatibility,
    # exact client dependencies, or real-model behavior.
    return result
