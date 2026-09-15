"""Run metric Python environments directly, without Conda shell startup."""

import os
from pathlib import Path


def metric_command(name, program, *args):
    if name not in ("memcam", "vbench") or program != "python":
        raise ValueError("Expected a Python command in memcam or vbench")
    prefix = Path(os.environ.get(f"{name.upper()}_ENV_PATH", Path.home() / ".conda/envs" / name))
    python = prefix / "bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"No executable {python}; set {name.upper()}_ENV_PATH")
    # Set subprocess-local paths without sourcing activation hooks or shell rc files.
    return [
        "/usr/bin/env", "-u", "PYTHONHOME", "-u", "PYTHONPATH",
        f"PATH={prefix / 'bin'}:{os.environ.get('PATH', '')}",
        f"CONDA_PREFIX={prefix}", f"CONDA_DEFAULT_ENV={name}",
        str(python), "-u", *map(str, args),
    ]
