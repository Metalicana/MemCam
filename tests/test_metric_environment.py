import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from metric_environment import metric_command
sys.path.pop(0)


class MetricEnvironmentTests(unittest.TestCase):
    def test_direct_interpreter_without_conda_or_shell(self):
        with tempfile.TemporaryDirectory(prefix="metric env ") as directory:
            prefix = Path(directory)
            (prefix / "bin").mkdir()
            (prefix / "bin/python").symlink_to(sys.executable)
            with patch.dict(os.environ, {"VBENCH_ENV_PATH": str(prefix), "PATH": "/usr/bin:/bin"}):
                command = metric_command("vbench", "python", "-c",
                    "import os,json; print(json.dumps({k:os.environ.get(k) for k in "
                    "['CONDA_PREFIX','CONDA_DEFAULT_ENV','PATH','PYTHONHOME','PYTHONPATH']}))")
                self.assertNotIn("conda", command)
                with patch.dict(os.environ, {"PYTHONHOME": "/invalid", "PYTHONPATH": "/invalid"}):
                    result = subprocess.run(command, capture_output=True, text=True, check=True)
            env = json.loads(result.stdout)
            self.assertEqual(env["CONDA_PREFIX"], str(prefix))
            self.assertEqual(env["CONDA_DEFAULT_ENV"], "vbench")
            self.assertTrue(env["PATH"].startswith(str(prefix / "bin") + ":"))
            self.assertIsNone(env["PYTHONHOME"])
            self.assertIsNone(env["PYTHONPATH"])

    def test_missing_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MEMCAM_ENV_PATH": directory}):
                with self.assertRaisesRegex(FileNotFoundError, "MEMCAM_ENV_PATH"):
                    metric_command("memcam", "python", "-V")

    def test_rejects_unknown_environment_or_program(self):
        for env, program in [("base", "python"), ("memcam", "bash")]:
            with self.assertRaises(ValueError):
                metric_command(env, program)
