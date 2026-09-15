from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "slurm/newton_budget_metric_grid.sbatch"


class BudgetLauncherTests(unittest.TestCase):
    def run_launcher(self, root, extra=None, conda_available=True):
        bin_dir = root / "bin"
        bin_dir.mkdir()
        commands = {"module": "exit 0", "python": 'printf "%s\\n" "$@" > "$HOME/calls"'}
        if conda_available:
            commands["conda"] = "exit 0"
        for name, body in commands.items():
            path = bin_dir / name
            path.write_text("#!/bin/bash\n" + body + "\n")
            path.chmod(0o755)
        env_dir = root / ".conda/envs/memcam/bin"
        env_dir.mkdir(parents=True)
        (env_dir / "python").symlink_to(bin_dir / "python")
        env = {
            "HOME": str(root), "PATH": f"{bin_dir}:/usr/bin:/bin",
            "MEMCAM_ROOT": str(root), "TMPDIR": str(root),
            "SLURM_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "4",
        }
        env.update(extra or {})
        return subprocess.run(
            ["/bin/bash", str(SCRIPT), str(root / "plan.json")],
            env=env, capture_output=True, text=True,
        )

    def test_no_inherited_conda_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_launcher(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "calls").read_text().splitlines(), [
                "utils/run_budget_metric_grid.py", "run", "--plan",
                str(root / "plan.json"), "--task", "4",
            ])

    def test_unrelated_active_environment_is_not_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_launcher(root, {"CONDA_PREFIX": "/missing/base", "CONDA_DEFAULT_ENV": "base"})
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_explicit_environment_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_launcher(root, {"MEMCAM_ENV_PATH": str(root / "missing")})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("set MEMCAM_ENV_PATH", result.stderr)
            self.assertFalse((root / "calls").exists())

    def test_no_conda_executable_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_launcher(root, conda_available=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "calls").exists())


if __name__ == "__main__":
    unittest.main()
