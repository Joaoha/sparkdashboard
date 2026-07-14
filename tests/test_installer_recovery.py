#!/usr/bin/env python3
"""Regression tests for retry-safe Spark Dashboard installation."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER_PATH = REPO_ROOT / "scripts/install_packages.py"
BOOTSTRAP_PATH = REPO_ROOT / "bootstrap.sh"


def load_installer():
    spec = importlib.util.spec_from_file_location("install_packages_under_test", INSTALLER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InstallerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.installer = load_installer()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp = Path(self.temp_dir.name)

    def test_pixal3d_installs_python_headers_before_cloning(self):
        """A missing Python.h must fail before it can leave a cloned app root."""
        root = self.temp / "Pixal3D"
        root.mkdir()
        (root / "requirements.txt").write_text("# fixture\n")
        events: list[tuple[str, list[str]]] = []
        manifest = dict(self.installer.MANIFEST)
        manifest["pixal3d"] = {**manifest["pixal3d"], "root": str(root)}

        def record_run(cmd, **_kwargs):
            events.append(("run", list(cmd)))

        def record_clone(*_args, **_kwargs):
            events.append(("clone", []))

        with (
            patch.object(self.installer, "MANIFEST", manifest),
            patch.object(self.installer, "ensure_owned_dir"),
            patch.object(self.installer, "run", side_effect=record_run),
            patch.object(self.installer, "clone_repo", side_effect=record_clone),
            patch.object(self.installer, "venv_python", return_value=root / ".venv/bin/python"),
            patch.object(self.installer, "install_torch"),
        ):
            self.installer.install_package(
                "pixal3d",
                download_models=False,
                skip_deps=False,
                build_pixal3d_trellis_flag=False,
                dry_run=False,
            )

        apt_install = next(cmd for kind, cmd in events if kind == "run" and cmd[:2] == ["apt-get", "install"])
        self.assertIn("python3.12-dev", apt_install)
        self.assertLess(events.index(("run", apt_install)), events.index(("clone", [])))

    def test_existing_valid_clone_is_reused_without_a_network_fetch(self):
        """A retry must use a complete checkout rather than fail on a new fetch."""
        origin = self.temp / "origin"
        dest = self.temp / "dest"
        subprocess.run(["git", "init", "--initial-branch=main", str(origin)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(origin), "config", "user.name", "Test"], check=True)
        subprocess.run(["git", "-C", str(origin), "config", "user.email", "test@example.invalid"], check=True)
        (origin / "README").write_text("fixture\n")
        subprocess.run(["git", "-C", str(origin), "add", "README"], check=True)
        subprocess.run(["git", "-C", str(origin), "commit", "-m", "fixture"], check=True, capture_output=True)
        subprocess.run(["git", "clone", str(origin), str(dest)], check=True, capture_output=True)
        calls: list[list[str]] = []

        with patch.object(self.installer, "run", side_effect=lambda cmd, **_kwargs: calls.append(list(cmd))):
            self.installer.clone_repo(origin.as_uri(), dest, "main", dry_run=False)

        self.assertEqual(calls, [])

    def test_incomplete_clone_is_quarantined_then_retried(self):
        """A corrupt .git directory from an interrupted clone cannot block retry."""
        dest = self.temp / "Pixal3D"
        (dest / ".git").mkdir(parents=True)
        (dest / "partial-file").write_text("partial\n")
        calls: list[tuple[list[str], dict]] = []

        def record_run(cmd, **kwargs):
            calls.append((list(cmd), kwargs))

        with patch.object(self.installer, "run", side_effect=record_run):
            self.installer.clone_repo("https://example.invalid/Pixal3D.git", dest, "main", dry_run=False)

        move = next((cmd, kwargs) for cmd, kwargs in calls if cmd[0] == "mv")
        self.assertFalse(move[1]["sudo"], "a user-writable test/install parent must not require sudo")
        self.assertTrue(any(cmd[:2] == ["git", "clone"] for cmd, _kwargs in calls), calls)

    def test_bootstrap_can_be_rerun_without_reusing_a_previous_checkout(self):
        """The published one-command entry point uses a new temporary checkout each time."""
        origin = self.temp / "origin"
        subprocess.run(["git", "init", "--initial-branch=main", str(origin)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(origin), "config", "user.name", "Test"], check=True)
        subprocess.run(["git", "-C", str(origin), "config", "user.email", "test@example.invalid"], check=True)
        (origin / "install.sh").write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$BOOTSTRAP_TEST_LOG\"\n")
        (origin / "install.sh").chmod(0o755)
        subprocess.run(["git", "-C", str(origin), "add", "install.sh"], check=True)
        subprocess.run(["git", "-C", str(origin), "commit", "-m", "fixture"], check=True, capture_output=True)
        log = self.temp / "runs.log"
        env = {
            **os.environ,
            "SPARKDASHBOARD_REPO_URL": origin.as_uri(),
            "SPARKDASHBOARD_REF": "main",
            "BOOTSTRAP_TEST_LOG": str(log),
            "TMPDIR": str(self.temp),
        }

        for _ in range(2):
            completed = subprocess.run(
                ["bash", str(BOOTSTRAP_PATH), "--models", "none", "--start", "none"],
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

        self.assertEqual(log.read_text().splitlines(), ["--models none --start none", "--models none --start none"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
