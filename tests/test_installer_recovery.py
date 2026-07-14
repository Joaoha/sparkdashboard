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

    def test_head_bearing_checkout_missing_a_tracked_file_is_repaired(self):
        """A resolved HEAD alone does not prove an interrupted checkout is usable."""
        origin = self.temp / "origin"
        dest = self.temp / "dest"
        subprocess.run(["git", "init", "--initial-branch=main", str(origin)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(origin), "config", "user.name", "Test"], check=True)
        subprocess.run(["git", "-C", str(origin), "config", "user.email", "test@example.invalid"], check=True)
        (origin / "requirements.txt").write_text("fixture-dependency\n")
        subprocess.run(["git", "-C", str(origin), "add", "requirements.txt"], check=True)
        subprocess.run(["git", "-C", str(origin), "commit", "-m", "fixture"], check=True, capture_output=True)
        subprocess.run(["git", "clone", str(origin), str(dest)], check=True, capture_output=True)
        (dest / "requirements.txt").unlink()

        self.assertFalse(self.installer.is_complete_git_checkout(dest))
        self.installer.clone_repo(origin.as_uri(), dest, "main", dry_run=False)

        self.assertTrue(self.installer.is_complete_git_checkout(dest))
        self.assertEqual((dest / "requirements.txt").read_text(), "fixture-dependency\n")
        self.assertTrue(list(self.temp.glob(".dest.interrupted-clone-*")))

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

    def test_quarantine_recreates_a_root_clone_destination_under_opt(self):
        """After sudo moves /opt/<app>, the installer must recreate it for Git."""
        dest = self.temp / "personaplex-bnb4"
        (dest / ".git").mkdir(parents=True)
        (dest / "partial-file").write_text("partial\n")
        calls: list[tuple[list[str], dict]] = []

        def record_run(cmd, **kwargs):
            calls.append((list(cmd), kwargs))

        with (
            patch.object(self.installer, "run", side_effect=record_run),
            patch.object(self.installer.os, "access", return_value=False),
        ):
            self.installer.clone_repo("https://example.invalid/personaplex.git", dest, None, dry_run=False)

        move_index = next(i for i, (cmd, _kwargs) in enumerate(calls) if cmd[0] == "mv")
        install_index = next(i for i, (cmd, _kwargs) in enumerate(calls) if cmd[:3] == ["install", "-d", "-m"] and cmd[-1] == str(dest))
        clone_index = next(i for i, (cmd, _kwargs) in enumerate(calls) if cmd[:2] == ["git", "clone"])
        self.assertTrue(calls[move_index][1]["sudo"])
        self.assertTrue(calls[install_index][1]["sudo"])
        self.assertLess(move_index, install_index)
        self.assertLess(install_index, clone_index)

    def test_domainshuttle_builds_decord_from_official_source(self):
        """ARM64/Python 3.12 must not ask PyPI for a nonexistent decord wheel."""
        root = self.temp / "domainshuttle"
        repo = root / "repo"
        repo.mkdir(parents=True)
        (repo / "requirements.txt").write_text(
            "numpy\nDecord >= 0.6\ndecord ; python_version >= '3.12'\n"
            "decord[torch]\ndecord @ git+https://example.invalid/decord.git\n"
            "-r shared-requirements.txt\ntimm\n"
        )
        decord_header = root / "decord/src/video/ffmpeg/ffmpeg_common.h"
        decord_header.parent.mkdir(parents=True)
        decord_header.write_text("#include <libavcodec/avcodec.h>\n")
        decord_reader = root / "decord/src/video/video_reader.cc"
        decord_reader.parent.mkdir(parents=True, exist_ok=True)
        decord_reader.write_text("AVCodec *dec;\n")
        events: list[tuple[str, list[str]]] = []
        clone_calls: list[tuple[str, Path, bool]] = []
        manifest = dict(self.installer.MANIFEST)
        manifest["domainshuttle"] = {**manifest["domainshuttle"], "root": str(root)}

        def record_run(cmd, **_kwargs):
            events.append(("run", list(cmd)))

        def record_clone(url, dest, _branch, *, recursive=False, **_kwargs):
            clone_calls.append((url, dest, recursive))

        with (
            patch.object(self.installer, "MANIFEST", manifest),
            patch.object(self.installer, "ensure_owned_dir"),
            patch.object(self.installer, "clone_repo", side_effect=record_clone),
            patch.object(self.installer, "copy_tree"),
            patch.object(self.installer, "venv_python", return_value=root / ".venv/bin/python"),
            patch.object(self.installer, "install_torch"),
            patch.object(self.installer, "run", side_effect=record_run),
            patch.object(self.installer, "pip_install"),
        ):
            self.installer.install_package(
                "domainshuttle",
                download_models=False,
                skip_deps=False,
                build_pixal3d_trellis_flag=False,
                dry_run=False,
            )

        filtered = root / ".spark-domainshuttle-requirements.txt"
        self.assertEqual(filtered.read_text().splitlines(), ["numpy", "-r shared-requirements.txt", "timm"])
        self.assertIn("#include <libavcodec/bsf.h>", decord_header.read_text())
        self.assertIn("const AVCodec *dec;", decord_reader.read_text())
        self.assertIn(
            ("https://github.com/dmlc/decord.git", root / "decord", True),
            clone_calls,
        )
        apt_install = next(cmd for kind, cmd in events if kind == "run" and cmd[:2] == ["apt-get", "install"])
        self.assertIn("libavcodec-dev", apt_install)
        fetch_index = next(i for i, (_kind, cmd) in enumerate(events) if cmd[:3] == ["git", "fetch", "--depth"] and self.installer.DECORD_COMMIT in cmd)
        checkout_index = next(i for i, (_kind, cmd) in enumerate(events) if cmd[:2] == ["git", "checkout"] and self.installer.DECORD_COMMIT in cmd)
        cmake_indices = [i for i, (_kind, cmd) in enumerate(events) if cmd[0] == "cmake"]
        binding_index = next(i for i, (_kind, cmd) in enumerate(events) if cmd[:4] == [str(root / ".venv/bin/python"), "-m", "pip", "install"] and str(root / "decord/python") in cmd)
        import_index = next(i for i, (_kind, cmd) in enumerate(events) if cmd[:2] == [str(root / ".venv/bin/python"), "-c"] and "import decord" in cmd[2])
        self.assertLess(fetch_index, checkout_index)
        self.assertLess(checkout_index, cmake_indices[0])
        self.assertLess(cmake_indices[0], cmake_indices[-1])
        self.assertLess(cmake_indices[-1], binding_index)
        self.assertLess(binding_index, import_index)

    def test_decord_ffmpeg_compatibility_patch_includes_bsf_header(self):
        """Decord omits bsf.h although it uses AVBSFContext on modern FFmpeg."""
        decord_root = self.temp / "decord"
        header = decord_root / "src/video/ffmpeg/ffmpeg_common.h"
        header.parent.mkdir(parents=True)
        header.write_text("#include <libavcodec/avcodec.h>\n#include <libavformat/avformat.h>\n")

        self.installer.patch_decord_ffmpeg_bsf_header(decord_root)
        self.installer.patch_decord_ffmpeg_bsf_header(decord_root)

        lines = header.read_text().splitlines()
        self.assertIn("#include <libavcodec/bsf.h>", lines)
        self.assertEqual(lines.count("#include <libavcodec/bsf.h>"), 1)
        self.assertEqual(lines.index("#include <libavcodec/bsf.h>"), lines.index("#include <libavcodec/avcodec.h>") + 1)

    def test_decord_ffmpeg_compatibility_patch_uses_const_codec(self):
        """FFmpeg 5+ requires av_find_best_stream's codec pointer to be const."""
        decord_root = self.temp / "decord"
        source = decord_root / "src/video/video_reader.cc"
        source.parent.mkdir(parents=True)
        source.write_text("void f() {\n    AVCodec *dec;\n}\n")

        self.installer.patch_decord_ffmpeg_codec_const(decord_root)
        self.installer.patch_decord_ffmpeg_codec_const(decord_root)

        self.assertIn("const AVCodec *dec;", source.read_text())
        self.assertNotIn("\n    AVCodec *dec;", source.read_text())

    def test_domainshuttle_dry_run_shows_decord_build_steps(self):
        """A clean-host dry-run must expose the Decord source-build work."""
        root = self.temp / "domainshuttle"
        events: list[list[str]] = []
        manifest = dict(self.installer.MANIFEST)
        manifest["domainshuttle"] = {**manifest["domainshuttle"], "root": str(root)}

        with (
            patch.object(self.installer, "MANIFEST", manifest),
            patch.object(self.installer, "ensure_owned_dir"),
            patch.object(self.installer, "clone_repo"),
            patch.object(self.installer, "copy_tree"),
            patch.object(self.installer, "venv_python", return_value=root / ".venv/bin/python"),
            patch.object(self.installer, "install_torch"),
            patch.object(self.installer, "pip_install"),
            patch.object(self.installer, "run", side_effect=lambda cmd, **_kwargs: events.append(list(cmd))),
        ):
            self.installer.install_package(
                "domainshuttle",
                download_models=False,
                skip_deps=False,
                build_pixal3d_trellis_flag=False,
                dry_run=True,
            )

        self.assertTrue(any(cmd[0] == "cmake" for cmd in events))
        self.assertTrue(any("decord/python" in " ".join(cmd) for cmd in events))

    def test_install_dry_run_includes_domainshuttle_dependency_steps(self):
        """Top-level --dry-run must not hide package dependency work."""
        completed = subprocess.run(
            [
                "bash", str(REPO_ROOT / "install.sh"), "--dry-run",
                "--models", "none", "--packages", "domainshuttle", "--start", "none",
                "--install-root", str(self.temp / "install-root"),
                "--model-dir", str(self.temp / "models"),
                "--public-host", "spark-test.local", "--dashboard-port", "17862",
            ],
            env={**os.environ, "TMPDIR": str(self.temp)},
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("git fetch --depth 1 origin", completed.stdout)
        self.assertIn("decord/python", completed.stdout)

    def test_install_dry_run_defaults_to_no_model_download(self):
        """Dry-runs must not imply a surprise all-model download."""
        env = os.environ.copy()
        env.pop("SPARK_MODELS", None)
        completed = subprocess.run(
            [
                "bash", str(REPO_ROOT / "install.sh"), "--dry-run",
                "--packages", "none", "--start", "none",
                "--install-root", str(self.temp / "install-root"),
                "--model-dir", str(self.temp / "models"),
                "--public-host", "spark-test.local", "--dashboard-port", "17862",
            ],
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("models:         none", completed.stdout)
        self.assertIn("dry-run default", completed.stdout)

    def test_install_dry_run_accepts_an_explicit_model_subset(self):
        completed = subprocess.run(
            [
                "bash", str(REPO_ROOT / "install.sh"), "--dry-run",
                "--models", "qwen,mistral", "--packages", "none", "--start", "none",
                "--install-root", str(self.temp / "install-root"),
                "--model-dir", str(self.temp / "models"),
                "--public-host", "spark-test.local", "--dashboard-port", "17862",
            ],
            env={**os.environ, "SPARK_MODELS": "all"},
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("models:         qwen,mistral", completed.stdout)

    def test_install_rejects_unknown_model_before_it_can_download(self):
        completed = subprocess.run(
            [
                "bash", str(REPO_ROOT / "install.sh"), "--dry-run",
                "--models", "qwen,not-a-model", "--packages", "none", "--start", "none",
            ],
            env=os.environ.copy(),
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("Unknown model(s): not-a-model", completed.stderr)

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
        commit = subprocess.check_output(["git", "-C", str(origin), "rev-parse", "HEAD"], text=True).strip()
        log = self.temp / "runs.log"
        env = {
            **os.environ,
            "SPARKDASHBOARD_REPO_URL": origin.as_uri(),
            "SPARKDASHBOARD_REF": commit,
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

    def test_bootstrap_rejects_a_mutable_branch_ref(self):
        """The public entry point must not execute an arbitrary moving branch."""
        completed = subprocess.run(
            ["bash", str(BOOTSTRAP_PATH), "--help"],
            env={
                **os.environ,
                "SPARKDASHBOARD_REPO_URL": "file:///does-not-matter",
                "SPARKDASHBOARD_REF": "main",
            },
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("40-character commit SHA", completed.stderr)

    def test_agent3dify_is_skipped_by_all_on_linux_arm64(self):
        """A package lacking ARM64 artifacts cannot abort every optional package."""
        installed: list[str] = []
        with (
            patch.object(self.installer.sys, "argv", ["install_packages.py", "all"]),
            patch.object(self.installer.sys, "platform", "linux"),
            patch.object(self.installer.platform, "machine", return_value="aarch64"),
            patch.object(self.installer, "install_package", side_effect=lambda key, **_kwargs: installed.append(key)),
        ):
            self.assertEqual(self.installer.main(), 0)

        self.assertNotIn("agent3dify", installed)
        self.assertIn("triposplat", installed)

    def test_agent3dify_explicit_selection_preflights_on_linux_arm64(self):
        """A mixed explicit selection must fail before installing any package."""
        installed: list[str] = []
        with (
            patch.object(self.installer.sys, "argv", ["install_packages.py", "z-image,agent3dify"]),
            patch.object(self.installer.sys, "platform", "linux"),
            patch.object(self.installer.platform, "machine", return_value="aarch64"),
            patch.object(self.installer, "install_package", side_effect=lambda key, **_kwargs: installed.append(key)),
        ):
            with self.assertRaisesRegex(SystemExit, "cadquery-ocp"):
                self.installer.main()

        self.assertEqual(installed, [])

    def test_agent3dify_remains_selectable_on_linux_x86_64(self):
        """The ARM64 guard must not block Agent3Dify on supported Linux x86_64."""
        installed: list[str] = []
        with (
            patch.object(self.installer.sys, "argv", ["install_packages.py", "agent3dify"]),
            patch.object(self.installer.sys, "platform", "linux"),
            patch.object(self.installer.platform, "machine", return_value="x86_64"),
            patch.object(self.installer, "install_package", side_effect=lambda key, **_kwargs: installed.append(key)),
        ):
            self.assertEqual(self.installer.main(), 0)

        self.assertEqual(installed, ["agent3dify"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
