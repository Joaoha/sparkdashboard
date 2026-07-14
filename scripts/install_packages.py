#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((REPO_ROOT / "config/packages.json").read_text())["packages"]

TEXT_MODEL_UNITS = {
    "qwen-nvfp4-vllm.service",
    "ornith-vllm.service",
    "mistral-medium-vllm.service",
}

COMMON_WEB_DEPS = [
    "fastapi",
    "uvicorn[standard]",
    "pydantic",
    "pillow",
    "requests",
    "python-multipart",
    "huggingface_hub",
    "hf_transfer",
    "accelerate",
    "transformers",
    "safetensors",
    "sentencepiece",
    "psutil",
]
DIFFUSERS_DEPS = COMMON_WEB_DEPS + ["diffusers", "bitsandbytes", "protobuf"]
DIFFUSERS_GIT_DEPS = COMMON_WEB_DEPS + [
    "git+https://github.com/huggingface/diffusers.git",
    "bitsandbytes",
    "protobuf",
    "einops",
]


def parse_selection(value: str) -> list[str]:
    keys = list(MANIFEST.keys())
    if value in ("", "none"):
        return []
    if value == "all":
        return keys
    chosen = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in chosen if x not in MANIFEST]
    if bad:
        raise SystemExit(f"Unknown package(s): {', '.join(bad)}. Valid: {', '.join(keys)}")
    return chosen


def run(cmd: list[str], *, sudo: bool = False, cwd: Path | None = None, dry_run: bool = False, env: dict[str, str] | None = None) -> None:
    full = (["sudo"] if sudo and os.geteuid() != 0 else []) + cmd
    print("+", " ".join(str(x) for x in full), flush=True)
    if dry_run:
        return
    subprocess.run(full, cwd=str(cwd) if cwd else None, check=True, env=env)


def ensure_owned_dir(path: Path, *, dry_run: bool) -> None:
    run(["install", "-d", "-m", "0755", str(path)], sudo=True, dry_run=dry_run)
    user = os.environ.get("SUDO_USER") or os.environ.get("USER") or "joao"
    run(["chown", "-R", f"{user}:{user}", str(path)], sudo=True, dry_run=dry_run)


def venv_python(root: Path, *, dry_run: bool) -> Path:
    py = root / ".venv/bin/python"
    if not py.exists():
        run(["python3", "-m", "venv", str(root / ".venv")], dry_run=dry_run)
    return py


def pip_install(py: Path, args: list[str], *, dry_run: bool) -> None:
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], dry_run=dry_run)
    run([str(py), "-m", "pip", "install", *args], dry_run=dry_run)


def install_torch(py: Path, *, dry_run: bool) -> None:
    # Spark/GB10 known-good path: CUDA 13 PyTorch wheels. If this fails on a
    # future distro, install the target platform's NVIDIA/PyTorch stack first
    # and rerun with --skip-deps.
    run([str(py), "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cu130", "torch", "torchvision", "torchaudio"], dry_run=dry_run)


def clone_repo(url: str, dest: Path, branch: str | None, *, dry_run: bool, recursive: bool = False) -> None:
    if dest.exists() and (dest / ".git").exists():
        run(["git", "fetch", "--depth", "1", "origin"], cwd=dest, dry_run=dry_run)
        if branch:
            run(["git", "checkout", branch], cwd=dest, dry_run=dry_run)
            run(["git", "pull", "--ff-only", "origin", branch], cwd=dest, dry_run=dry_run)
        if recursive:
            run(["git", "submodule", "update", "--init", "--recursive"], cwd=dest, dry_run=dry_run)
        return
    if dest.exists():
        entries = list(dest.iterdir())
        if len(entries) == 1 and entries[0].name == ".venv" and (entries[0] / "pyvenv.cfg").exists():
            # Older installer builds created the package venv before cloning
            # repos directly into the install root, leaving root/.venv as the
            # only entry and making git clone fail with "not an empty directory".
            # That venv is disposable bootstrap state; remove it so reruns of
            # personaplex/pixal3d can recover without manual cleanup.
            print(f"remove stale pre-clone venv {entries[0]}")
            if not dry_run:
                shutil.rmtree(entries[0])
    cmd = ["git", "clone", "--depth", "1"]
    if recursive:
        cmd.append("--recursive")
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(dest)]
    run(cmd, dry_run=dry_run)


def copy_tree(src: Path, dst: Path, *, dry_run: bool) -> None:
    print(f"copy {src} -> {dst}")
    if dry_run:
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def write_hidream_helpers(root: Path, *, dry_run: bool) -> None:
    content = f"""#!/usr/bin/env bash
set -euo pipefail
BASE={root}
PORT=${{PORT:-7861}}
HOST=${{HOST:-0.0.0.0}}
sudo systemctl stop ollama.service 2>/dev/null || true
if [ "${{HIDREAM_STOP_QWEN:-0}}" = "1" ]; then
  systemctl --user stop qwen-nvfp4-vllm.service 2>/dev/null || true
  docker rm -f qwen-nvfp4-vllm >/dev/null 2>&1 || true
fi
cd "$BASE/repo"
export FA_VERSION=0
export HF_HOME="$BASE/hf-cache"
export PYTORCH_CUDA_ALLOC_CONF=${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}
exec "$BASE/.venv/bin/python" app.py --model_path "$BASE/model" --model_type dev --host "$HOST" --port "$PORT"
"""
    path = root / "bin/start-hidream-o1-web.sh"
    print(f"write {path}")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o755)


def download_snapshot(py: Path, repo_id: str, local_dir: Path | None = None, cache_dir: Path | None = None, *, dry_run: bool) -> None:
    kwargs = {"repo_id": repo_id, "max_workers": 8}
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    code = (
        "from huggingface_hub import snapshot_download\n"
        f"print(snapshot_download(**{kwargs!r}))\n"
    )
    run([str(py), "-c", code], dry_run=dry_run, env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"})



def build_pixal3d_trellis(pixal_root: Path, py: Path, *, dry_run: bool) -> None:
    """Build TRELLIS.2 native CUDA extensions for Pixal3D on Spark/GB10.

    This is intentionally opt-in because it is a long source build and depends
    on the target's CUDA/toolchain state. It follows the known-good Spark ARM64
    runbook: CUDA 13, GB10 sm_121, SDPA instead of flash-attn.
    """
    trellis_root = Path(os.environ.get("PIXAL3D_TRELLIS_ROOT", "/opt/TRELLIS.2"))
    ensure_owned_dir(trellis_root, dry_run=dry_run)
    clone_repo("https://github.com/microsoft/TRELLIS.2.git", trellis_root, None, dry_run=dry_run, recursive=True)

    build_env = {
        **os.environ,
        "PATH": f"/usr/local/cuda-13.0/bin:{os.environ.get('PATH', '')}",
        "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0"),
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "12.1"),
        "NATTEN_CUDA_ARCH": os.environ.get("NATTEN_CUDA_ARCH", "12.1"),
        "NATTEN_N_WORKERS": os.environ.get("NATTEN_N_WORKERS", "4"),
        "MAX_JOBS": os.environ.get("MAX_JOBS", "4"),
        "CMAKE_BUILD_PARALLEL_LEVEL": os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4"),
        "ATTN_BACKEND": os.environ.get("ATTN_BACKEND", "sdpa"),
    }

    # Pixal3D upstream expects this utility wheel; install it before the native
    # extension build so app startup does not force-reinstall it every time.
    run([
        str(py), "-m", "pip", "install",
        "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl",
    ], dry_run=dry_run, env=build_env)

    run([str(py), "-m", "pip", "install", "natten==0.21.0", "--no-build-isolation"], dry_run=dry_run, env=build_env)
    run(["bash", "-lc", f"source {pixal_root}/.venv/bin/activate && ./setup.sh --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm"], cwd=trellis_root, dry_run=dry_run, env=build_env)

    verify = """
import importlib.util, sys, torch
mods = ["natten", "nvdiffrast", "nvdiffrec_render", "cumesh", "flex_gemm", "o_voxel", "spaces", "gradio"]
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
missing = [m for m in mods if importlib.util.find_spec(m) is None]
for m in mods:
    print(m, "ok" if importlib.util.find_spec(m) else "missing")
if missing:
    raise SystemExit("Missing Pixal3D/TRELLIS modules: " + ", ".join(missing))
"""
    run([str(py), "-c", verify], dry_run=dry_run, env=build_env)

def install_package(key: str, *, download_models: bool, skip_deps: bool, build_pixal3d_trellis_flag: bool, dry_run: bool) -> None:
    meta = MANIFEST[key]
    root = Path(meta["root"])
    print(f"\n== Installing optional package: {key} ({meta['name']}) ==")
    ensure_owned_dir(root, dry_run=dry_run)

    kind = meta["kind"]

    if kind == "bundled_diffusers_app":
        src_dir = REPO_ROOT / "packages" / key
        copy_tree(src_dir / "app.py", root / "app.py", dry_run=dry_run)
        if (src_dir / "scripts").exists():
            copy_tree(src_dir / "scripts", root / "scripts", dry_run=dry_run)
        py = venv_python(root, dry_run=dry_run)
        if not skip_deps:
            install_torch(py, dry_run=dry_run)
            pip_install(py, DIFFUSERS_GIT_DEPS if key in {"flux2", "krea2"} else DIFFUSERS_DEPS, dry_run=dry_run)
        if download_models and key == "krea2":
            run([str(py), str(root / "scripts/download.py")], dry_run=dry_run)
        elif download_models and meta.get("model_repo"):
            download_snapshot(py, meta["model_repo"], cache_dir=root / "hf-cache", dry_run=dry_run)

    elif key == "hidream":
        clone_repo(meta["repo"], root / "repo", meta.get("branch"), dry_run=dry_run)
        write_hidream_helpers(root, dry_run=dry_run)
        py = venv_python(root, dry_run=dry_run)
        if not skip_deps:
            install_torch(py, dry_run=dry_run)
            req = root / "repo/requirements.txt"
            if req.exists() or dry_run:
                run([str(py), "-m", "pip", "install", "-r", str(req), "huggingface_hub", "hf_transfer", "python-dotenv"], dry_run=dry_run)
        if download_models:
            download_snapshot(py, meta["model_repo"], local_dir=root / "model", dry_run=dry_run)
            # Disable flash-attn path on Spark if the file exists.
            pipeline = root / "repo/models/pipeline.py"
            if pipeline.exists() and not dry_run:
                text = pipeline.read_text().replace('"use_flash_attn": True', '"use_flash_attn": False')
                pipeline.write_text(text)

    elif key == "personaplex":
        # The live BNB4 install is a Hugging Face git repo with the quantized
        # checkpoint. This may require git-lfs/HF auth.
        if not (root / ".git").exists():
            clone_repo(meta["repo"], root, meta.get("branch"), dry_run=dry_run)
        py = venv_python(root, dry_run=dry_run)
        if not skip_deps:
            run(["apt-get", "update", "-qq"], sudo=True, dry_run=dry_run)
            run(["apt-get", "install", "-y", "pkg-config", "libopus-dev"], sudo=True, dry_run=dry_run)
            install_torch(py, dry_run=dry_run)
            req = root / "moshi/requirements.txt"
            if req.exists() or dry_run:
                # moshi pins torch<2.5, which would downgrade Spark/GB10's
                # CUDA-13 PyTorch wheel. Keep the known-good torch installed by
                # install_torch() and install the rest of the PersonaPlex deps.
                filtered_req = root / ".spark-personaplex-requirements.txt"
                print(f"write {filtered_req} without upstream torch pin")
                if not dry_run:
                    filtered_lines = []
                    for line in req.read_text().splitlines():
                        stripped = line.strip().lower()
                        if stripped == "torch" or stripped.startswith(("torch<", "torch=", "torch>", "torch~", "torch!")):
                            continue
                        filtered_lines.append(line)
                    filtered_req.write_text("\n".join(filtered_lines) + "\n")
                run([str(py), "-m", "pip", "install", "-r", str(filtered_req), "accelerate", "hf_transfer", "bitsandbytes"], dry_run=dry_run)
            moshi = root / "moshi"
            if moshi.exists() or dry_run:
                run([str(py), "-m", "pip", "install", "--no-deps", str(moshi)], dry_run=dry_run)

    elif key == "pixal3d":
        clone_repo(meta["repo"], root, meta.get("branch"), dry_run=dry_run)
        py = venv_python(root, dry_run=dry_run)
        if not skip_deps:
            run(["apt-get", "update", "-qq"], sudo=True, dry_run=dry_run)
            run(["apt-get", "install", "-y", "git", "cmake", "ninja-build", "build-essential", "python3.12-venv", "libx11-dev", "libegl1-mesa-dev", "libgl1-mesa-dev", "libxext-dev"], sudo=True, dry_run=dry_run)
            install_torch(py, dry_run=dry_run)
            req = root / "requirements.txt"
            if req.exists() or dry_run:
                run([str(py), "-m", "pip", "install", "-r", str(req), "spaces", "nest_asyncio"], dry_run=dry_run)
        if build_pixal3d_trellis_flag:
            build_pixal3d_trellis(root, py, dry_run=dry_run)
        else:
            print("NOTE: Pixal3D TRELLIS.2 native extension build was not requested. Re-run with --build-pixal3d-trellis for full generation support.")

    elif kind == "git_plus_bundled_web":
        clone_repo(meta["repo"], root / "repo", meta.get("branch"), dry_run=dry_run)
        src_dir = REPO_ROOT / "packages" / key
        # Copy package-specific web/app files.
        for child in src_dir.iterdir():
            if child.name == "__pycache__":
                continue
            copy_tree(child, root / child.name, dry_run=dry_run)
        py = venv_python(root, dry_run=dry_run)
        if not skip_deps:
            install_torch(py, dry_run=dry_run)
            req = root / "repo/requirements.txt"
            if req.exists():
                run([str(py), "-m", "pip", "install", "-r", str(req)], dry_run=dry_run)
            pip_install(py, DIFFUSERS_GIT_DEPS if key in {"un0", "triposplat"} else COMMON_WEB_DEPS, dry_run=dry_run)
            if key in {"un0", "agent3dify"}:
                run([str(py), "-m", "pip", "install", "--no-deps", "-e", str(root / "repo")], dry_run=dry_run)
        if download_models and key == "domainshuttle":
            run([str(py), str(root / "scripts/download_models.py")], dry_run=dry_run)

    else:
        raise SystemExit(f"Unsupported package kind for {key}: {kind}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Install optional Spark Dashboard app packages")
    ap.add_argument("packages", nargs="?", default="none", help="all, none, or comma list")
    ap.add_argument("--download-models", action="store_true", help="also download package model weights where implemented")
    ap.add_argument("--skip-deps", action="store_true", help="copy/clone apps only; do not pip/apt install")
    ap.add_argument("--build-pixal3d-trellis", action="store_true", help="when installing pixal3d, also clone/build TRELLIS.2 native CUDA extensions")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    selected = parse_selection(args.packages)
    if not selected:
        print("No optional packages selected.")
        return 0
    if args.build_pixal3d_trellis and "pixal3d" not in selected:
        raise SystemExit("--build-pixal3d-trellis requires selecting the pixal3d package")
    for key in selected:
        install_package(key, download_models=args.download_models, skip_deps=args.skip_deps, build_pixal3d_trellis_flag=args.build_pixal3d_trellis, dry_run=args.dry_run)
    print("\nOptional package install step complete. Units are installed by install.sh; start services with systemctl --user start <unit>.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
