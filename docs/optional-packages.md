# Optional Spark app packages

The base installer deploys the dashboard and text LLM stack. Use `--packages` to add the other Spark app packages that appear in the dashboard.

## Quick install all optional packages

Install dashboard, text model services, and all optional app package scaffolding:

```bash
./install.sh --models all --packages all --start dashboard
```

Install everything and also download optional package model weights where the installer has a downloader:

```bash
./install.sh --models all --packages all --package-models all --start dashboard
```

This is very large. Budget several hundred GiB of disk and a long download/build window.

## Install selected packages

```bash
./install.sh --skip-model-download --packages z-image,qwen-image,flux2,krea2 --start dashboard
```

Later download optional package weights:

```bash
sparkdashboard-install-packages z-image,qwen-image,flux2,krea2 --skip-deps --download-models
```

## Package names

| Package key | Service | Port | Notes |
|---|---|---:|---|
| `personaplex` | `personaplex.service` | 8998 | BNB4/NF4 speech-to-speech; Hugging Face auth may be required |
| `hidream` | `hidream-o1.service` | 7861 | HiDream O1 image app, branch `dev`, CUDA 13 PyTorch |
| `pixal3d` | `pixal3d.service` | 7863 | Pixal3D; add `--build-pixal3d-trellis` to build TRELLIS.2 native extensions |
| `z-image` | `z-image.service` | 7864 | Bundled Spark FastAPI wrapper for `Tongyi-MAI/Z-Image` |
| `qwen-image` | `qwen-image.service` | 7865 | Bundled Spark FastAPI wrapper for `Qwen/Qwen-Image` |
| `flux2` | `flux2.service` | 7866 | Bundled Spark FastAPI wrapper for FLUX.2 bnb-4bit; may need HF auth |
| `domainshuttle` | `domainshuttle-web.service` | 7867 | Upstream repo plus bundled Spark WebUI and downloader |
| `krea2` | `krea-2.service` | 7868 | Bundled Spark FastAPI wrapper and model downloader |
| `agent3dify` | `agent3dify-web.service` | 7869 | Upstream Agent3Dify plus local Qwen WebUI |
| `un0` | `un0-web.service` | 7870 | Upstream Un-0 plus bundled Spark WebUI |
| `triposplat` | `triposplat-web.service` | 7871 | Upstream TripoSplat plus bundled Spark WebUI |

## What the optional installer does

For each selected package, `scripts/install_packages.py` will:

1. create the package root under `/opt`
2. clone the upstream repo if applicable
3. copy bundled Spark wrapper apps/scripts from `packages/<name>/`
4. create a package-local Python venv
5. install CUDA 13 PyTorch and Python dependencies unless `--skip-deps` is set
6. optionally download model weights when `--download-models` is set and implemented

The main `install.sh` always renders the user systemd units so the dashboard controls have matching units available.

## Commands

Install/copy optional package apps only:

```bash
sparkdashboard-install-packages z-image,qwen-image --skip-deps
```

Install apps and Python dependencies:

```bash
sparkdashboard-install-packages z-image,qwen-image
```

Install apps, deps, and model weights:

```bash
sparkdashboard-install-packages z-image,qwen-image --download-models
```

Install Pixal3D with full TRELLIS.2 native CUDA extension build:

```bash
sparkdashboard-install-packages pixal3d --build-pixal3d-trellis
```

Or through the top-level installer:

```bash
./install.sh --skip-model-download --packages pixal3d --build-pixal3d-trellis --start dashboard
```

Dry-run all packages:

```bash
sparkdashboard-install-packages all --dry-run --skip-deps
```

## Start services

```bash
systemctl --user start z-image.service
systemctl --user start qwen-image.service
systemctl --user start flux2.service
systemctl --user start krea-2.service
systemctl --user start domainshuttle-web.service
systemctl --user start un0-web.service
systemctl --user start triposplat-web.service
systemctl --user start agent3dify-web.service
systemctl --user start hidream-o1.service
systemctl --user start pixal3d.service
systemctl --user start personaplex.service
```

Use the dashboard at `http://<spark-host>:7862` to start/stop many of these through fixed allow-listed controls.

## Important caveats

- These optional packages are much heavier and less uniform than the text LLM stack. Some are upstream research repos with native CUDA extensions.
- Pixal3D full generation needs the TRELLIS.2 extension build. It is opt-in with `--build-pixal3d-trellis` because it clones `microsoft/TRELLIS.2`, builds native CUDA extensions from source, and can take a long time.
- PersonaPlex and FLUX.2 may require Hugging Face login/access. Run `huggingface-cli login` on the target Spark before downloading gated assets.
- Do not run every heavy app simultaneously. Most image/video apps should be treated as on-demand services.
- The bundled FastAPI wrappers intentionally store generated images/jobs under each `/opt/<package>/outputs` or `/opt/<package>/jobs` directory.

## Verify

```bash
systemctl --user status z-image.service --no-pager
curl -fsS http://127.0.0.1:7864/health | python3 -m json.tool

systemctl --user status krea-2.service --no-pager
curl -fsS http://127.0.0.1:7868/health | python3 -m json.tool
```

Use `journalctl` when a service is slow to start:

```bash
journalctl --user -u z-image.service -f
journalctl --user -u domainshuttle-web.service -f
```
