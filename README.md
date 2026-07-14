# Spark Dashboard

Portable installer for Joao's NVIDIA Spark / GB10 local model dashboard and text-model services.

It installs:

- Spark Dashboard (`spark-dashboard.service`) on port `7862`
- Qwen 3.6 35B NVFP4 vLLM service on port `8000`
- Ornith 1.0 35B vLLM service on port `8001`
- Mistral Medium 3.5 128B NVFP4 vLLM service on port `8002`
- Qwen no-think OpenAI-compatible proxy on port `8010`
- Ornith no-think OpenAI-compatible proxy on port `8011`

The dashboard is dependency-light Python stdlib. Model serving uses Docker and `vllm/vllm-openai:nightly`.

Detailed guide: [Deployment and usage](docs/deployment-and-usage.md)

Optional app packages: [Optional Spark app packages](docs/optional-packages.md)

## One-command install on a fresh Spark

```bash
curl -fsSL https://raw.githubusercontent.com/joaoha/sparkdashboard/07c25d552b5ad461a366b9a45eeb77ebe7ed3acf/bootstrap.sh | SPARKDASHBOARD_REF=07c25d552b5ad461a366b9a45eeb77ebe7ed3acf bash -s -- --models all --packages all --start dashboard
```

This command pins both the bootstrap script and installed source tree to immutable Git commit `07c25d552b5ad461a366b9a45eeb77ebe7ed3acf`. It uses a temporary checkout and can be run again after any failed install. It reuses complete optional-package checkouts, quarantines incomplete clones instead of deleting them, and resumes from the failed dependency step. Add `--package-models all` if you also want optional app model weights downloaded in the same run. This is very large; see [Optional Spark app packages](docs/optional-packages.md).

To install the dashboard/services first and download models later:

```bash
curl -fsSL https://raw.githubusercontent.com/joaoha/sparkdashboard/07c25d552b5ad461a366b9a45eeb77ebe7ed3acf/bootstrap.sh | SPARKDASHBOARD_REF=07c25d552b5ad461a366b9a45eeb77ebe7ed3acf bash -s -- --skip-model-download --start dashboard
sparkdashboard-download-models qwen,ornith,mistral --model-dir ~/models/hf
```

## Requirements

- Ubuntu/Linux on NVIDIA Spark / GB10 or similar NVIDIA machine
- NVIDIA driver and container runtime working with Docker (`docker run --gpus all ...`)
- Python 3
- `systemd --user`
- Internet access to Hugging Face and Docker Hub
- Hugging Face access if any model repository requires auth/terms acceptance

The installer can install `docker.io` via apt if Docker is missing, but it does not fully bootstrap NVIDIA drivers or the NVIDIA Container Toolkit on arbitrary Linux distributions.

## Installed commands

- `sparkdashboard-download-models` — downloads text model snapshots using `huggingface_hub`
- `sparkdashboard-install-packages` — installs optional Spark app packages such as Z-Image, Qwen-Image, FLUX.2, Krea-2, DomainShuttle, Un-0, TripoSplat, Agent3Dify, HiDream, Pixal3D, and PersonaPlex. Use `sparkdashboard-install-packages pixal3d --build-pixal3d-trellis` for Pixal3D's full TRELLIS.2 extension build.
- `sparkdashboard-status` — shows model/proxy service state and `/v1/models` probes

## Service map

| Service | Port | Purpose |
|---|---:|---|
| `spark-dashboard.service` | 7862 | Dashboard UI/API |
| `qwen-nvfp4-vllm.service` | 8000 | Qwen direct vLLM API |
| `qwen-no-think-proxy.service` | 8010 | Qwen fast/no-think OpenAI-compatible proxy |
| `ornith-vllm.service` | 8001 | Ornith direct vLLM API |
| `ornith-no-think-proxy.service` | 8011 | Ornith fast/no-think OpenAI-compatible proxy |
| `mistral-medium-vllm.service` | 8002 | Mistral direct vLLM API |

Qwen, Ornith, and Mistral model services are mutually exclusive by systemd/Docker cleanup. The lightweight proxy services may remain running; if their upstream model is stopped they return upstream errors.

## OpenCode model names

Use these providers in OpenCode if you also configure OpenCode against this Spark:

- `qwen-fast/Qwen3.6-35B-A3B-NVFP4` -> `http://<spark>:8010/v1`
- `qwen-think/Qwen3.6-35B-A3B-NVFP4` -> `http://<spark>:8000/v1`
- `ornith-fast/Ornith-1.0-35B` -> `http://<spark>:8011/v1`
- `ornith-think/Ornith-1.0-35B` -> `http://<spark>:8001/v1`
- `mistral-local/Mistral-Medium-3.5-128B-NVFP4` -> `http://<spark>:8002/v1`

## Custom install options

```bash
curl -fsSL https://raw.githubusercontent.com/joaoha/sparkdashboard/07c25d552b5ad461a366b9a45eeb77ebe7ed3acf/bootstrap.sh | SPARKDASHBOARD_REF=07c25d552b5ad461a366b9a45eeb77ebe7ed3acf bash -s -- \
  --install-root /opt/spark-dashboard \
  --model-dir /home/$USER/models/hf \
  --public-host my-spark.local \
  --dashboard-port 7862 \
  --models qwen,ornith,mistral \
  --start dashboard
```

`--models` accepts `all`, `none`, or a comma-separated subset: `qwen,ornith,mistral`.

## Verify

```bash
systemctl --user status spark-dashboard.service --no-pager
curl -fsS http://127.0.0.1:7862/api/status | python3 -m json.tool
sparkdashboard-status
```

## Notes

- Mistral Medium 3.5 is very slow on Spark compared with Qwen/Ornith and has a conservative default context of `37888`.
- The no-think proxy is intentionally tiny and stdlib-only. It adds `/no_think` and `chat_template_kwargs.enable_thinking=false` for Qwen-style models so coding agents receive visible assistant content.
- The dashboard UI still displays optional services from Joao's broader Spark lab. On a fresh install without those services, they simply show inactive/unhealthy.
