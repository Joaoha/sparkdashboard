# Spark Dashboard deployment and usage guide

This document explains how to deploy Spark Dashboard on a freshly installed NVIDIA Spark / GB10-style machine, download the supported local models, operate the dashboard, and connect tools such as OpenCode.

Repository:

```text
https://github.com/joaoha/sparkdashboard
```

## 1. What this installs

The installer deploys a dashboard plus local text-model services:

| Component | User systemd service | Port | Purpose |
|---|---|---:|---|
| Spark Dashboard | `spark-dashboard.service` | `7862` | Browser UI, service status, controls, benchmarks |
| Qwen vLLM | `qwen-nvfp4-vllm.service` | `8000` | Direct Qwen OpenAI-compatible API |
| Qwen no-think proxy | `qwen-no-think-proxy.service` | `8010` | Qwen fast/no-think API for coding agents |
| Ornith vLLM | `ornith-vllm.service` | `8001` | Direct Ornith OpenAI-compatible API |
| Ornith no-think proxy | `ornith-no-think-proxy.service` | `8011` | Ornith fast/no-think API for coding agents |
| Mistral Medium vLLM | `mistral-medium-vllm.service` | `8002` | Direct Mistral Medium 3.5 API |

The dashboard can also display optional Joao home-lab services if they exist on a machine. On a clean Spark with only this package installed, optional services show as inactive/unhealthy, which is expected.

## 2. Hardware and OS assumptions

Recommended target:

- NVIDIA Spark / GB10 or similar NVIDIA Linux workstation
- Ubuntu-like Linux with `systemd --user`
- NVIDIA driver installed
- Docker installed and able to run GPU containers
- Enough disk space for models
- Internet access to Docker Hub and Hugging Face

The installer can install `docker.io` via `apt` if Docker is missing, but it does not install NVIDIA drivers or fully bootstrap NVIDIA Container Toolkit on arbitrary distributions.

Before installing, confirm GPU Docker works:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If that command fails, fix the NVIDIA driver / container runtime first.

## 3. Disk space planning

The exact model sizes can change, but plan for roughly:

| Model | Hugging Face repo | Local directory | Approx disk |
|---|---|---|---:|
| Qwen | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `Qwen3.6-35B-A3B-NVFP4` | ~24+ GiB |
| Ornith | `deepreinforce-ai/Ornith-1.0-35B` | `Ornith-1.0-35B` | ~66+ GiB |
| Mistral | `nvidia/Mistral-Medium-3.5-128B-NVFP4` | `Mistral-Medium-3.5-128B-NVFP4` | ~90+ GiB |

Budget at least 200 GiB free for all three models plus Docker image cache and temporary Hugging Face download data.

Default model directory:

```text
~/models/hf
```

You can choose another directory with `--model-dir`.

## 4. Fresh-machine quick install

On the new Spark:

```bash
curl -fsSL https://raw.githubusercontent.com/joaoha/sparkdashboard/24fca5306972211b2968777176e8a367c67717a9/bootstrap.sh | SPARKDASHBOARD_REF=24fca5306972211b2968777176e8a367c67717a9 bash -s -- --models all --packages all --start dashboard
```

Add `--package-models all` if this fresh machine should also download optional app model weights in the same run.

The command pins both the bootstrap script and installed source tree to immutable Git commit `24fca5306972211b2968777176e8a367c67717a9`. The bootstrap checks out that installer revision in a disposable temporary directory, so it is safe to rerun after an interrupted or failed install. Completed optional-package Git checkouts are reused; a partial clone is preserved as a timestamped `.interrupted-clone-*` sibling and replaced with a clean retry.

This will:

1. copy package files into `/opt/spark-dashboard`
2. render user systemd units into `~/.config/systemd/user/`
3. enable the dashboard and proxy units
4. create convenience commands:
   - `sparkdashboard-download-models`
   - `sparkdashboard-install-packages`
   - `sparkdashboard-status`
5. download selected Hugging Face text model snapshots
6. install selected optional app packages when `--packages` is used
7. start the dashboard

Open the dashboard:

```text
http://<spark-hostname>:7862
```

Example:

```text
http://my-spark.local:7862
```

## 5. Install first, download models later

If you want the dashboard online before downloading large model snapshots:

```bash
curl -fsSL https://raw.githubusercontent.com/joaoha/sparkdashboard/24fca5306972211b2968777176e8a367c67717a9/bootstrap.sh | SPARKDASHBOARD_REF=24fca5306972211b2968777176e8a367c67717a9 bash -s -- --skip-model-download --start dashboard
```

Then download models individually:

```bash
sparkdashboard-download-models qwen --model-dir ~/models/hf
sparkdashboard-download-models ornith --model-dir ~/models/hf
sparkdashboard-download-models mistral --model-dir ~/models/hf
```

Or all at once:

```bash
sparkdashboard-download-models qwen,ornith,mistral --model-dir ~/models/hf
```

## 5. Install optional app packages

The dashboard can also install the optional app packages shown in its service grid:

```bash
sparkdashboard-install-packages all
```

Install optional app packages and download optional package weights where implemented:

```bash
sparkdashboard-install-packages all --download-models
```

For selected packages:

```bash
sparkdashboard-install-packages z-image,qwen-image,flux2,krea2
```

See [Optional Spark app packages](optional-packages.md) for package names, ports, caveats, and verification.

## 6. Custom install options

Common custom install:

```bash
curl -fsSL https://raw.githubusercontent.com/joaoha/sparkdashboard/24fca5306972211b2968777176e8a367c67717a9/bootstrap.sh | SPARKDASHBOARD_REF=24fca5306972211b2968777176e8a367c67717a9 bash -s -- \
  --install-root /opt/spark-dashboard \
  --model-dir /data/models/hf \
  --public-host my-spark.local \
  --dashboard-port 7862 \
  --models qwen,ornith \
  --start dashboard
```

Installer options:

| Option | Meaning |
|---|---|
| `--install-root PATH` | Where package files are installed. Default `/opt/spark-dashboard` |
| `--model-dir PATH` | Parent directory for model snapshots. Default `~/models/hf` |
| `--public-host HOST` | Hostname used in dashboard links. Default `hostname -f` or `hostname` |
| `--dashboard-port PORT` | Dashboard port. Default `7862` |
| `--models all` | Download all known text models |
| `--packages all` | Install all optional Spark app package scaffolding/dependencies |
| `--package-models all` | Also download optional app model weights where implemented |
| `--skip-package-deps` | Copy/clone optional package apps only; skip apt/pip dependency install |
| `--build-pixal3d-trellis` | With `--packages pixal3d`, build TRELLIS.2 CUDA extensions from source |
| `--models none` | Do not download models |
| `--models qwen,ornith,mistral` | Download selected models |
| `--start none` | Do not start services after install |
| `--start dashboard` | Start/restart dashboard only |
| `--start qwen` | Start dashboard, proxies, and Qwen vLLM |
| `--skip-model-download` | Equivalent to `--models none` |
| `--skip-docker-install` | Do not attempt apt Docker installation |
| `--dry-run` | Render/check plan without writing files or starting services |

Dry-run example:

```bash
curl -fsSL https://raw.githubusercontent.com/joaoha/sparkdashboard/24fca5306972211b2968777176e8a367c67717a9/bootstrap.sh | SPARKDASHBOARD_REF=24fca5306972211b2968777176e8a367c67717a9 bash -s -- --dry-run --models none --start none \
  --install-root /tmp/spark-dashboard-test \
  --model-dir /tmp/spark-models \
  --public-host spark-test.local \
  --dashboard-port 17862
```

## 7. Verify installation

Dashboard service:

```bash
systemctl --user status spark-dashboard.service --no-pager
curl -fsS http://127.0.0.1:7862/api/status | python3 -m json.tool
```

Convenience status command:

```bash
sparkdashboard-status
```

Expected output includes systemd state plus `/v1/models` probes for the model and proxy ports.

Dashboard page:

```bash
curl -fsSI http://127.0.0.1:7862/
```

## 8. Starting and stopping models

Qwen is the recommended default interactive coding model:

```bash
systemctl --user start qwen-no-think-proxy.service ornith-no-think-proxy.service
systemctl --user start qwen-nvfp4-vllm.service
```

Stop Qwen:

```bash
systemctl --user stop qwen-nvfp4-vllm.service
```

Start Ornith:

```bash
systemctl --user start ornith-vllm.service
```

Start Mistral:

```bash
systemctl --user start mistral-medium-vllm.service
```

Qwen, Ornith, and Mistral are treated as mutually exclusive text model services. The launch scripts stop conflicting model containers before starting the requested model.

The proxy services are lightweight and can stay running even when their upstream model is stopped. If a proxy is running but its upstream model is stopped, OpenAI API calls through the proxy will return upstream errors until the model starts.

## 9. Reading logs

Dashboard:

```bash
journalctl --user -u spark-dashboard.service -f
```

Qwen:

```bash
journalctl --user -u qwen-nvfp4-vllm.service -f
```

Ornith:

```bash
journalctl --user -u ornith-vllm.service -f
```

Mistral:

```bash
journalctl --user -u mistral-medium-vllm.service -f
```

No-think proxies:

```bash
journalctl --user -u qwen-no-think-proxy.service -f
journalctl --user -u ornith-no-think-proxy.service -f
```

## 10. API endpoints

Direct model APIs:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool  # Qwen direct
curl -fsS http://127.0.0.1:8001/v1/models | python3 -m json.tool  # Ornith direct
curl -fsS http://127.0.0.1:8002/v1/models | python3 -m json.tool  # Mistral direct
```

Fast/no-think proxy APIs:

```bash
curl -fsS http://127.0.0.1:8010/v1/models | python3 -m json.tool  # Qwen fast proxy
curl -fsS http://127.0.0.1:8011/v1/models | python3 -m json.tool  # Ornith fast proxy
```

Minimal chat test through Qwen fast proxy:

```bash
curl -fsS http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Qwen3.6-35B-A3B-NVFP4",
    "messages":[{"role":"user","content":"Reply exactly: QWEN_OK"}],
    "temperature":0,
    "max_tokens":16
  }' | python3 -m json.tool
```

## 11. What the no-think proxies do

Qwen and Ornith are Qwen-style reasoning models. In direct thinking mode, they may return a `reasoning` field before normal assistant `content`. Some coding agents expect visible assistant content on every turn and can behave poorly if output is reasoning-only.

The no-think proxy is a small Python stdlib proxy that mutates only `/v1/chat/completions` requests:

- adds `chat_template_kwargs.enable_thinking=false`
- prefixes the first user message with `/no_think`

It leaves `/v1/models`, streaming behavior, tools, and most other request fields untouched.

Use direct ports for thinking:

- Qwen direct thinking: `http://<spark>:8000/v1`
- Ornith direct thinking: `http://<spark>:8001/v1`

Use proxy ports for fast/no-think coding agents:

- Qwen fast/no-think: `http://<spark>:8010/v1`
- Ornith fast/no-think: `http://<spark>:8011/v1`

## 12. OpenCode setup

OpenCode can use the model APIs as custom OpenAI-compatible providers.

Suggested model names:

| OpenCode model | Base URL | Meaning |
|---|---|---|
| `qwen-fast/Qwen3.6-35B-A3B-NVFP4` | `http://<spark>:8010/v1` | Recommended default; no-think proxy |
| `qwen-think/Qwen3.6-35B-A3B-NVFP4` | `http://<spark>:8000/v1` | Direct Qwen thinking mode |
| `ornith-fast/Ornith-1.0-35B` | `http://<spark>:8011/v1` | Ornith no-think proxy |
| `ornith-think/Ornith-1.0-35B` | `http://<spark>:8001/v1` | Direct Ornith thinking mode |
| `mistral-local/Mistral-Medium-3.5-128B-NVFP4` | `http://<spark>:8002/v1` | Direct Mistral |

Example OpenCode provider snippet for a client machine:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "qwen-fast/Qwen3.6-35B-A3B-NVFP4",
  "provider": {
    "qwen-fast": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Qwen 3.6 35B (fast / no-think proxy)",
      "options": {
        "baseURL": "http://YOUR-SPARK-HOST:8010/v1",
        "apiKey": "local-qwen-fast"
      },
      "models": {
        "Qwen3.6-35B-A3B-NVFP4": {
          "name": "Qwen 3.6 35B A3B NVFP4 — fast / no-think",
          "tools": true,
          "reasoning": false,
          "limit": {"context": 262144, "output": 65536}
        }
      }
    }
  }
}
```

Smoke test:

```bash
opencode models qwen-fast --refresh
opencode run 'Respond with exactly: OPENCODE_OK' --model qwen-fast/Qwen3.6-35B-A3B-NVFP4
```

For direct thinking mode, use OpenCode's thinking flag:

```bash
opencode run 'Respond with exactly: OPENCODE_THINK_OK' \
  --model qwen-think/Qwen3.6-35B-A3B-NVFP4 \
  --thinking
```

## 13. Dashboard usage

Open the dashboard:

```text
http://<spark-host>:7862
```

Primary sections:

- Overview: system status and high-level telemetry
- Services: service cards for model/web services
- Controls: fixed allow-listed start/stop/unload actions
- Benchmarks: active LLM speed benchmark history
- Telemetry: CPU/GPU/memory and process residency

Controls intentionally run only fixed allow-listed commands. The dashboard does not accept arbitrary shell input.

If the control panel says a command was requested, follow progress with the matching service journal:

```bash
journalctl --user -u qwen-nvfp4-vllm.service -f
```

## 14. Benchmarks

The dashboard benchmark button targets the active text LLM. It records result history to:

```text
/opt/spark-dashboard/benchmarks.jsonl
```

Override with:

```bash
SPARK_BENCHMARK_PATH=/path/to/benchmarks.jsonl
```

Qwen and Ornith benchmarks use no-think style prompts. Mistral does not accept Qwen-specific `chat_template_kwargs`; the dashboard handles Mistral separately.

## 15. Upgrade an existing install

On the Spark:

```bash
curl -fsSL https://raw.githubusercontent.com/joaoha/sparkdashboard/24fca5306972211b2968777176e8a367c67717a9/bootstrap.sh | SPARKDASHBOARD_REF=24fca5306972211b2968777176e8a367c67717a9 bash -s -- --skip-model-download --start dashboard
```

This refreshes installed app files, scripts, and systemd units without redownloading models.

Then restart or start the model you want:

```bash
systemctl --user restart qwen-nvfp4-vllm.service
```

## 16. Uninstall

Stop services:

```bash
systemctl --user stop \
  spark-dashboard.service \
  qwen-nvfp4-vllm.service \
  ornith-vllm.service \
  mistral-medium-vllm.service \
  qwen-no-think-proxy.service \
  ornith-no-think-proxy.service
```

Disable services:

```bash
systemctl --user disable \
  spark-dashboard.service \
  qwen-no-think-proxy.service \
  ornith-no-think-proxy.service
```

Remove user units:

```bash
rm -f ~/.config/systemd/user/spark-dashboard.service \
      ~/.config/systemd/user/qwen-nvfp4-vllm.service \
      ~/.config/systemd/user/ornith-vllm.service \
      ~/.config/systemd/user/mistral-medium-vllm.service \
      ~/.config/systemd/user/qwen-no-think-proxy.service \
      ~/.config/systemd/user/ornith-no-think-proxy.service
systemctl --user daemon-reload
```

Remove package files and symlinks:

```bash
sudo rm -rf /opt/spark-dashboard
sudo rm -f /usr/local/bin/sparkdashboard-download-models /usr/local/bin/sparkdashboard-status
```

Model snapshots are not removed automatically. If you want to remove them too:

```bash
rm -rf ~/models/hf/Qwen3.6-35B-A3B-NVFP4 \
       ~/models/hf/Ornith-1.0-35B \
       ~/models/hf/Mistral-Medium-3.5-128B-NVFP4
```

## 17. Troubleshooting

### Dashboard does not load

```bash
systemctl --user status spark-dashboard.service --no-pager
journalctl --user -u spark-dashboard.service -n 120 --no-pager
```

Check port:

```bash
ss -ltnp | grep ':7862'
```

### Model service says running but API fails

Check the direct model endpoint:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

Check logs:

```bash
journalctl --user -u qwen-nvfp4-vllm.service -n 200 --no-pager
```

If it is still loading, wait. vLLM model load can take a while, especially Mistral.

### OpenCode gets an API error

Test the exact base URL OpenCode uses.

For Qwen fast:

```bash
curl -fsS http://127.0.0.1:8010/v1/models | python3 -m json.tool
```

If direct Qwen on `8000` works but fast proxy on `8010` fails:

```bash
systemctl --user restart qwen-no-think-proxy.service
```

For Ornith fast:

```bash
curl -fsS http://127.0.0.1:8011/v1/models | python3 -m json.tool
systemctl --user restart ornith-no-think-proxy.service
```

### Docker permission denied

If Docker works with `sudo` but not your user:

```bash
sudo usermod -aG docker "$USER"
```

Then log out and back in, or reboot.

### Hugging Face download fails

Make sure you accepted any model license/terms and have credentials if needed:

```bash
huggingface-cli login
```

Then rerun:

```bash
sparkdashboard-download-models qwen,ornith,mistral --model-dir ~/models/hf
```

### Mistral is slow

Expected. Mistral Medium 3.5 128B is a dense 128B model and is much slower on Spark than Qwen/Ornith. Treat it as a high-quality consultant model, not the default interactive coding model.

### Out of memory / low unified memory

Stop conflicting heavyweight services:

```bash
systemctl --user stop qwen-nvfp4-vllm.service ornith-vllm.service mistral-medium-vllm.service
```

Then start only the one model you need.

## 18. Files installed

Default install root:

```text
/opt/spark-dashboard
```

Important paths:

```text
/opt/spark-dashboard/app/server.py
/opt/spark-dashboard/app/no_think_proxy.py
/opt/spark-dashboard/bin/*.sh
/opt/spark-dashboard/scripts/download_models.py
/opt/spark-dashboard/config/models.json
~/.config/systemd/user/*.service
```

Default model snapshots:

```text
~/models/hf/Qwen3.6-35B-A3B-NVFP4
~/models/hf/Ornith-1.0-35B
~/models/hf/Mistral-Medium-3.5-128B-NVFP4
```
