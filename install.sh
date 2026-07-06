#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT=${INSTALL_ROOT:-/opt/spark-dashboard}
MODEL_DIR=${SPARK_MODEL_DIR:-$HOME/models/hf}
PUBLIC_HOST=${SPARK_PUBLIC_HOST:-$(hostname -f 2>/dev/null || hostname)}
DASHBOARD_PORT=${SPARK_DASHBOARD_PORT:-7862}
VLLM_IMAGE=${SPARK_VLLM_IMAGE:-vllm/vllm-openai:nightly}
MODELS=${SPARK_MODELS:-all}
START=${SPARK_START:-dashboard}
INSTALL_DOCKER=${SPARK_INSTALL_DOCKER:-auto}
DRY_RUN=${SPARK_DRY_RUN:-0}

usage() {
  cat <<USAGE
Usage: ./install.sh [options]

Options:
  --install-root PATH       Default: /opt/spark-dashboard
  --model-dir PATH          Default: ~/models/hf
  --public-host HOST        Default: hostname -f
  --dashboard-port PORT     Default: 7862
  --models LIST             all|none|qwen,ornith,mistral (default: all)
  --start WHAT              none|dashboard|qwen (default: dashboard)
  --skip-model-download     Same as --models none
  --skip-docker-install     Do not attempt Docker installation if docker is missing
  --dry-run                 Validate/render plan without writing files or starting services
  -h, --help                Show this help

One-command remote use:
  git clone https://github.com/joaoha/sparkdashboard.git && cd sparkdashboard && ./install.sh --models all --start dashboard
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --install-root) INSTALL_ROOT="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --public-host) PUBLIC_HOST="$2"; shift 2 ;;
    --dashboard-port) DASHBOARD_PORT="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --start) START="$2"; shift 2 ;;
    --skip-model-download) MODELS=none; shift ;;
    --skip-docker-install) INSTALL_DOCKER=never; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SUDO=sudo
if [ "$(id -u)" -eq 0 ]; then SUDO=; fi

require_cmd() { command -v "$1" >/dev/null 2>&1 || return 1; }

if ! require_cmd python3; then
  echo "python3 is required" >&2
  exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "Spark Dashboard dry run"
  echo "  repo:           $REPO_DIR"
  echo "  install root:   $INSTALL_ROOT"
  echo "  model dir:      $MODEL_DIR"
  echo "  public host:    $PUBLIC_HOST"
  echo "  dashboard port: $DASHBOARD_PORT"
  echo "  models:         $MODELS"
  echo "  start:          $START"
  tmp=$(mktemp -d)
  python3 - "$REPO_DIR" "$tmp" "$INSTALL_ROOT" "$MODEL_DIR" "$PUBLIC_HOST" "$DASHBOARD_PORT" "$VLLM_IMAGE" <<'PY'
from pathlib import Path
import sys
repo = Path(sys.argv[1]); out = Path(sys.argv[2])
values = {
    "INSTALL_ROOT": sys.argv[3],
    "MODEL_DIR": sys.argv[4],
    "PUBLIC_HOST": sys.argv[5],
    "DASHBOARD_PORT": sys.argv[6],
    "VLLM_IMAGE": sys.argv[7],
}
for src in sorted((repo / "systemd/user").glob("*.service.in")):
    text = src.read_text()
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    dst = out / src.name.removesuffix(".in")
    dst.write_text(text)
    print(dst)
PY
  python3 -m py_compile "$REPO_DIR/app/server.py" "$REPO_DIR/app/no_think_proxy.py" "$REPO_DIR/scripts/download_models.py"
  bash -n "$REPO_DIR/install.sh" "$REPO_DIR"/bin/*.sh "$REPO_DIR/scripts/smoke.sh"
  python3 -m json.tool "$REPO_DIR/config/models.json" >/dev/null
  rm -rf "$tmp"
  echo "Dry run OK."
  exit 0
fi

if ! require_cmd docker; then
  if [ "$INSTALL_DOCKER" = "never" ]; then
    echo "Docker is missing and --skip-docker-install was set." >&2
    exit 1
  fi
  if require_cmd apt-get; then
    echo "Installing Docker via apt..."
    $SUDO apt-get update
    $SUDO apt-get install -y docker.io
    $SUDO systemctl enable --now docker || true
    $SUDO usermod -aG docker "$USER" || true
    echo "Docker installed. You may need to log out/in for docker group membership; installer will continue using sudo where needed."
  else
    echo "Docker is required. Install Docker + NVIDIA Container Toolkit, then rerun." >&2
    exit 1
  fi
fi

if ! docker info >/dev/null 2>&1; then
  if sudo docker info >/dev/null 2>&1; then
    echo "Docker works via sudo, but user Docker access is not active. Add user to docker group or re-login. Continuing for file install; services may need group membership."
  else
    echo "Docker daemon is not reachable." >&2
    exit 1
  fi
fi

mkdir -p "$HOME/.config/systemd/user" "$MODEL_DIR"
$SUDO install -d -m 0755 "$INSTALL_ROOT" "$INSTALL_ROOT/app" "$INSTALL_ROOT/bin" "$INSTALL_ROOT/config" "$INSTALL_ROOT/scripts"
$SUDO cp -R "$REPO_DIR/app/." "$INSTALL_ROOT/app/"
$SUDO cp -R "$REPO_DIR/bin/." "$INSTALL_ROOT/bin/"
$SUDO cp -R "$REPO_DIR/config/." "$INSTALL_ROOT/config/"
$SUDO cp -R "$REPO_DIR/scripts/." "$INSTALL_ROOT/scripts/"
$SUDO chmod +x "$INSTALL_ROOT/bin/"*.sh "$INSTALL_ROOT/scripts/download_models.py"
$SUDO chown -R root:root "$INSTALL_ROOT" || true

# Convenience command for model downloads.
$SUDO ln -sf "$INSTALL_ROOT/scripts/download_models.py" /usr/local/bin/sparkdashboard-download-models
$SUDO ln -sf "$INSTALL_ROOT/bin/status-text-models.sh" /usr/local/bin/sparkdashboard-status

render_unit() {
  local src="$1" dst="$2"
  python3 - "$src" "$dst" <<PY
from pathlib import Path
import sys
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()
values = {
    "INSTALL_ROOT": "$INSTALL_ROOT",
    "MODEL_DIR": "$MODEL_DIR",
    "PUBLIC_HOST": "$PUBLIC_HOST",
    "DASHBOARD_PORT": "$DASHBOARD_PORT",
    "VLLM_IMAGE": "$VLLM_IMAGE",
}
for k, v in values.items():
    text = text.replace("{{" + k + "}}", v)
dst.write_text(text)
PY
}

for tmpl in "$REPO_DIR"/systemd/user/*.service.in; do
  unit=$(basename "$tmpl" .in)
  render_unit "$tmpl" "$HOME/.config/systemd/user/$unit"
done

systemctl --user daemon-reload
systemctl --user enable spark-dashboard.service qwen-no-think-proxy.service ornith-no-think-proxy.service >/dev/null
# Model services are installed but not enabled: start them from dashboard or manually.

if [ "$MODELS" != "none" ]; then
  echo "Preparing Python venv for Hugging Face downloads..."
  python3 -m venv "$INSTALL_ROOT/.venv-download" 2>/dev/null || $SUDO python3 -m venv "$INSTALL_ROOT/.venv-download"
  if [ ! -x "$INSTALL_ROOT/.venv-download/bin/python" ]; then
    echo "Could not create venv at $INSTALL_ROOT/.venv-download" >&2
    exit 1
  fi
  $SUDO "$INSTALL_ROOT/.venv-download/bin/python" -m pip install -U pip huggingface_hub hf_transfer
  "$INSTALL_ROOT/.venv-download/bin/python" "$INSTALL_ROOT/scripts/download_models.py" "$MODELS" --model-dir "$MODEL_DIR"
fi

if [ "$START" = "dashboard" ] || [ "$START" = "qwen" ]; then
  systemctl --user restart spark-dashboard.service
fi
if [ "$START" = "qwen" ]; then
  systemctl --user start qwen-no-think-proxy.service ornith-no-think-proxy.service
  systemctl --user start qwen-nvfp4-vllm.service
fi

cat <<DONE

Spark Dashboard installed.

Dashboard:      http://$PUBLIC_HOST:$DASHBOARD_PORT
Install root:   $INSTALL_ROOT
Model dir:      $MODEL_DIR
Status command: sparkdashboard-status
Download cmd:   sparkdashboard-download-models qwen,ornith,mistral --model-dir "$MODEL_DIR"

Installed user services:
  spark-dashboard.service
  qwen-nvfp4-vllm.service
  ornith-vllm.service
  mistral-medium-vllm.service
  qwen-no-think-proxy.service
  ornith-no-think-proxy.service

Start examples:
  systemctl --user start spark-dashboard.service
  systemctl --user start qwen-nvfp4-vllm.service
  systemctl --user start ornith-vllm.service
  systemctl --user start mistral-medium-vllm.service

DONE
