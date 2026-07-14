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
PACKAGES=${SPARK_PACKAGES:-none}
PACKAGE_MODELS=${SPARK_PACKAGE_MODELS:-none}
SKIP_PACKAGE_DEPS=${SPARK_SKIP_PACKAGE_DEPS:-0}
BUILD_PIXAL3D_TRELLIS=${SPARK_BUILD_PIXAL3D_TRELLIS:-0}
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
  --packages LIST           all|none or comma list of optional apps (default: none)
  --package-models LIST     all|none or comma list; download optional package weights (default: none)
  --skip-package-deps       Copy/clone optional packages only; skip apt/pip deps
  --build-pixal3d-trellis   When installing Pixal3D, also build TRELLIS.2 CUDA extensions
  --start WHAT              none|dashboard|qwen (default: dashboard)
  --skip-model-download     Same as --models none
  --skip-docker-install     Do not attempt Docker installation if docker is missing
  --dry-run                 Validate/render plan without writing files or starting services
  -h, --help                Show this help

One-command remote use:
  See README.md for the current pinned one-command installer.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --install-root) INSTALL_ROOT="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --public-host) PUBLIC_HOST="$2"; shift 2 ;;
    --dashboard-port) DASHBOARD_PORT="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --packages) PACKAGES="$2"; shift 2 ;;
    --package-models) PACKAGE_MODELS="$2"; shift 2 ;;
    --skip-package-deps) SKIP_PACKAGE_DEPS=1; shift ;;
    --build-pixal3d-trellis) BUILD_PIXAL3D_TRELLIS=1; shift ;;
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
  echo "  packages:       $PACKAGES"
  echo "  package models: $PACKAGE_MODELS"
  echo "  pixal3d trellis:$BUILD_PIXAL3D_TRELLIS"
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
    "PERSONAPLEX_ROOT": "/opt/personaplex-bnb4",
    "HIDREAM_ROOT": "/opt/hidream-o1-image-dev-2604",
    "PIXAL3D_ROOT": "/opt/Pixal3D",
    "Z_IMAGE_ROOT": "/opt/z-image",
    "QWEN_IMAGE_ROOT": "/opt/qwen-image",
    "FLUX2_ROOT": "/opt/flux2",
    "DOMAINSHUTTLE_ROOT": "/opt/domainshuttle",
    "KREA2_ROOT": "/opt/krea-2",
    "AGENT3DIFY_ROOT": "/opt/agent3dify",
    "UN0_ROOT": "/opt/un0",
    "TRIPOSPLAT_ROOT": "/opt/triposplat",
}
for src in sorted((repo / "systemd/user").glob("*.service.in")):
    text = src.read_text()
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    dst = out / src.name.removesuffix(".in")
    dst.write_text(text)
    print(dst)
PY
  python3 -m py_compile "$REPO_DIR/app/server.py" "$REPO_DIR/app/no_think_proxy.py" "$REPO_DIR/scripts/download_models.py" "$REPO_DIR/scripts/install_packages.py" "$REPO_DIR"/packages/*/*.py "$REPO_DIR"/packages/*/*/*.py 2>/dev/null || python3 -m py_compile "$REPO_DIR/app/server.py" "$REPO_DIR/app/no_think_proxy.py" "$REPO_DIR/scripts/download_models.py" "$REPO_DIR/scripts/install_packages.py"
  bash -n "$REPO_DIR/install.sh" "$REPO_DIR"/bin/*.sh "$REPO_DIR/scripts/smoke.sh"
  python3 -m json.tool "$REPO_DIR/config/models.json" >/dev/null
  python3 -m json.tool "$REPO_DIR/config/packages.json" >/dev/null
  dry_pkg_args=("$PACKAGES" --dry-run)
  if [ "$BUILD_PIXAL3D_TRELLIS" = "1" ]; then dry_pkg_args+=(--build-pixal3d-trellis); fi
  "$REPO_DIR/scripts/install_packages.py" "${dry_pkg_args[@]}"
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
$SUDO install -d -m 0755 "$INSTALL_ROOT" "$INSTALL_ROOT/app" "$INSTALL_ROOT/bin" "$INSTALL_ROOT/config" "$INSTALL_ROOT/scripts" "$INSTALL_ROOT/packages"
$SUDO cp -R "$REPO_DIR/app/." "$INSTALL_ROOT/app/"
$SUDO cp -R "$REPO_DIR/bin/." "$INSTALL_ROOT/bin/"
$SUDO cp -R "$REPO_DIR/config/." "$INSTALL_ROOT/config/"
$SUDO cp -R "$REPO_DIR/scripts/." "$INSTALL_ROOT/scripts/"
$SUDO cp -R "$REPO_DIR/packages/." "$INSTALL_ROOT/packages/"
$SUDO chmod +x "$INSTALL_ROOT/bin/"*.sh "$INSTALL_ROOT/scripts/download_models.py" "$INSTALL_ROOT/scripts/install_packages.py"
$SUDO chown -R root:root "$INSTALL_ROOT" || true

# Convenience command for model downloads.
$SUDO ln -sf "$INSTALL_ROOT/scripts/download_models.py" /usr/local/bin/sparkdashboard-download-models
$SUDO ln -sf "$INSTALL_ROOT/scripts/install_packages.py" /usr/local/bin/sparkdashboard-install-packages
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
    "PERSONAPLEX_ROOT": "/opt/personaplex-bnb4",
    "HIDREAM_ROOT": "/opt/hidream-o1-image-dev-2604",
    "PIXAL3D_ROOT": "/opt/Pixal3D",
    "Z_IMAGE_ROOT": "/opt/z-image",
    "QWEN_IMAGE_ROOT": "/opt/qwen-image",
    "FLUX2_ROOT": "/opt/flux2",
    "DOMAINSHUTTLE_ROOT": "/opt/domainshuttle",
    "KREA2_ROOT": "/opt/krea-2",
    "AGENT3DIFY_ROOT": "/opt/agent3dify",
    "UN0_ROOT": "/opt/un0",
    "TRIPOSPLAT_ROOT": "/opt/triposplat",
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
# Model and optional app services are installed but not enabled by default.

if [ "$PACKAGES" != "none" ]; then
  pkg_args=("$PACKAGES")
  if [ "$SKIP_PACKAGE_DEPS" = "1" ]; then pkg_args+=(--skip-deps); fi
  if [ "$BUILD_PIXAL3D_TRELLIS" = "1" ]; then pkg_args+=(--build-pixal3d-trellis); fi
  if [ "$PACKAGE_MODELS" != "none" ]; then
    # install_packages.py downloads model weights for selected packages; if package-models is narrower, run that second pass.
    if [ "$PACKAGE_MODELS" = "$PACKAGES" ] || [ "$PACKAGE_MODELS" = "all" ]; then pkg_args+=(--download-models); fi
  fi
  "$INSTALL_ROOT/scripts/install_packages.py" "${pkg_args[@]}"
  if [ "$PACKAGE_MODELS" != "none" ] && [ "$PACKAGE_MODELS" != "$PACKAGES" ] && [ "$PACKAGE_MODELS" != "all" ]; then
    "$INSTALL_ROOT/scripts/install_packages.py" "$PACKAGE_MODELS" --skip-deps --download-models
  fi
  systemctl --user daemon-reload
fi

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
Package cmd:    sparkdashboard-install-packages all --download-models
Pixal3D full:   sparkdashboard-install-packages pixal3d --build-pixal3d-trellis

Installed user services:
  spark-dashboard.service
  qwen-nvfp4-vllm.service
  ornith-vllm.service
  mistral-medium-vllm.service
  qwen-no-think-proxy.service
  ornith-no-think-proxy.service

Optional app services available with --packages:
  personaplex.service, hidream-o1.service, pixal3d.service
  z-image.service, qwen-image.service, flux2.service
  domainshuttle-web.service, krea-2.service, agent3dify-web.service
  un0-web.service, triposplat-web.service

Start examples:
  systemctl --user start spark-dashboard.service
  systemctl --user start qwen-nvfp4-vllm.service
  systemctl --user start ornith-vllm.service
  systemctl --user start mistral-medium-vllm.service

DONE
