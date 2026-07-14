#!/usr/bin/env bash
# Retry-safe entry point for a fresh Spark Dashboard installation.
#
# This script is intended to be run directly from the canonical curl command.
# It always clones the installer source into a temporary directory, so a failed
# installation never leaves a local `sparkdashboard/` checkout that blocks the
# next attempt with "destination path already exists".
set -euo pipefail

REPO_URL=${SPARKDASHBOARD_REPO_URL:-https://github.com/joaoha/sparkdashboard.git}
REF=${SPARKDASHBOARD_REF:-main}
WORK_PARENT=${SPARKDASHBOARD_BOOTSTRAP_TMPDIR:-${TMPDIR:-/tmp}}

if ! command -v git >/dev/null 2>&1; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "git is required and no apt-get package manager is available." >&2
    exit 1
  fi
  if [ "$(id -u)" -eq 0 ]; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y git
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y git
  else
    echo "git is required; install it first or run this command as root." >&2
    exit 1
  fi
fi

mkdir -p "$WORK_PARENT"
work_dir=$(mktemp -d "$WORK_PARENT/sparkdashboard-bootstrap.XXXXXX")
checkout="$work_dir/repo"
cleanup() { rm -rf "$work_dir"; }
trap cleanup EXIT

echo "Cloning Spark Dashboard installer ($REF) into a temporary checkout..."
git clone --depth 1 --branch "$REF" "$REPO_URL" "$checkout"
"$checkout/install.sh" "$@"