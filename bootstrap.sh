#!/usr/bin/env bash
# Retry-safe entry point for a fresh Spark Dashboard installation.
#
# This script is intended to be run directly from the canonical curl command.
# It always clones the installer source into a temporary directory, so a failed
# installation never leaves a local `sparkdashboard/` checkout that blocks the
# next attempt with "destination path already exists".
set -euo pipefail

REPO_URL=${SPARKDASHBOARD_REPO_URL:-https://github.com/joaoha/sparkdashboard.git}
REF=${SPARKDASHBOARD_REF:-}
WORK_PARENT=${SPARKDASHBOARD_BOOTSTRAP_TMPDIR:-${TMPDIR:-/tmp}}

if ! [[ "$REF" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "SPARKDASHBOARD_REF must be an immutable 40-character commit SHA." >&2
  exit 2
fi

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

echo "Fetching Spark Dashboard installer revision $REF into a temporary checkout..."
git init --quiet "$checkout"
git -C "$checkout" remote add origin "$REPO_URL"
git -C "$checkout" fetch --depth 1 origin "$REF"
git -C "$checkout" checkout --quiet --detach FETCH_HEAD
"$checkout/install.sh" "$@"