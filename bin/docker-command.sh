#!/usr/bin/env bash
# Execute Docker directly when the user has socket access, otherwise use a
# noninteractive sudo fallback. This makes freshly installed user services work
# before a docker-group membership change takes effect at the next login.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: $(basename "$0") <docker arguments...>" >&2
  exit 2
fi

if docker info >/dev/null 2>&1; then
  exec docker "$@"
fi

if command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
  echo "Docker socket is unavailable to $(id -un); using passwordless sudo Docker fallback." >&2
  exec sudo -n docker "$@"
fi

echo "Docker is unavailable to $(id -un). Add the user to the docker group and log in again, or configure passwordless sudo for docker." >&2
exit 1
