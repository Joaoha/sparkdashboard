#!/usr/bin/env bash
set -euo pipefail
BASE=${1:-http://127.0.0.1:7862}
curl -fsSI "$BASE/" >/dev/null
curl -fsS "$BASE/api/status" | python3 -m json.tool >/dev/null
echo "Dashboard smoke OK: $BASE"
