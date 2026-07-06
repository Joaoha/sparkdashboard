#!/usr/bin/env bash
set -euo pipefail
systemctl --user --no-pager --plain list-units 'qwen*' 'ornith*' 'mistral*' '*proxy*' || true
for port in 8000 8001 8002 8010 8011; do
  printf '\n== :%s /v1/models ==\n' "$port"
  curl -fsS --max-time 5 "http://127.0.0.1:${port}/v1/models" 2>/dev/null | python3 -m json.tool || true
done
