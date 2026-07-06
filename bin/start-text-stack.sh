#!/usr/bin/env bash
set -euo pipefail
# Start the default fast Qwen text stack and both lightweight no-think proxies.
systemctl --user start qwen-no-think-proxy.service ornith-no-think-proxy.service
systemctl --user start qwen-nvfp4-vllm.service
echo "Qwen text stack starting. Watch with: journalctl --user -u qwen-nvfp4-vllm.service -f"
