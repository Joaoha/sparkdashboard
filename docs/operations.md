# Operations

## Start/stop text models

```bash
systemctl --user start qwen-nvfp4-vllm.service
systemctl --user stop qwen-nvfp4-vllm.service

systemctl --user start ornith-vllm.service
systemctl --user start mistral-medium-vllm.service
```

## Logs

```bash
journalctl --user -u spark-dashboard.service -f
journalctl --user -u qwen-nvfp4-vllm.service -f
journalctl --user -u ornith-vllm.service -f
journalctl --user -u mistral-medium-vllm.service -f
```

## Model downloads

```bash
sparkdashboard-download-models qwen --model-dir ~/models/hf
sparkdashboard-download-models ornith --model-dir ~/models/hf
sparkdashboard-download-models mistral --model-dir ~/models/hf
```

## Health checks

```bash
curl -fsS http://127.0.0.1:7862/api/status | python3 -m json.tool
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
curl -fsS http://127.0.0.1:8010/v1/models | python3 -m json.tool
```
