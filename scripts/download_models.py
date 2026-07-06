#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MANIFEST = json.loads((REPO_ROOT / "config/models.json").read_text())

def parse_models(selection: str) -> list[str]:
    keys = list(MANIFEST["models"].keys())
    if selection in ("all", "text"):
        return keys
    if selection in ("none", ""):
        return []
    chosen = [x.strip() for x in selection.split(",") if x.strip()]
    bad = [x for x in chosen if x not in MANIFEST["models"]]
    if bad:
        raise SystemExit(f"Unknown model(s): {', '.join(bad)}. Valid: {', '.join(keys)}")
    return chosen

def main() -> int:
    ap = argparse.ArgumentParser(description="Download Spark Dashboard model snapshots from Hugging Face")
    ap.add_argument("models", nargs="?", default="all", help="all, none, or comma list: qwen,ornith,mistral")
    ap.add_argument("--model-dir", default=os.environ.get("SPARK_MODEL_DIR", str(Path.home() / "models/hf")))
    ap.add_argument("--max-workers", type=int, default=int(os.environ.get("HF_MAX_WORKERS", "8")))
    args = ap.parse_args()
    selected = parse_models(args.models)
    if not selected:
        print("No models selected.")
        return 0
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("huggingface_hub is required. Install with: python3 -m pip install huggingface_hub hf_transfer", file=sys.stderr)
        raise
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    parent = Path(args.model_dir).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    for key in selected:
        meta = MANIFEST["models"][key]
        local_dir = parent / meta["local_name"]
        print(f"== Downloading {key}: {meta['repo_id']} -> {local_dir}", flush=True)
        path = snapshot_download(
            repo_id=meta["repo_id"],
            local_dir=str(local_dir),
            max_workers=args.max_workers,
        )
        print(path, flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
