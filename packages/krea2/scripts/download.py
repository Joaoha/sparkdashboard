from huggingface_hub import snapshot_download
from pathlib import Path
root = Path('/opt/krea-2')
cache = root / 'hf-cache'
local_dir = root / 'models' / 'Krea-2-Turbo'
local_dir.mkdir(parents=True, exist_ok=True)
path = snapshot_download(
    repo_id='krea/Krea-2-Turbo',
    cache_dir=str(cache),
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(path)
for required in ['model_index.json', 'transformer/diffusion_pytorch_model.safetensors.index.json', 'text_encoder/model.safetensors', 'vae/diffusion_pytorch_model.safetensors']:
    p = local_dir / required
    print(required, p.exists(), p.stat().st_size if p.exists() else None)
