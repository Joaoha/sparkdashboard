from huggingface_hub import snapshot_download
from pathlib import Path
import os
import shutil

root = Path('/opt/domainshuttle')
model_dir = root / 'models' / 'Diffusion_Transformers' / 'Wan2.2-DomainShuttle-A14B'
base_dir = root / 'base' / 'Wan2.2-T2V-A14B'
cache = root / 'hf-cache'
model_dir.mkdir(parents=True, exist_ok=True)
base_dir.mkdir(parents=True, exist_ok=True)
cache.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(cache)

print('Downloading DomainShuttle weights...')
snapshot_download(
    repo_id='CNcreator0331/DomainShuttle_weight',
    local_dir=str(model_dir),
    cache_dir=str(cache),
    ignore_patterns=['.DS_Store', '.gitattributes', 'README.md'],
    max_workers=4,
)

print('Downloading selected Wan2.2 base assets...')
snapshot_download(
    repo_id='Wan-AI/Wan2.2-T2V-A14B',
    local_dir=str(base_dir),
    cache_dir=str(cache),
    allow_patterns=[
        'google/**',
        'Wan2.1_VAE.pth',
        'configuration.json',
        'models_t5_umt5-xxl-enc-bf16.pth',
    ],
    max_workers=4,
)

print('Merging required base assets into DomainShuttle model directory...')
for name in ['google', 'Wan2.1_VAE.pth', 'configuration.json', 'models_t5_umt5-xxl-enc-bf16.pth']:
    src = base_dir / name
    dst = model_dir / name
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

required = [
    model_dir / 'high_noise_model' / 'diffusion_pytorch_model.safetensors',
    model_dir / 'low_noise_model' / 'diffusion_pytorch_model.safetensors',
    model_dir / 'Wan2.1_VAE.pth',
    model_dir / 'configuration.json',
    model_dir / 'models_t5_umt5-xxl-enc-bf16.pth',
    model_dir / 'google' / 'umt5-xxl' / 'tokenizer.json',
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit('Missing required files:\n' + '\n'.join(missing))
print('DomainShuttle model directory ready:', model_dir)
