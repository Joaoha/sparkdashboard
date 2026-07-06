from __future__ import annotations

import gc
import json
import threading
import time
import uuid
import shutil
from datetime import datetime
from pathlib import Path
import os
from typing import Any

import torch
from diffusers import Krea2Pipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get('KREA2_ROOT', '/opt/krea-2'))
MODEL_DIR = ROOT / 'models' / 'Krea-2-Turbo'
OUTPUT_DIR = ROOT / 'outputs'
MANIFEST = OUTPUT_DIR / 'manifest.json'
APP_VERSION = 'krea2-web-2026-06-29-save-delete'
PUBLIC_BASE = os.environ.get('KREA2_PUBLIC_BASE', 'http://localhost:7868')
MODEL_ID = 'krea/Krea-2-Turbo'

REQUIRED_MODEL_FILES = [
    'model_index.json',
    'text_encoder/model.safetensors',
    'transformer/diffusion_pytorch_model.safetensors.index.json',
    'transformer/diffusion_pytorch_model-00001-of-00003.safetensors',
    'transformer/diffusion_pytorch_model-00002-of-00003.safetensors',
    'transformer/diffusion_pytorch_model-00003-of-00003.safetensors',
    'vae/diffusion_pytorch_model.safetensors',
]

def model_downloaded() -> bool:
    return all((MODEL_DIR / rel).exists() for rel in REQUIRED_MODEL_FILES)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='Krea-2 Turbo WebUI', version=APP_VERSION)
app.mount('/outputs', StaticFiles(directory=str(OUTPUT_DIR)), name='outputs')

_pipe = None
_pipe_lock = threading.Lock()
_gen_lock = threading.Lock()
_last_error: str | None = None


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def load_manifest() -> list[dict[str, Any]]:
    if not MANIFEST.exists():
        return []
    try:
        data = json.loads(MANIFEST.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_manifest(items: list[dict[str, Any]]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(items, indent=2, sort_keys=True))
    tmp.replace(MANIFEST)


def gpu_apps() -> list[dict[str, Any]]:
    import subprocess
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
            text=True, capture_output=True, timeout=10,
        )
        apps = []
        for line in (proc.stdout or '').splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                apps.append({'pid': parts[0], 'process_name': parts[1], 'used_memory_mib': int(float(parts[2]))})
        return apps
    except Exception:
        return []


def pipeline_loaded() -> bool:
    return _pipe is not None


def get_pipe():
    global _pipe, _last_error
    with _pipe_lock:
        if _pipe is None:
            if not model_downloaded():
                missing = [rel for rel in REQUIRED_MODEL_FILES if not (MODEL_DIR / rel).exists()]
                raise RuntimeError(f'Model not fully downloaded at {MODEL_DIR}; missing: {missing}')
            started = time.time()
            pipe = Krea2Pipeline.from_pretrained(
                str(MODEL_DIR),
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            pipe = pipe.to('cuda')
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
            _pipe = pipe
            _last_error = None
            print(f'[krea2] loaded pipeline in {time.time() - started:.1f}s', flush=True)
        return _pipe


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str = ''
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    steps: int = Field(8, ge=1, le=80)
    guidance_scale: float = Field(0.0, ge=0.0, le=20.0)
    seed: int = Field(0, ge=0, le=2_147_483_647)
    num_images: int = Field(1, ge=1, le=4)


@app.get('/', response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get('/health')
def health() -> dict[str, Any]:
    return {'ok': True, 'service': 'krea-2', 'version': APP_VERSION, 'timestamp': time.time()}


@app.get('/api/status')
def status() -> dict[str, Any]:
    return {
        'ok': True,
        'version': APP_VERSION,
        'model_id': MODEL_ID,
        'model_dir': str(MODEL_DIR),
        'model_downloaded': model_downloaded(),
        'model_loaded': pipeline_loaded(),
        'last_error': _last_error,
        'images': load_manifest(),
        'gpu_apps': gpu_apps(),
    }


@app.post('/api/unload')
def unload() -> dict[str, Any]:
    global _pipe
    with _pipe_lock:
        loaded_before = _pipe is not None
        if _pipe is not None:
            del _pipe
            _pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    return {'ok': True, 'loaded_before': loaded_before, 'model_loaded': False}


@app.get('/api/images')
def images() -> dict[str, Any]:
    return {'ok': True, 'images': load_manifest()}


def _image_dir_for_id(image_id: str) -> Path:
    if not image_id or chr(47) in image_id or chr(92) in image_id or chr(46)+chr(46) in image_id:
        raise HTTPException(status_code=400, detail='invalid image id')
    return OUTPUT_DIR / image_id


def _find_image_record(image_id: str) -> dict[str, Any]:
    for item in load_manifest():
        if item.get('id') == image_id:
            return item
    raise HTTPException(status_code=404, detail='image not found')


@app.get('/api/images/{image_id}/download')
def download_image(image_id: str) -> FileResponse:
    item = _find_image_record(image_id)
    primary = item.get('primary_url')
    if not primary or not str(primary).startswith('/outputs/'):
        raise HTTPException(status_code=404, detail='primary image not found')
    rel = str(primary).removeprefix('/outputs/')
    path = OUTPUT_DIR / rel
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail='image file missing')
    safe_model = MODEL_ID.split('/')[-1].replace('/', '-')
    filename = f'{safe_model}_{image_id}{path.suffix or ".png"}'
    return FileResponse(path, media_type='image/png', filename=filename)


@app.delete('/api/images/{image_id}')
def delete_image(image_id: str) -> dict[str, Any]:
    _find_image_record(image_id)
    before = load_manifest()
    after = [item for item in before if item.get('id') != image_id]
    save_manifest(after)
    image_dir = _image_dir_for_id(image_id)
    removed_dir = False
    if image_dir.exists():
        shutil.rmtree(image_dir)
        removed_dir = True
    return {'ok': True, 'id': image_id, 'removed_manifest': len(before) - len(after), 'removed_dir': removed_dir, 'images': after}


@app.post('/api/generate')
def generate(req: GenerateRequest) -> JSONResponse:
    global _last_error
    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail='generation already running')
    try:
        # Align to multiples of 16; Krea pipeline also pads internally, but keeping metadata exact is nicer.
        width = ((req.width + 15) // 16) * 16
        height = ((req.height + 15) // 16) * 16
        pipe = get_pipe()
        generator = torch.Generator(device='cuda').manual_seed(req.seed)
        started = time.time()
        result = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            num_images_per_prompt=req.num_images,
            generator=generator,
        )
        duration = time.time() - started
        image_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        out_dir = OUTPUT_DIR / image_id
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for i, image in enumerate(result.images):
            filename = f'image_{i:02d}.png' if len(result.images) > 1 else 'image.png'
            path = out_dir / filename
            image.save(path)
            records.append('/outputs/' + str(path.relative_to(OUTPUT_DIR)))
        meta = {
            'id': image_id,
            'created_at': now_iso(),
            'model': MODEL_ID,
            'prompt': req.prompt,
            'negative_prompt': req.negative_prompt,
            'width': width,
            'height': height,
            'steps': req.steps,
            'guidance_scale': req.guidance_scale,
            'seed': req.seed,
            'num_images': req.num_images,
            'duration_sec': round(duration, 3),
            'images': records,
            'primary_url': records[0] if records else None,
        }
        (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, sort_keys=True))
        items = load_manifest()
        items.insert(0, meta)
        save_manifest(items[:200])
        return JSONResponse({'ok': True, **meta})
    except Exception as exc:
        _last_error = repr(exc)
        raise
    finally:
        _gen_lock.release()


HTML = r'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Krea-2 Turbo · Spark</title>
  <style>
    :root{color-scheme:dark;--bg:#0a0807;--panel:#11100e;--line:#29251f;--text:#f4efe7;--muted:#aaa094;--green:#5FE3A0;--warn:#ffc857;--bad:#ff6b6b;}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 18% -5%,#25351f 0,#0a0807 34%,#060505 100%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{padding:28px 32px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:flex-end;gap:18px}h1{font-size:34px;margin:0;letter-spacing:-.04em}.sub{color:var(--muted);font-size:13px}.pill{font:12px ui-monospace,monospace;color:var(--green);border:1px solid #315d45;background:#0d1711;border-radius:999px;padding:8px 12px}main{max-width:1500px;margin:0 auto;padding:22px;display:grid;grid-template-columns:minmax(360px,520px) 1fr;gap:18px}.card{border:1px solid var(--line);background:rgba(17,16,14,.92);border-radius:18px;padding:18px;box-shadow:0 18px 60px #0008}.card h2{margin:0 0 14px;font-size:18px}label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.field{display:flex;flex-direction:column;gap:6px;margin-bottom:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}input,textarea{width:100%;border:1px solid var(--line);background:#090807;color:var(--text);border-radius:10px;padding:10px;font:inherit}textarea{min-height:140px;resize:vertical}.hint{font-size:12px;color:var(--muted);line-height:1.45}.btn{border:1px solid #416b55;background:linear-gradient(180deg,#1c5f3f,#113622);color:#eafff3;border-radius:12px;padding:12px 16px;font-weight:700;cursor:pointer}.btn.secondary{background:#10100f;color:var(--text);border-color:var(--line)}.btn.danger{background:#3b1717;border-color:#773232}.btn:disabled{opacity:.55;cursor:not-allowed}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.status{font-family:ui-monospace,monospace;font-size:13px;border:1px solid var(--line);border-radius:12px;padding:10px;background:#090807;white-space:pre-wrap}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}.item{border:1px solid var(--line);border-radius:14px;padding:10px;background:#0c0b0a}.item img{width:100%;border-radius:10px;background:#000}.item-actions{margin-top:10px}.item-actions .btn,.item-actions a.btn{display:inline-flex;align-items:center;justify-content:center;text-decoration:none;padding:8px 10px;font-size:12px}.small{font-size:12px;color:var(--muted)}a{color:var(--green)}@media(max-width:1000px){main{grid-template-columns:1fr}header{display:block}}
  </style>
</head>
<body>
<header><div><h1>Krea-2 Turbo</h1><div class="sub">Aesthetic text-to-image generation on Spark · Turbo defaults: 8 steps, CFG 0</div></div><div class="pill" id="health">checking…</div></header>
<main>
<section class="card">
  <h2>Generate image</h2>
  <form id="form">
    <div class="field"><label>Prompt</label><textarea name="prompt">A cinematic editorial photo of a chrome espresso machine in a sunlit brutalist kitchen, soft shadows, tasteful design magazine aesthetic, high detail.</textarea></div>
    <div class="field"><label>Negative prompt</label><input name="negative_prompt" value="" /></div>
    <div class="grid">
      <div class="field"><label>Width</label><input name="width" type="number" min="256" max="2048" step="16" value="1024"></div>
      <div class="field"><label>Height</label><input name="height" type="number" min="256" max="2048" step="16" value="1024"></div>
      <div class="field"><label>Steps</label><input name="steps" type="number" min="1" max="80" value="8"></div>
      <div class="field"><label>CFG</label><input name="guidance_scale" type="number" min="0" max="20" step="0.1" value="0.0"></div>
      <div class="field"><label>Seed</label><input name="seed" type="number" min="0" value="0"></div>
      <div class="field"><label>Images</label><input name="num_images" type="number" min="1" max="4" value="1"></div>
    </div>
    <p class="hint">First generation loads ~60GB of model assets and will be slower. Use UNLOAD MODEL from here or dashboard to free memory while keeping UI online.</p>
    <div class="row"><button class="btn" id="submit">Generate</button><button class="btn secondary" type="button" onclick="refresh()">Refresh</button><button class="btn danger" type="button" onclick="unload()">Unload model</button></div>
  </form>
</section>
<section class="card"><h2>Status</h2><div id="status" class="status">Loading…</div></section>
<section class="card" style="grid-column:1/-1"><h2>Gallery</h2><div id="gallery" class="gallery"></div></section>
</main>
<script>
async function api(path, opts){const r=await fetch(path, opts); if(!r.ok){throw new Error(await r.text())} return await r.json()}
function render(st){
  document.getElementById('health').textContent = st.model_loaded ? 'MODEL LOADED' : (st.model_downloaded ? 'READY' : 'MODEL MISSING');
  document.getElementById('status').textContent = JSON.stringify({model_downloaded:st.model_downloaded, model_loaded:st.model_loaded, images:(st.images||[]).length, last_error:st.last_error, gpu_apps:st.gpu_apps}, null, 2);
  document.getElementById('gallery').innerHTML = (st.images||[]).map(x=>`<div class="item"><a href="${x.primary_url}" target="_blank"><img src="${x.primary_url}"></a><div class="small"><b>${x.id}</b><br>${x.width}×${x.height} · ${x.steps} steps · CFG ${x.guidance_scale} · ${x.duration_sec}s<br>${(x.prompt||'').slice(0,180)}</div><div class="row item-actions"><a class="btn secondary" href="/api/images/${encodeURIComponent(x.id)}/download" download>SAVE</a><button class="btn danger" type="button" onclick="deleteImage('${x.id}')">DELETE</button></div></div>`).join('') || '<div class="small">No images yet.</div>';
}
async function refresh(){try{render(await api('/api/status'))}catch(e){document.getElementById('status').textContent=String(e)}}
async function unload(){if(!confirm('Unload Krea-2 model from memory?')) return; await api('/api/unload',{method:'POST'}); await refresh()}
async function deleteImage(id){if(!confirm('Delete image '+id+' from Krea-2 outputs?')) return; await api('/api/images/'+encodeURIComponent(id),{method:'DELETE'}); await refresh()}
document.getElementById('form').addEventListener('submit', async ev=>{ev.preventDefault(); const b=document.getElementById('submit'); b.disabled=true; b.textContent='Generating…'; try{const fd=new FormData(ev.target); const obj=Object.fromEntries(fd.entries()); ['width','height','steps','seed','num_images'].forEach(k=>obj[k]=Number(obj[k])); obj.guidance_scale=Number(obj.guidance_scale); const r=await api('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)}); await refresh(); window.scrollTo({top:document.body.scrollHeight, behavior:'smooth'});}catch(e){alert(String(e)); await refresh()} finally{b.disabled=false; b.textContent='Generate'}});
setInterval(refresh, 5000); refresh();
</script>
</body>
</html>
'''
