#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path('/opt/un0')
REPO = ROOT / 'repo'
PYTHON = ROOT / '.venv/bin/python'
OUTPUTS = ROOT / 'outputs'
JOBS = ROOT / 'jobs'
TRAINING = ROOT / 'training'
LOGS = ROOT / 'logs'
for d in (OUTPUTS, JOBS, TRAINING, LOGS):
    d.mkdir(parents=True, exist_ok=True)

PUBLIC_HOST = os.environ.get('UN0_PUBLIC_HOST', 'localhost')
PORT = int(os.environ.get('UN0_PORT', '7870'))

app = FastAPI(title='Un-0 Spark UI', version='1.0')
app.mount('/outputs', StaticFiles(directory=str(OUTPUTS)), name='outputs')
app.mount('/jobs', StaticFiles(directory=str(JOBS)), name='jobs')
app.mount('/training', StaticFiles(directory=str(TRAINING)), name='training')

_lock = threading.Lock()
_processes: dict[str, subprocess.Popen] = {}

PRETRAINED = [
    'cifar10/n1024', 'cifar10/n2048', 'cifar10/n4096',
    'imagenet64/n6656', 'imagenet64/n10240', 'imagenet64/n16384',
]

CIFAR_LABELS = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck',
}

class InferRequest(BaseModel):
    pretrained: str = Field(default='cifar10/n1024')
    classes: list[int] = Field(default_factory=lambda: list(range(10)))
    samples_per_class: int = Field(default=2, ge=1, le=32)
    seed: int = Field(default=42)

class CifarTrainRequest(BaseModel):
    epochs: int = Field(default=1, ge=1, le=10000)
    batch_size: int = Field(default=512, ge=1, le=16384)
    n_oscillators: int = Field(default=1024, ge=16, le=32768)
    n_conditional_oscillators: int = Field(default=64, ge=0, le=4096)
    num_steps: int = Field(default=4, ge=0, le=128)
    precision: Literal['fp32','tf32','bf16','fp16'] = 'tf32'
    solver: Literal['euler','rk4'] = 'euler'
    lr: float = Field(default=1e-3, gt=0)
    seed: int = Field(default=42)
    queue_size: int = Field(default=256, ge=0, le=100000)
    num_workers: int = Field(default=4, ge=0, le=64)
    fid_every_epochs: int = Field(default=0, ge=0, le=10000)
    fid_num_samples: int = Field(default=1000, ge=10, le=50000)
    fid_batch_size: int = Field(default=256, ge=1, le=4096)
    resume: str | None = None
    wandb_project: str | None = None
    dry_run: bool = False

class ImagenetTrainRequest(BaseModel):
    data_root: str = ''
    val_root: str = ''
    epochs: int = Field(default=1, ge=1, le=10000)
    batch_size: int = Field(default=256, ge=1, le=8192)
    precision: Literal['fp32','tf32','bf16','fp16'] = 'bf16'
    lr: float = Field(default=1e-3, gt=0)
    seed: int = Field(default=42)
    num_workers: int = Field(default=8, ge=0, le=64)
    queue_size: int = Field(default=4096, ge=0, le=1000000)
    num_pos: int = Field(default=8, ge=1, le=1024)
    fid_every_epochs: int = Field(default=0, ge=0, le=10000)
    resume: str | None = None
    wandb_project: str | None = None
    dry_run: bool = True


def _now() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _job_id(prefix: str) -> str:
    return f'{prefix}_{_now()}_{secrets.token_hex(4)}'


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def _tail(path: Path, n: int = 120) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(errors='replace').splitlines()
    return lines[-n:]


def _run_job(job_id: str, cmd: list[str], cwd: Path, env: dict[str,str] | None = None) -> None:
    meta_path = JOBS / job_id / 'meta.json'
    log_path = JOBS / job_id / 'run.log'
    meta = _read_json(meta_path, {})
    meta.update({'status': 'running', 'started_at': time.time(), 'command': cmd, 'cwd': str(cwd), 'log': str(log_path)})
    _write_json(meta_path, meta)
    with log_path.open('ab', buffering=0) as log:
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
        proc_env.setdefault('PYTHONUNBUFFERED', '1')
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, env=proc_env, preexec_fn=os.setsid)
        with _lock:
            _processes[job_id] = proc
        rc = proc.wait()
    with _lock:
        _processes.pop(job_id, None)
    meta = _read_json(meta_path, {})
    meta.update({'status': 'completed' if rc == 0 else 'failed', 'exit_code': rc, 'finished_at': time.time()})
    # Attach obvious output files.
    out_files = []
    job_dir = JOBS / job_id
    for p in sorted(job_dir.rglob('*')):
        if p.is_file() and p.name != 'meta.json':
            out_files.append(str(p))
    meta['files'] = out_files
    _write_json(meta_path, meta)


def _jobs() -> list[dict[str, Any]]:
    rows = []
    for meta_path in sorted(JOBS.glob('*/meta.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = _read_json(meta_path, {})
        job_id = meta_path.parent.name
        with _lock:
            proc = _processes.get(job_id)
        if proc and proc.poll() is None:
            meta['status'] = 'running'
        meta['job_id'] = job_id
        meta['log_tail'] = _tail(meta_path.parent / 'run.log', 60)
        rows.append(meta)
    return rows


def _infer_cmd(req: InferRequest, output: Path) -> list[str]:
    classes = [str(int(c)) for c in req.classes]
    return [
        str(PYTHON), 'un0/inference.py',
        '--pretrained', req.pretrained,
        '--classes', *classes,
        '--samples-per-class', str(req.samples_per_class),
        '--seed', str(req.seed),
        '--output', str(output),
    ]


def _cifar_train_cmd(req: CifarTrainRequest, checkpoint_dir: Path) -> list[str]:
    cmd = [
        str(PYTHON), 'un0/train_cifar10.py',
        '--epochs', str(req.epochs),
        '--batch-size', str(req.batch_size),
        '--n-oscillators', str(req.n_oscillators),
        '--n-conditional-oscillators', str(req.n_conditional_oscillators),
        '--num-steps', str(req.num_steps),
        '--precision', req.precision,
        '--solver', req.solver,
        '--lr', str(req.lr),
        '--seed', str(req.seed),
        '--queue-size', str(req.queue_size),
        '--num-workers', str(req.num_workers),
        '--fid-every-epochs', str(req.fid_every_epochs),
        '--fid-num-samples', str(req.fid_num_samples),
        '--fid-batch-size', str(req.fid_batch_size),
        '--checkpoint-dir', str(checkpoint_dir),
    ]
    if req.resume:
        cmd += ['--resume', req.resume]
    if req.wandb_project:
        cmd += ['--wandb-project', req.wandb_project]
    return cmd


def _imagenet_train_cmd(req: ImagenetTrainRequest, checkpoint_dir: Path) -> list[str]:
    if not req.data_root:
        raise HTTPException(400, 'ImageNet training requires data_root (<data-root>/<class:05d>/*.png).')
    cmd = [
        str(PYTHON), 'un0/train_imagenet.py',
        '--data-root', req.data_root,
        '--epochs', str(req.epochs),
        '--batch-size', str(req.batch_size),
        '--precision', req.precision,
        '--lr', str(req.lr),
        '--seed', str(req.seed),
        '--num-workers', str(req.num_workers),
        '--queue-size', str(req.queue_size),
        '--num-pos', str(req.num_pos),
        '--fid-every-epochs', str(req.fid_every_epochs),
        '--checkpoint-dir', str(checkpoint_dir),
    ]
    if req.val_root:
        cmd += ['--val-root', req.val_root]
    if req.resume:
        cmd += ['--resume', req.resume]
    if req.wandb_project:
        cmd += ['--wandb-project', req.wandb_project]
    return cmd

@app.get('/health')
def health() -> dict[str, Any]:
    return {'ok': True, 'service': 'un0-web', 'repo': str(REPO), 'python': str(PYTHON), 'timestamp': time.time()}

@app.get('/api/status')
def status() -> dict[str, Any]:
    return {
        'ok': True,
        'repo': str(REPO),
        'venv': str(ROOT / '.venv'),
        'outputs': str(OUTPUTS),
        'pretrained': PRETRAINED,
        'cifar_labels': CIFAR_LABELS,
        'jobs': _jobs()[:20],
    }

@app.post('/api/infer')
def infer(req: InferRequest, bg: BackgroundTasks) -> dict[str, Any]:
    if req.pretrained not in PRETRAINED:
        raise HTTPException(400, f'Unknown pretrained checkpoint: {req.pretrained}')
    if not req.classes:
        raise HTTPException(400, 'At least one class id is required')
    family = req.pretrained.split('/', 1)[0]
    max_class = 9 if family == 'cifar10' else 999
    bad = [c for c in req.classes if c < 0 or c > max_class]
    if bad:
        raise HTTPException(400, f'Invalid class ids for {family}: {bad}')
    job_id = _job_id('infer')
    job_dir = JOBS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output = job_dir / 'grid.png'
    cmd = _infer_cmd(req, output)
    meta = {'job_id': job_id, 'type': 'inference', 'status': 'queued', 'request': req.model_dump(), 'output': str(output), 'output_url': f'/jobs/{job_id}/grid.png', 'created_at': time.time(), 'command': cmd}
    _write_json(job_dir / 'meta.json', meta)
    bg.add_task(_run_job, job_id, cmd, REPO)
    return meta

@app.post('/api/train/cifar10')
def train_cifar10(req: CifarTrainRequest, bg: BackgroundTasks) -> dict[str, Any]:
    job_id = _job_id('train_cifar10')
    job_dir = JOBS / job_id
    checkpoint_dir = TRAINING / job_id
    cmd = _cifar_train_cmd(req, checkpoint_dir)
    meta = {'job_id': job_id, 'type': 'train_cifar10', 'status': 'dry_run' if req.dry_run else 'queued', 'request': req.model_dump(), 'checkpoint_dir': str(checkpoint_dir), 'created_at': time.time(), 'command': cmd}
    _write_json(job_dir / 'meta.json', meta)
    if not req.dry_run:
        bg.add_task(_run_job, job_id, cmd, REPO)
    return meta

@app.post('/api/train/imagenet64')
def train_imagenet(req: ImagenetTrainRequest, bg: BackgroundTasks) -> dict[str, Any]:
    job_id = _job_id('train_imagenet64')
    job_dir = JOBS / job_id
    checkpoint_dir = TRAINING / job_id
    cmd = _imagenet_train_cmd(req, checkpoint_dir)
    meta = {'job_id': job_id, 'type': 'train_imagenet64', 'status': 'dry_run' if req.dry_run else 'queued', 'request': req.model_dump(), 'checkpoint_dir': str(checkpoint_dir), 'created_at': time.time(), 'command': cmd}
    _write_json(job_dir / 'meta.json', meta)
    if not req.dry_run:
        bg.add_task(_run_job, job_id, cmd, REPO)
    return meta

@app.get('/api/jobs')
def jobs() -> dict[str, Any]:
    return {'jobs': _jobs()}

@app.get('/api/jobs/{job_id}')
def job(job_id: str) -> dict[str, Any]:
    meta_path = JOBS / job_id / 'meta.json'
    if not meta_path.exists():
        raise HTTPException(404, 'job not found')
    meta = _read_json(meta_path, {})
    meta['job_id'] = job_id
    meta['log_tail'] = _tail(JOBS / job_id / 'run.log', 200)
    return meta

@app.post('/api/jobs/{job_id}/stop')
def stop_job(job_id: str) -> dict[str, Any]:
    with _lock:
        proc = _processes.get(job_id)
    if not proc or proc.poll() is not None:
        raise HTTPException(404, 'running job not found')
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    return {'ok': True, 'job_id': job_id, 'signal': 'SIGTERM'}

@app.get('/api/jobs/{job_id}/log')
def job_log(job_id: str) -> PlainTextResponse:
    path = JOBS / job_id / 'run.log'
    if not path.exists():
        raise HTTPException(404, 'log not found')
    return PlainTextResponse(path.read_text(errors='replace'))

@app.get('/api/jobs/{job_id}/image')
def job_image(job_id: str) -> FileResponse:
    path = JOBS / job_id / 'grid.png'
    if not path.exists():
        raise HTTPException(404, 'image not found')
    return FileResponse(path)

HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Un-0 Spark UI</title>
<style>
:root{color-scheme:dark;--bg:#0a0807;--panel:#13110f;--muted:#9b958b;--line:#2c2924;--accent:#5FE3A0;--warn:#ffcc66;--bad:#ff6b6b}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#183526 0,#0a0807 38%),var(--bg);color:#f3eee7;font-family:Inter,ui-sans-serif,system-ui,sans-serif}main{max-width:1320px;margin:0 auto;padding:28px}h1{font-size:44px;letter-spacing:-.04em;margin:0}.sub{color:var(--muted);font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px;margin-top:18px}.card{background:rgba(19,17,15,.86);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 14px 40px rgba(0,0,0,.22)}label{display:block;font-size:12px;text-transform:uppercase;color:var(--muted);margin:12px 0 5px}input,select,textarea,button{width:100%;border-radius:11px;border:1px solid var(--line);background:#0d0c0b;color:#f3eee7;padding:10px;font:inherit}textarea{min-height:72px}button{background:linear-gradient(135deg,#244936,#14251d);border-color:#3b7c59;color:#d9ffe9;font-weight:700;cursor:pointer;margin-top:14px}button.secondary{background:#151310;color:#ddd;border-color:#353029}.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.pill{display:inline-flex;border:1px solid #375b48;border-radius:999px;padding:5px 9px;margin:3px;color:#c8f8dd;background:#102018}.job{border-top:1px solid var(--line);padding:12px 0}.status-running{color:var(--warn)}.status-completed{color:var(--accent)}.status-failed{color:var(--bad)}pre{white-space:pre-wrap;max-height:260px;overflow:auto;background:#080706;border:1px solid var(--line);border-radius:12px;padding:12px;color:#d4cec5}.gallery img{max-width:100%;image-rendering:auto;border-radius:12px;border:1px solid var(--line);background:#050505}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-top:20px}.tabs button{width:auto;margin:0}.hidden{display:none}a{color:var(--accent)}</style></head>
<body><main>
<header><div class="sub">KURAMOTO OSCILLATOR IMAGE GENERATION · SPARK GB10</div><h1>Un-0</h1><p class="sub">Inference for released CIFAR-10/ImageNet-64 checkpoints plus background training workflows. Outputs and checkpoints stay under /opt/un0.</p></header>
<section class="grid">
  <div class="card"><h2>Inference</h2><div class="sub">Generate low-resolution sample grids from released checkpoints.</div>
    <label>Checkpoint</label><select id="infer-pretrained"><option>cifar10/n1024</option><option>cifar10/n2048</option><option>cifar10/n4096</option><option>imagenet64/n6656</option><option>imagenet64/n10240</option><option>imagenet64/n16384</option></select>
    <label>Classes</label><input id="infer-classes" value="0 1 2 3 4 5 6 7 8 9"><div class="sub">CIFAR: 0-9. ImageNet-64: 0-999.</div>
    <div class="row"><div><label>Samples/class</label><input id="infer-spc" type="number" value="2" min="1" max="32"></div><div><label>Seed</label><input id="infer-seed" type="number" value="42"></div></div>
    <button onclick="startInference()">RUN INFERENCE</button>
  </div>
  <div class="card"><h2>CIFAR-10 training</h2><div class="sub">Self-contained training path; downloads uoft-cs/cifar10 to HF cache on first real run.</div>
    <div class="row"><div><label>Epochs</label><input id="c-epochs" type="number" value="1" min="1"></div><div><label>Batch size</label><input id="c-batch" type="number" value="512"></div></div>
    <div class="row"><div><label>Oscillators</label><input id="c-osc" type="number" value="1024"></div><div><label>Cond oscillators</label><input id="c-cond" type="number" value="64"></div></div>
    <div class="row"><div><label>Steps</label><input id="c-steps" type="number" value="4"></div><div><label>Precision</label><select id="c-prec"><option>tf32</option><option>bf16</option><option>fp16</option><option>fp32</option></select></div></div>
    <div class="row"><div><label>Queue size</label><input id="c-queue" type="number" value="256"></div><div><label>Workers</label><input id="c-workers" type="number" value="4"></div></div>
    <label>Resume checkpoint (optional)</label><input id="c-resume" placeholder="/opt/un0/training/.../latest.pt">
    <div class="row"><button onclick="startCifar(true)" class="secondary">DRY RUN / BUILD COMMAND</button><button onclick="startCifar(false)">START CIFAR TRAINING</button></div>
  </div>
  <div class="card"><h2>ImageNet-64 training</h2><div class="sub">Requires preprocessed ImageFolder PNG tree: &lt;data-root&gt;/&lt;class:05d&gt;/*.png. Use dry-run until data paths are ready.</div>
    <label>Data root</label><input id="i-data" placeholder="/data/imagenet64/train">
    <label>Val root (optional; required for FID)</label><input id="i-val" placeholder="/data/imagenet64/val">
    <div class="row"><div><label>Epochs</label><input id="i-epochs" type="number" value="1"></div><div><label>Batch size</label><input id="i-batch" type="number" value="256"></div></div>
    <div class="row"><div><label>Precision</label><select id="i-prec"><option>bf16</option><option>tf32</option><option>fp16</option><option>fp32</option></select></div><div><label>Workers</label><input id="i-workers" type="number" value="8"></div></div>
    <div class="row"><button onclick="startImagenet(true)" class="secondary">DRY RUN / BUILD COMMAND</button><button onclick="startImagenet(false)">START IMAGENET TRAINING</button></div>
  </div>
</section>
<section class="grid"><div class="card"><h2>Latest jobs</h2><div id="jobs"></div></div><div class="card"><h2>Preview</h2><div class="gallery" id="preview"><div class="sub">Run an inference job to preview its grid.</div></div></div></section>
</main><script>
function ints(s){return s.trim().split(/[ ,]+/).filter(Boolean).map(x=>parseInt(x,10));}
async function post(url, body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const j=await r.json(); if(!r.ok) throw new Error(j.detail||JSON.stringify(j)); return j;}
async function startInference(){try{await post('/api/infer',{pretrained:val('infer-pretrained'),classes:ints(val('infer-classes')),samples_per_class:num('infer-spc'),seed:num('infer-seed')}); refresh();}catch(e){alert(e.message)}}
async function startCifar(dry){try{let body={epochs:num('c-epochs'),batch_size:num('c-batch'),n_oscillators:num('c-osc'),n_conditional_oscillators:num('c-cond'),num_steps:num('c-steps'),precision:val('c-prec'),queue_size:num('c-queue'),num_workers:num('c-workers'),dry_run:dry}; if(val('c-resume')) body.resume=val('c-resume'); const j=await post('/api/train/cifar10',body); if(dry) alert('Command built:\n'+j.command.join(' ')); refresh();}catch(e){alert(e.message)}}
async function startImagenet(dry){try{let body={data_root:val('i-data'),val_root:val('i-val'),epochs:num('i-epochs'),batch_size:num('i-batch'),precision:val('i-prec'),num_workers:num('i-workers'),dry_run:dry}; const j=await post('/api/train/imagenet64',body); if(dry) alert('Command built:\n'+j.command.join(' ')); refresh();}catch(e){alert(e.message)}}
async function stopJob(id){if(!confirm('Stop '+id+'?'))return; await fetch('/api/jobs/'+id+'/stop',{method:'POST'}); refresh();}
function val(id){return document.getElementById(id).value} function num(id){return Number(val(id))}
function cls(s){return 'status-'+String(s||'').replace(/[^a-z]/g,'')}
async function refresh(){const j=await (await fetch('/api/status',{cache:'no-store'})).json(); const root=document.getElementById('jobs'); root.innerHTML=(j.jobs||[]).map(job=>`<div class="job"><b>${job.job_id}</b> <span class="${cls(job.status)}">${job.status}</span><div class="sub">${job.type||''} · ${job.checkpoint_dir||job.output||''}</div>${job.output_url?`<a href="${job.output_url}" target="_blank">open image</a>`:''} ${job.status==='running'?`<button class="secondary" onclick="stopJob('${job.job_id}')">STOP</button>`:''}<pre>${(job.log_tail||[]).slice(-24).join('\n')||'No log yet.'}</pre></div>`).join('')||'<div class="sub">No jobs yet.</div>'; const first=(j.jobs||[]).find(x=>x.output_url); const p=document.getElementById('preview'); if(first) p.innerHTML=`<a href="${first.output_url}" target="_blank"><img src="${first.output_url}?t=${Date.now()}"></a><div class="sub">${first.job_id}</div>`;}
refresh(); setInterval(refresh,3000);
</script></body></html>'''

@app.get('/', response_class=HTMLResponse)
def index() -> str:
    return HTML

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=PORT)
