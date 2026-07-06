from __future__ import annotations

import base64
import binascii
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(os.environ.get('TRIPOSPLAT_ROOT', '/opt/triposplat'))
REPO = ROOT / 'repo'
VENV = ROOT / '.venv'
SCRIPT = ROOT / 'scripts' / 'run_job.py'
OUTPUT_ROOT = ROOT / 'outputs'
JOB_ROOT = ROOT / 'jobs'
SAMPLE_ROOT = REPO / 'static' / 'example_inputs'
VIEWER_ROOT = REPO / 'static' / 'viewer'
APP_VERSION = 'triposplat-web-2026-06-30'
PORT = int(os.environ.get('TRIPOSPLAT_PORT', '7871'))

for d in [OUTPUT_ROOT, JOB_ROOT, ROOT / 'scripts']:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='TripoSplat WebUI', version=APP_VERSION)
app.mount('/outputs', StaticFiles(directory=str(OUTPUT_ROOT)), name='outputs')
app.mount('/viewer', StaticFiles(directory=str(VIEWER_ROOT)), name='viewer')
app.mount('/samples', StaticFiles(directory=str(SAMPLE_ROOT)), name='samples')

_state_lock = threading.Lock()
_running: dict[str, Any] = {'job_id': None, 'proc': None}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def safe_int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(v)
    except Exception:
        return default
    return max(lo, min(hi, n))


def safe_float(v: Any, default: float, lo: float, hi: float) -> float:
    try:
        n = float(v)
    except Exception:
        return default
    return max(lo, min(hi, n))


def load_job(job_id: str) -> dict[str, Any]:
    p = JOB_ROOT / job_id / 'job.json'
    if not p.exists():
        raise HTTPException(status_code=404, detail='job not found')
    return json.loads(p.read_text())


def save_job(job: dict[str, Any]) -> None:
    d = JOB_ROOT / job['id']
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / 'job.json.tmp'
    tmp.write_text(json.dumps(job, indent=2, sort_keys=True))
    tmp.replace(d / 'job.json')


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for p in JOB_ROOT.glob('*/job.json'):
        try:
            jobs.append(json.loads(p.read_text()))
        except Exception:
            continue
    jobs.sort(key=lambda j: j.get('created_at', ''), reverse=True)
    return jobs[:limit]


def tail(path: str | Path, max_chars: int = 20000) -> str:
    p = Path(path)
    if not p.exists():
        return ''
    return p.read_bytes()[-max_chars:].decode('utf-8', errors='replace')


def safe_job_id(job_id: str) -> str:
    if not re.fullmatch(r'[0-9A-Za-z_\-]+', job_id):
        raise HTTPException(status_code=400, detail='invalid job id')
    return job_id


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    out_url = f'/outputs/{job["id"]}'
    result = dict(job)
    if job.get('status') == 'completed':
        result['urls'] = {
            'preprocessed': f'{out_url}/preprocessed_image.webp',
            'ply': f'{out_url}/output.ply',
            'splat': f'{out_url}/output.splat',
            'metadata': f'{out_url}/metadata.json',
            'viewer': f'/viewer/viewer.html?ply={out_url}/output.ply',
        }
    return result


def decode_data_url(data_url: str, target: Path) -> None:
    m = re.match(r'^data:([^;,]+)?(;base64)?,(.*)$', data_url, re.S)
    if not m:
        raise HTTPException(status_code=400, detail='invalid image_data_url')
    if not m.group(2):
        raise HTTPException(status_code=400, detail='image_data_url must be base64')
    try:
        raw = base64.b64decode(m.group(3), validate=False)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail=f'invalid base64: {exc}')
    if len(raw) > 32 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='image too large')
    target.write_bytes(raw)


class GenerateRequest(BaseModel):
    image_data_url: str | None = None
    sample: str | None = 'creature_butterfly.webp'
    seed: int = 42
    steps: int = 10
    guidance_scale: float = 3.0
    num_gaussians: int = 32768
    shift: float = 3.0
    erode_radius: int = 1


def run_job_thread(job_id: str) -> None:
    job = load_job(job_id)
    log_path = Path(job['log_path'])
    proc: subprocess.Popen[str] | None = None
    try:
        job['status'] = 'running'
        job['started_at'] = now_iso()
        save_job(job)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(REPO)
        env.setdefault('CUDA_VISIBLE_DEVICES', '0')
        cmd = [
            str(VENV / 'bin' / 'python'), str(SCRIPT),
            '--job-dir', str(OUTPUT_ROOT / job_id),
            '--input', job['input_path'],
            '--seed', str(job['params']['seed']),
            '--steps', str(job['params']['steps']),
            '--guidance-scale', str(job['params']['guidance_scale']),
            '--num-gaussians', str(job['params']['num_gaussians']),
            '--shift', str(job['params']['shift']),
            '--erode-radius', str(job['params']['erode_radius']),
        ]
        job['cmd'] = cmd
        save_job(job)
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'[webui] starting {job_id} at {now_iso()}\n')
            log.write('[webui] command: ' + ' '.join(cmd) + '\n')
            log.flush()
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, text=True, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with _state_lock:
                _running['job_id'] = job_id
                _running['proc'] = proc
            rc = proc.wait()
        job = load_job(job_id)
        job['returncode'] = rc
        job['finished_at'] = now_iso()
        if rc == 0 and (OUTPUT_ROOT / job_id / 'metrics.json').exists():
            metrics = json.loads((OUTPUT_ROOT / job_id / 'metrics.json').read_text())
            job['status'] = 'completed'
            job['metrics'] = metrics
            job['urls'] = public_job(job).get('urls', {})
            meta = {
                **job,
                'app_version': APP_VERSION,
                'repo': 'VAST-AI-Research/TripoSplat',
                'checkpoint_repo': 'VAST-AI/TripoSplat',
            }
            (OUTPUT_ROOT / job_id / 'metadata.json').write_text(json.dumps(meta, indent=2, sort_keys=True))
        else:
            job['status'] = 'failed'
            job['error'] = f'worker exited with rc={rc}'
        save_job(job)
    except Exception as exc:
        try:
            job = load_job(job_id)
        except Exception:
            job = {'id': job_id}
        job['status'] = 'failed'
        job['error'] = repr(exc)
        job['finished_at'] = now_iso()
        save_job(job)
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'ERROR: {exc!r}\n')
    finally:
        with _state_lock:
            if _running.get('job_id') == job_id:
                _running['job_id'] = None
                _running['proc'] = None


@app.get('/health')
def health() -> dict[str, Any]:
    return {'ok': True, 'version': APP_VERSION}


@app.get('/api/status')
def status() -> dict[str, Any]:
    with _state_lock:
        proc = _running.get('proc')
        running_id = _running.get('job_id') if proc and proc.poll() is None else None
    return {
        'ok': True,
        'version': APP_VERSION,
        'running_job_id': running_id,
        'repo': str(REPO),
        'venv': str(VENV),
        'outputs': str(OUTPUT_ROOT),
        'jobs': len(list(JOB_ROOT.glob('*/job.json'))),
        'samples': [p.name for p in sorted(SAMPLE_ROOT.glob('*')) if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}],
    }


@app.get('/api/jobs')
def api_jobs(limit: int = 30) -> dict[str, Any]:
    return {'ok': True, 'jobs': [public_job(j) for j in list_jobs(limit=safe_int(limit, 30, 1, 100))]}


@app.get('/api/jobs/{job_id}')
def api_job(job_id: str) -> dict[str, Any]:
    return {'ok': True, 'job': public_job(load_job(safe_job_id(job_id)))}


@app.get('/api/jobs/{job_id}/log')
def api_log(job_id: str) -> PlainTextResponse:
    job = load_job(safe_job_id(job_id))
    return PlainTextResponse(tail(job.get('log_path', '')))


@app.delete('/api/jobs/{job_id}')
def api_delete(job_id: str) -> dict[str, Any]:
    job_id = safe_job_id(job_id)
    with _state_lock:
        if _running.get('job_id') == job_id:
            raise HTTPException(status_code=409, detail='job is running')
    removed = []
    for p in [JOB_ROOT / job_id, OUTPUT_ROOT / job_id]:
        if p.exists():
            shutil.rmtree(p)
            removed.append(str(p))
    return {'ok': True, 'removed': removed}


@app.post('/api/jobs/{job_id}/stop')
def api_stop(job_id: str) -> dict[str, Any]:
    job_id = safe_job_id(job_id)
    with _state_lock:
        proc = _running.get('proc') if _running.get('job_id') == job_id else None
    if not proc or proc.poll() is not None:
        return {'ok': True, 'stopped': False, 'message': 'not running'}
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    job = load_job(job_id)
    job['status'] = 'stopping'
    save_job(job)
    return {'ok': True, 'stopped': True}


@app.post('/api/generate')
def api_generate(req: GenerateRequest) -> dict[str, Any]:
    with _state_lock:
        proc = _running.get('proc')
        if proc and proc.poll() is None:
            raise HTTPException(status_code=409, detail=f'job already running: {_running.get("job_id")}')
    job_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
    job_dir = JOB_ROOT / job_id
    out_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / 'input_image'
    sample_name = req.sample or 'creature_butterfly.webp'
    if req.image_data_url:
        input_path = job_dir / 'input_upload.png'
        decode_data_url(req.image_data_url, input_path)
        source = 'upload'
    else:
        sample = SAMPLE_ROOT / Path(sample_name).name
        if not sample.exists():
            raise HTTPException(status_code=400, detail='sample not found')
        input_path = job_dir / sample.name
        shutil.copy2(sample, input_path)
        source = f'sample:{sample.name}'
    params = {
        'seed': safe_int(req.seed, 42, 0, 2_147_483_647),
        'steps': safe_int(req.steps, 10, 1, 50),
        'guidance_scale': safe_float(req.guidance_scale, 3.0, 1.0, 10.0),
        'num_gaussians': safe_int(req.num_gaussians, 32768, 32768, 262144),
        'shift': safe_float(req.shift, 3.0, 1.0, 8.0),
        'erode_radius': safe_int(req.erode_radius, 1, 0, 8),
    }
    job = {
        'id': job_id,
        'status': 'queued',
        'created_at': now_iso(),
        'input_path': str(input_path),
        'source': source,
        'params': params,
        'log_path': str(job_dir / 'job.log'),
        'output_dir': str(out_dir),
    }
    save_job(job)
    thread = threading.Thread(target=run_job_thread, args=(job_id,), daemon=True)
    thread.start()
    return {'ok': True, 'job': public_job(job)}


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TripoSplat | Spark</title>
<style>
:root{--bg:#0a0807;--panel:#11100f;--panel2:#161412;--text:#f3eee7;--dim:#9b9187;--line:#2a2520;--green:#5FE3A0;--warn:#f0b35a;--bad:#ff6b6b;--blue:#7db7ff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#17251d 0,#0a0807 36%,#070605 100%);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui}header{padding:22px 26px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:18px;align-items:center}h1{margin:0;font-size:24px;letter-spacing:.04em}.sub{color:var(--dim);font-size:12px}.wrap{display:grid;grid-template-columns:390px 1fr;gap:18px;padding:18px}.card{background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.018));border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:0 12px 40px rgba(0,0,0,.25)}label{display:block;margin:12px 0 5px;color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.06em}input,select,button{font:inherit}input,select{width:100%;background:#090807;color:var(--text);border:1px solid var(--line);border-radius:10px;padding:9px}button,.btn{border:1px solid rgba(95,227,160,.35);background:rgba(95,227,160,.1);color:var(--green);border-radius:999px;padding:8px 12px;cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;gap:6px}button:hover,.btn:hover{background:rgba(95,227,160,.17)}button.danger{border-color:rgba(255,107,107,.35);background:rgba(255,107,107,.08);color:var(--bad)}button.secondary,.btn.secondary{border-color:rgba(125,183,255,.35);background:rgba(125,183,255,.08);color:var(--blue)}button:disabled{opacity:.45;cursor:not-allowed}.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}.actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}.status{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--green);white-space:pre-wrap}.viewer{height:620px;border:1px solid var(--line);border-radius:16px;overflow:hidden;background:#050505}.viewer iframe{width:100%;height:100%;border:0}.empty{height:100%;display:grid;place-items:center;color:var(--dim);text-align:center;padding:40px}.library{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;margin-top:14px}.item{border:1px solid var(--line);border-radius:14px;background:rgba(0,0,0,.18);padding:10px}.item img{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:10px;border:1px solid var(--line);background:#000}.item-title{font-weight:700;font-size:13px;margin:8px 0 3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.meta{font-size:12px;color:var(--dim)}pre{background:#070605;border:1px solid var(--line);border-radius:12px;padding:10px;max-height:240px;overflow:auto;color:#d7cfc6}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 8px;color:var(--dim);font-size:11px;margin-left:6px}.ok{color:var(--green)}.bad{color:var(--bad)}.warn{color:var(--warn)}@media(max-width:980px){.wrap{grid-template-columns:1fr}.viewer{height:520px}}
</style>
</head>
<body>
<header><div><h1>TripoSplat <span class="pill">Spark WebUI</span></h1><div class="sub">single image → 3D Gaussian PLY/SPLAT · per-job model process · no resident model while idle</div></div><div class="status" id="top-status">syncing...</div></header>
<div class="wrap">
  <aside class="card">
    <h2 style="margin-top:0">Generate</h2>
    <label>Input image</label><input id="file" type="file" accept="image/*">
    <label>Or sample</label><select id="sample"></select>
    <div class="row"><div><label>Seed</label><input id="seed" type="number" value="42"></div><div><label>Steps</label><input id="steps" type="number" min="1" max="50" value="10"></div></div>
    <div class="row"><div><label>Guidance</label><input id="guidance" type="number" step="0.5" value="3.0"></div><div><label>Gaussians</label><select id="gaussians"><option>32768</option><option>65536</option><option>131072</option><option>262144</option></select></div></div>
    <div class="row"><div><label>Shift</label><input id="shift" type="number" step="0.5" value="3.0"></div><div><label>Erode radius</label><input id="erode" type="number" value="1"></div></div>
    <div class="actions"><button id="generate">Generate</button><button class="secondary" id="refresh">Refresh</button></div>
    <div class="sub" style="margin-top:12px">Tip: 32k/10 steps is a quick preview; 262k/20 steps is the quality path.</div>
    <h3>Current job</h3><div id="job-box" class="status">none</div><pre id="log"></pre>
  </aside>
  <main>
    <div class="card"><div style="display:flex;justify-content:space-between;align-items:center;gap:10px"><h2 style="margin:0">3D Gaussian Preview</h2><div id="viewer-actions" class="actions"></div></div><div class="viewer" id="viewer"><div class="empty">Open a completed library item to view the generated Gaussian splat.</div></div></div>
    <div class="card" style="margin-top:18px"><h2 style="margin-top:0">Generated asset library</h2><div id="library" class="library"></div></div>
  </main>
</div>
<script>
let currentJob = null;
let pollTimer = null;
async function api(path, opts){ const r = await fetch(path, opts); if(!r.ok) throw new Error(await r.text()); const ct=r.headers.get('content-type')||''; return ct.includes('json') ? r.json() : r.text(); }
function readFileDataURL(file){ return new Promise((res,rej)=>{ const fr=new FileReader(); fr.onload=()=>res(fr.result); fr.onerror=rej; fr.readAsDataURL(file); }); }
function fmtJob(j){ if(!j) return 'none'; const p=j.params||{}; return `${j.id}\nstatus: ${j.status}\nsource: ${j.source||''}\nseed ${p.seed} · steps ${p.steps} · cfg ${p.guidance_scale} · ${p.num_gaussians} gaussians`; }
async function refreshStatus(){
  const s = await api('/api/status');
  document.getElementById('top-status').textContent = `jobs ${s.jobs} · running ${s.running_job_id||'none'} · ${s.version}`;
  const sel = document.getElementById('sample');
  if(!sel.dataset.loaded){ sel.innerHTML = (s.samples||[]).map(x=>`<option>${x}</option>`).join(''); sel.dataset.loaded='1'; }
}
async function refreshJobs(){
  await refreshStatus();
  const j = await api('/api/jobs?limit=40');
  const jobs = j.jobs || [];
  const running = jobs.find(x=>['queued','running','stopping'].includes(x.status));
  if(running) currentJob = running.id;
  document.getElementById('job-box').textContent = fmtJob(running || (currentJob ? jobs.find(x=>x.id===currentJob) : null));
  if(currentJob){ try { document.getElementById('log').textContent = await api(`/api/jobs/${currentJob}/log`); } catch(e){} }
  renderLibrary(jobs);
  if(!running && pollTimer){ clearInterval(pollTimer); pollTimer=null; }
}
function renderLibrary(jobs){
  const lib = document.getElementById('library');
  const complete = jobs.filter(j=>j.status==='completed');
  if(!complete.length){ lib.innerHTML = '<div class="sub">No completed TripoSplat jobs yet.</div>'; return; }
  lib.innerHTML = complete.map(j=>{
    const u=j.urls||{}; const m=j.metrics||{}; const p=j.params||{};
    return `<div class="item">
      ${u.preprocessed ? `<img src="${u.preprocessed}" onclick="openJob('${j.id}')" title="Open 3D preview">` : ''}
      <div class="item-title">${j.id}</div>
      <div class="meta">${m.gaussians||p.num_gaussians||''} gaussians · ${m.total_sec||'?'}s</div>
      <div class="meta">seed ${p.seed} · steps ${p.steps} · cfg ${p.guidance_scale}</div>
      <div class="actions">
        <button onclick="openJob('${j.id}')">Open 3D</button>
        ${u.ply ? `<a class="btn secondary" href="${u.ply}" download>PLY</a>` : ''}
        ${u.splat ? `<a class="btn secondary" href="${u.splat}" download>SPLAT</a>` : ''}
        ${u.metadata ? `<a class="btn secondary" href="${u.metadata}" target="_blank">Metadata</a>` : ''}
        <button class="danger" onclick="deleteJob('${j.id}')">Delete</button>
      </div>
    </div>`;
  }).join('');
}
async function openJob(id){
  const r = await api(`/api/jobs/${id}`); const j = r.job; const u=j.urls||{};
  if(!u.viewer) return alert('No viewer URL for job');
  document.getElementById('viewer').innerHTML = `<iframe src="${u.viewer}&ts=${Date.now()}"></iframe>`;
  document.getElementById('viewer-actions').innerHTML = `<a class="btn secondary" href="${u.ply}" download>Download PLY</a><a class="btn secondary" href="${u.splat}" download>Download SPLAT</a><a class="btn secondary" href="${u.metadata}" target="_blank">Metadata</a>`;
}
async function deleteJob(id){ if(!confirm(`Delete TripoSplat job ${id}?`)) return; await api(`/api/jobs/${id}`, {method:'DELETE'}); await refreshJobs(); }
async function generate(){
  const file = document.getElementById('file').files[0];
  const body = {
    sample: document.getElementById('sample').value,
    seed: parseInt(document.getElementById('seed').value||'42'),
    steps: parseInt(document.getElementById('steps').value||'10'),
    guidance_scale: parseFloat(document.getElementById('guidance').value||'3'),
    num_gaussians: parseInt(document.getElementById('gaussians').value||'32768'),
    shift: parseFloat(document.getElementById('shift').value||'3'),
    erode_radius: parseInt(document.getElementById('erode').value||'1')
  };
  if(file) body.image_data_url = await readFileDataURL(file);
  const r = await api('/api/generate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  currentJob = r.job.id;
  document.getElementById('job-box').textContent = fmtJob(r.job);
  if(!pollTimer) pollTimer=setInterval(refreshJobs, 2000);
  await refreshJobs();
}
document.getElementById('generate').onclick = () => generate().catch(e=>alert(e.message));
document.getElementById('refresh').onclick = () => refreshJobs().catch(e=>alert(e.message));
refreshJobs(); setInterval(refreshStatus, 5000);
</script>
</body>
</html>'''


@app.get('/')
def index() -> HTMLResponse:
    return HTMLResponse(HTML)
