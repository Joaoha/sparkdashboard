from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from urllib.error import URLError

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(os.environ.get('AGENT3DIFY_ROOT', '/opt/agent3dify'))
REPO = ROOT / 'repo'
VENV = ROOT / '.venv'
JOBS = ROOT / 'jobs'
OUTPUTS = ROOT / 'outputs'
LOGS = ROOT / 'logs'
MANIFEST = ROOT / 'manifest.json'
VLLM_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1').rstrip('/')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'local-qwen')
DEFAULT_MODEL = os.environ.get('AGENT3DIFY_LOCAL_QWEN_MODEL')

for p in [JOBS, OUTPUTS, LOGS]:
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='Agent3Dify Local Qwen WebUI')
app.mount('/outputs', StaticFiles(directory=str(OUTPUTS)), name='outputs')
app.mount('/jobs', StaticFiles(directory=str(JOBS)), name='jobs')

_lock = threading.Lock()
_active: str | None = None
_jobs: dict[str, dict[str, Any]] = {}


def _load_manifest() -> list[dict[str, Any]]:
    if not MANIFEST.exists():
        return []
    try:
        data = json.loads(MANIFEST.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_manifest(items: list[dict[str, Any]]) -> None:
    tmp = MANIFEST.with_suffix('.tmp')
    tmp.write_text(json.dumps(items[-100:], indent=2, ensure_ascii=False))
    tmp.replace(MANIFEST)


def _detect_qwen_model() -> str:
    if DEFAULT_MODEL:
        return DEFAULT_MODEL
    try:
        with urlopen(f'{VLLM_BASE_URL}/models', timeout=5) as r:
            payload = json.loads(r.read().decode('utf-8'))
        data = payload.get('data') if isinstance(payload, dict) else None
        if isinstance(data, list) and data:
            mid = data[0].get('id')
            if isinstance(mid, str) and mid.strip():
                return mid
    except Exception:
        pass
    return 'qwen-local'


def _vllm_ok() -> tuple[bool, str | None]:
    try:
        with urlopen(f'{VLLM_BASE_URL}/models', timeout=5) as r:
            payload = json.loads(r.read().decode('utf-8'))
        return True, json.dumps(payload)
    except Exception as e:
        return False, str(e)


def _status_payload() -> dict[str, Any]:
    ok, detail = _vllm_ok()
    return {
        'ok': True,
        'service': 'agent3dify-web',
        'vllm_base_url': VLLM_BASE_URL,
        'vllm_ok': ok,
        'vllm_detail': detail,
        'detected_model': _detect_qwen_model(),
        'active_job': _active,
        'jobs': list(_jobs.values())[-20:][::-1],
        'history': _load_manifest()[::-1],
    }


def _copy_tree_artifacts(workspace: Path, out_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    source = workspace / 'drawing_to_cad_workspace'
    if not source.exists():
        return copied
    for rel_root in ['generated', 'artifacts', 'review', 'preprocessed', 'input']:
        src = source / rel_root
        if src.exists():
            dst = out_dir / rel_root
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    for path in out_dir.rglob('*'):
        if path.is_file():
            rel = path.relative_to(out_dir).as_posix()
            copied[rel] = f'/outputs/{out_dir.name}/{rel}'
    return copied


def _run_job(job_id: str) -> None:
    global _active
    job = _jobs[job_id]
    job_dir = Path(job['job_dir'])
    out_dir = Path(job['out_dir'])
    log_path = Path(job['log_path'])
    model_id = job['model_id']
    image_editor_model = job.get('image_editor_model') or 'local:pillow'
    view_detector_model = job.get('view_detector_model') or 'local:none'

    cmd = [
        str(VENV / 'bin' / 'agent3dify'),
        '--drawing', str(job_dir / 'input.png'),
        '--model', f'openai:{model_id}',
        '--builder-model', f'openai:{model_id}',
        '--verifier-model', f'openai:{model_id}',
        '--image-editor-model', image_editor_model,
        '--view-detector-model', view_detector_model,
    ]
    env = os.environ.copy()
    env.update({
        'OPENAI_BASE_URL': VLLM_BASE_URL,
        'OPENAI_API_KEY': OPENAI_API_KEY,
        'AGENT3DIFY_LOCAL_QWEN_MODEL': model_id,
        'SUPERVISOR_MODEL': f'openai:{model_id}',
        'BUILDER_MODEL': f'openai:{model_id}',
        'VERIFIER_MODEL': f'openai:{model_id}',
        'IMAGE_EDITOR_MODEL': image_editor_model,
        'VIEW_DETECTOR_MODEL': view_detector_model,
        'LANGSMITH_TRACING': 'false',
        'PYTHONUNBUFFERED': '1',
    })

    job['status'] = 'running'
    job['started_at'] = datetime.now().isoformat(timespec='seconds')
    job['cmd'] = ' '.join(cmd)
    start = time.time()
    try:
        with log_path.open('w', encoding='utf-8') as log:
            log.write(f'Agent3Dify local-Qwen job {job_id}\n')
            log.write(f'vLLM: {VLLM_BASE_URL}\nmodel: {model_id}\ncmd: {job["cmd"]}\n\n')
            log.flush()
            proc = subprocess.Popen(cmd, cwd=str(job_dir), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            job['pid'] = proc.pid
            rc = proc.wait()
        job['returncode'] = rc
        job['duration_sec'] = round(time.time() - start, 3)
        artifacts = _copy_tree_artifacts(job_dir, out_dir)
        job['artifacts'] = artifacts
        job['status'] = 'completed' if rc == 0 else 'failed'
    except Exception as e:
        job['status'] = 'failed'
        job['error'] = repr(e)
        try:
            log_path.write_text((log_path.read_text() if log_path.exists() else '') + f'\nERROR: {e!r}\n')
        except Exception:
            pass
    finally:
        job['finished_at'] = datetime.now().isoformat(timespec='seconds')
        items = _load_manifest()
        items.append({k: v for k, v in job.items() if k not in {'pid'}})
        _save_manifest(items)
        with _lock:
            if _active == job_id:
                _active = None


@app.get('/health')
def health():
    return {'ok': True}


@app.get('/api/status')
def api_status():
    return _status_payload()


@app.get('/api/jobs/{job_id}')
def api_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        items = [x for x in _load_manifest() if x.get('id') == job_id]
        if not items:
            raise HTTPException(404, 'job not found')
        job = items[-1]
    log_path = Path(job.get('log_path', ''))
    text = ''
    if log_path.exists():
        text = log_path.read_text(errors='replace')[-20000:]
    return {**job, 'log_tail': text}


@app.post('/api/generate')
def api_generate(
    drawing: UploadFile = File(...),
    model_id: str = Form(''),
    image_editor_model: str = Form('local:pillow'),
    view_detector_model: str = Form('local:none'),
):
    global _active
    with _lock:
        if _active is not None:
            raise HTTPException(409, f'job already running: {_active}')
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        job_id = f'{now}_{uuid.uuid4().hex[:8]}'
        job_dir = JOBS / job_id
        out_dir = OUTPUTS / job_id
        job_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)
        content = drawing.file.read()
        if not content:
            raise HTTPException(400, 'empty drawing upload')
        (job_dir / 'input.png').write_bytes(content)
        mid = model_id.strip() or _detect_qwen_model()
        job = {
            'id': job_id,
            'status': 'queued',
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'job_dir': str(job_dir),
            'out_dir': str(out_dir),
            'log_path': str(LOGS / f'{job_id}.log'),
            'model_id': mid,
            'image_editor_model': image_editor_model,
            'view_detector_model': view_detector_model,
            'input_name': drawing.filename,
        }
        _jobs[job_id] = job
        _active = job_id
        threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
        return job


@app.post('/api/cancel')
def api_cancel():
    global _active
    if not _active:
        return {'ok': True, 'message': 'no active job'}
    job = _jobs.get(_active)
    if not job or not job.get('pid'):
        return {'ok': False, 'message': 'active job has no pid yet'}
    try:
        os.kill(int(job['pid']), 15)
        job['status'] = 'cancelled'
        return {'ok': True, 'cancelled': _active}
    except Exception as e:
        return {'ok': False, 'error': repr(e)}


@app.get('/')
def index():
    return HTMLResponse(r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Agent3Dify Local Qwen</title>
<style>
body{font-family:Inter,system-ui,sans-serif;background:#0a0807;color:#eee;margin:0;padding:24px}a{color:#5FE3A0}.card{border:1px solid #2b332f;background:#111;padding:18px;border-radius:14px;margin:12px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#b8f4d3;white-space:pre-wrap}button,input{font:inherit;padding:10px;border-radius:8px;border:1px solid #3a4;background:#161;color:#eee}button{background:#5FE3A0;color:#06120b;font-weight:700;cursor:pointer}small{color:#aaa}.badge{display:inline-block;border:1px solid #375;padding:3px 8px;border-radius:999px;color:#5FE3A0}</style>
</head><body>
<h1>Agent3Dify · Local Qwen</h1>
<p>2D drawing → CadQuery STEP/STL workflow using Spark's local OpenAI-compatible vLLM endpoint.</p>
<div class="card"><div id="status" class="mono">loading…</div></div>
<div class="grid">
<div class="card"><h2>New job</h2><form id="f"><p><input type="file" name="drawing" accept="image/*" required></p><p><label>Model ID override<br><input name="model_id" placeholder="auto-detect from /v1/models" style="width:100%"></label></p><p><label>Image editor model<br><input name="image_editor_model" value="local:pillow" style="width:100%"></label></p><p><label>View detector model<br><input name="view_detector_model" value="local:none" style="width:100%"></label></p><button>Run Agent3Dify</button></form><p><button onclick="cancelJob()">Cancel active job</button></p></div>
<div class="card"><h2>Active/recent</h2><div id="jobs"></div></div>
</div>
<div class="card"><h2>Selected job log/artifacts</h2><div id="job" class="mono">Select a job…</div></div>
<script>
let selected=null;
async function refresh(){const s=await fetch('/api/status').then(r=>r.json());status.textContent=JSON.stringify({vllm_ok:s.vllm_ok,detected_model:s.detected_model,active_job:s.active_job,vllm_base_url:s.vllm_base_url},null,2); const all=[...s.jobs,...s.history].filter((v,i,a)=>a.findIndex(x=>x.id===v.id)===i).slice(0,30); jobs.innerHTML=all.map(j=>`<p><span class="badge">${j.status}</span> <a href="#" onclick="sel('${j.id}')">${j.id}</a><br><small>${j.model_id||''}</small></p>`).join('')||'No jobs yet'; if(selected) await loadJob(selected)}
async function sel(id){selected=id;await loadJob(id)}
async function loadJob(id){const j=await fetch('/api/jobs/'+id).then(r=>r.json()); let arts=j.artifacts||{}; let links=Object.entries(arts).map(([k,v])=>`<a href="${v}" target="_blank">${k}</a>`).join('\n'); job.innerHTML=`status: ${j.status}\nmodel: ${j.model_id}\nreturncode: ${j.returncode}\nduration: ${j.duration_sec}\n\nARTIFACTS:\n${links}\n\nLOG:\n`+escapeHtml(j.log_tail||'')}
function escapeHtml(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
f.onsubmit=async e=>{e.preventDefault();let fd=new FormData(f);let r=await fetch('/api/generate',{method:'POST',body:fd});let j=await r.json(); if(!r.ok){alert(JSON.stringify(j));return} selected=j.id; refresh()}
async function cancelJob(){await fetch('/api/cancel',{method:'POST'});refresh()}
setInterval(refresh,3000);refresh();
</script></body></html>''')
