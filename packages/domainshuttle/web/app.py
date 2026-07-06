from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(os.environ.get("DOMAINSHUTTLE_ROOT", "/opt/domainshuttle"))
WEB_ROOT = ROOT / "web"
INPUT_ROOT = ROOT / "inputs" / "web"
OUTPUT_ROOT = ROOT / "outputs"
JOB_ROOT = ROOT / "jobs"
SCRIPT = ROOT / "scripts" / "run_smoke.sh"
SAMPLE_REFERENCE = ROOT / "inputs" / "reference_spark_flux_crab.png"
PUBLIC_BASE = os.environ.get("DOMAINSHUTTLE_PUBLIC_BASE", "http://localhost:7867")

HEAVY_SERVICES = [
    "z-image.service",
    "qwen-image.service",
    "flux2.service",
    "hidream-o1.service",
    "pixal3d.service",
    "qwen-nvfp4-vllm.service",
    "qwen-vllm.service",
    "personaplex.service",
    "personaplex-bnb4.service",
]

APP_VERSION = "domainshuttle-web-2026-06-28"

for d in [WEB_ROOT, INPUT_ROOT, OUTPUT_ROOT, JOB_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DomainShuttle WebUI", version=APP_VERSION)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")
app.mount("/inputs", StaticFiles(directory=str(ROOT / "inputs")), name="inputs")

_state_lock = threading.Lock()
_running: dict[str, Any] = {"job_id": None, "proc": None}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        v = int(value)
    except Exception:
        return default
    return max(minimum, min(maximum, v))


def safe_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return max(minimum, min(maximum, v))


def load_job(job_id: str) -> dict[str, Any]:
    path = JOB_ROOT / job_id / "job.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="job not found")
    return json.loads(path.read_text())


def save_job(job: dict[str, Any]) -> None:
    job_dir = JOB_ROOT / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    tmp = job_dir / "job.json.tmp"
    tmp.write_text(json.dumps(job, indent=2, sort_keys=True))
    tmp.replace(job_dir / "job.json")


def list_jobs(limit: int = 30) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for path in JOB_ROOT.glob("*/job.json"):
        try:
            jobs.append(json.loads(path.read_text()))
        except Exception:
            continue
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs[:limit]


def tail(path: str | Path, max_chars: int = 16000) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    data = p.read_bytes()[-max_chars:]
    return data.decode("utf-8", errors="replace")


def systemctl(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["systemctl", "--user", *args], text=True, capture_output=True, timeout=timeout)


def service_active(unit: str) -> bool:
    try:
        return systemctl("is-active", unit, timeout=10).stdout.strip() == "active"
    except Exception:
        return False


def append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def manage_heavy_services(job: dict[str, Any], log_path: Path) -> dict[str, bool]:
    pre: dict[str, bool] = {}
    append_log(log_path, "[webui] Checking heavyweight services before DomainShuttle run...")
    for unit in HEAVY_SERVICES:
        pre[unit] = service_active(unit)
    job["pre_services"] = pre
    save_job(job)

    # Unload FLUX.2 first if its web service is active.
    try:
        subprocess.run(["curl", "-fsS", "-X", "POST", "http://127.0.0.1:7866/api/unload"], text=True, capture_output=True, timeout=60)
    except Exception:
        pass

    to_stop = [unit for unit, active in pre.items() if active]
    if not to_stop:
        append_log(log_path, "[webui] No active heavyweight services to stop.")
        return pre
    append_log(log_path, "[webui] Stopping active heavyweight services: " + ", ".join(to_stop))
    for unit in to_stop:
        try:
            proc = systemctl("stop", unit, timeout=120)
            if proc.returncode != 0:
                append_log(log_path, f"[webui] stop {unit}: rc={proc.returncode} stderr={proc.stderr.strip()}")
        except Exception as exc:
            append_log(log_path, f"[webui] stop {unit}: {exc}")
    time.sleep(3)
    return pre


def restore_services(pre: dict[str, bool], log_path: Path) -> None:
    active_before = [unit for unit, was_active in pre.items() if was_active]
    if not active_before:
        return
    append_log(log_path, "[webui] Restoring services that were active before run: " + ", ".join(active_before))
    for unit in active_before:
        try:
            proc = systemctl("start", unit, timeout=180)
            if proc.returncode != 0:
                append_log(log_path, f"[webui] start {unit}: rc={proc.returncode} stderr={proc.stderr.strip()}")
        except Exception as exc:
            append_log(log_path, f"[webui] start {unit}: {exc}")


def run_job_thread(job_id: str) -> None:
    job = load_job(job_id)
    log_path = Path(job["log_path"])
    append_log(log_path, f"[webui] Job {job_id} started at {now_iso()}")
    proc: subprocess.Popen[str] | None = None
    try:
        job["status"] = "running"
        job["started_at"] = now_iso()
        save_job(job)

        pre: dict[str, bool] = {}
        if job["params"].get("auto_manage_services", True):
            pre = manage_heavy_services(job, log_path)

        env = os.environ.copy()
        env.update({
            "HF_HOME": str(ROOT / "hf-cache"),
            "VIDEOX_ATTENTION_TYPE": "SDPA",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "INPUT_JSON": job["input_json"],
            "OUTPUT_DIR": job["output_dir"],
            "HEIGHT": str(job["params"]["height"]),
            "WIDTH": str(job["params"]["width"]),
            "VIDEO_LENGTH": str(job["params"]["video_length"]),
            "STEPS": str(job["params"]["steps"]),
            "FPS": str(job["params"]["fps"]),
            "SEED": str(job["params"]["seed"]),
            "SHIFT": str(job["params"]["shift"]),
            "GUIDANCE_A": str(job["params"]["guidance_a"]),
            "GUIDANCE_B": str(job["params"]["guidance_b"]),
            "MEMORY_MODE": "model_full_load",
        })
        append_log(log_path, "[webui] Launching DomainShuttle subprocess...")
        append_log(log_path, "[webui] Output directory: " + job["output_dir"])
        with log_path.open("a", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                [str(SCRIPT)],
                cwd=str(ROOT),
                env=env,
                text=True,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with _state_lock:
                _running["job_id"] = job_id
                _running["proc"] = proc
            job["pid"] = proc.pid
            save_job(job)
            rc = proc.wait()
        job = load_job(job_id)
        job["returncode"] = rc
        job["finished_at"] = now_iso()
        videos = sorted(Path(job["output_dir"]).glob("*.mp4"))
        images = sorted(Path(job["output_dir"]).glob("*.png"))
        job["outputs"] = [str(p) for p in videos + images]
        if videos:
            job["video_url"] = "/outputs/" + str(videos[0].relative_to(OUTPUT_ROOT))
        elif images:
            job["video_url"] = "/outputs/" + str(images[0].relative_to(OUTPUT_ROOT))
        if rc == 0 and (videos or images):
            job["status"] = "completed"
            append_log(log_path, f"[webui] Job completed successfully at {job['finished_at']}")
        elif job.get("status") == "cancelled":
            append_log(log_path, "[webui] Job cancelled.")
        else:
            job["status"] = "failed"
            append_log(log_path, f"[webui] Job failed rc={rc} at {job['finished_at']}")
        save_job(job)
        if job["params"].get("restore_services", True):
            restore_services(pre, log_path)
    except Exception as exc:
        job = load_job(job_id)
        job["status"] = "failed"
        job["error"] = repr(exc)
        job["finished_at"] = now_iso()
        append_log(log_path, f"[webui] ERROR: {exc!r}")
        save_job(job)
    finally:
        with _state_lock:
            if _running.get("job_id") == job_id:
                _running["job_id"] = None
                _running["proc"] = None


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DomainShuttle WebUI · Spark</title>
  <style>
    :root{color-scheme:dark;--bg:#0a0807;--panel:#11100e;--line:#2a2621;--text:#f5efe6;--muted:#a79d91;--green:#5FE3A0;--warn:#ffc857;--bad:#ff6b6b;--blue:#77aaff;}
    *{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at 20% 0%,#173222 0,#0a0807 35%,#070605 100%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
    header{padding:28px 32px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:16px;align-items:flex-end;}
    h1{margin:0;font-size:34px;letter-spacing:-.04em}.sub{color:var(--muted);font-size:13px}.pill{border:1px solid var(--line);border-radius:999px;padding:8px 12px;color:var(--green);font-family:ui-monospace,monospace;font-size:12px;background:#0f1410}
    main{display:grid;grid-template-columns:minmax(360px,520px) 1fr;gap:18px;padding:22px;max-width:1500px;margin:0 auto}.card{border:1px solid var(--line);background:rgba(17,16,14,.92);border-radius:18px;padding:18px;box-shadow:0 18px 60px #0008}.card h2{margin:0 0 12px;font-size:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.field{display:flex;flex-direction:column;gap:6px;margin-bottom:12px}label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}input,textarea,select{width:100%;border:1px solid var(--line);background:#090807;color:var(--text);border-radius:10px;padding:10px;font:inherit}textarea{min-height:120px;resize:vertical}.hint{font-size:12px;color:var(--muted);line-height:1.45}.btn{border:1px solid #416b55;background:linear-gradient(180deg,#1c5f3f,#113622);color:#eafff3;border-radius:12px;padding:12px 16px;font-weight:700;cursor:pointer}.btn.secondary{background:#10100f;color:var(--text);border-color:var(--line)}.btn.danger{background:#3b1717;border-color:#773232}.btn:disabled{opacity:.5;cursor:not-allowed}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.status{font-family:ui-monospace,monospace;font-size:13px;border:1px solid var(--line);border-radius:12px;padding:10px;background:#090807}.ok{color:var(--green)}.warn{color:var(--warn)}.bad{color:var(--bad)}pre{white-space:pre-wrap;word-break:break-word;background:#050505;border:1px solid var(--line);border-radius:14px;padding:12px;max-height:420px;overflow:auto;font-size:12px;line-height:1.45;color:#ddd}.jobs{display:grid;gap:10px}.job{border:1px solid var(--line);border-radius:14px;padding:12px;background:#0c0b0a}.job a{color:var(--green)}video,img.preview{width:100%;border:1px solid var(--line);border-radius:14px;background:#000}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}.small{font-size:12px;color:var(--muted)}@media(max-width:1000px){main{grid-template-columns:1fr}header{display:block}}
  </style>
</head>
<body>
<header>
  <div><h1>DomainShuttle WebUI</h1><div class="sub">Subject-driven Wan2.2 video generation on Spark · one queued GPU job at a time</div></div>
  <div class="pill" id="health">checking…</div>
</header>
<main>
  <section class="card">
    <h2>Generate video</h2>
    <form id="genForm">
      <div class="field"><label>Reference image</label><input type="file" name="reference" accept="image/*" /><div class="hint">Leave empty to use the verified Spark FLUX crab reference.</div></div>
      <div class="field"><label>Prompt</label><textarea name="prompt">A short cinematic video of the same hermit crab with the blue Spark FLUX soda can shell slowly walking across a sunny beach, natural motion, sharp details.</textarea></div>
      <div class="grid">
        <div class="field"><label>Width</label><input name="width" type="number" min="128" max="832" step="8" value="448" /></div>
        <div class="field"><label>Height</label><input name="height" type="number" min="128" max="480" step="8" value="256" /></div>
        <div class="field"><label>Frames</label><input name="video_length" type="number" min="5" max="81" step="4" value="17" /></div>
        <div class="field"><label>FPS</label><input name="fps" type="number" min="1" max="24" value="8" /></div>
        <div class="field"><label>Steps</label><input name="steps" type="number" min="1" max="40" value="4" /></div>
        <div class="field"><label>Seed</label><input name="seed" type="number" value="42" /></div>
        <div class="field"><label>Shift</label><input name="shift" type="number" step="0.1" value="5" /></div>
        <div class="field"><label>Guidance</label><input name="guidance_a" type="number" step="0.1" value="4.0" /></div>
      </div>
      <input name="guidance_b" type="hidden" value="3.0" />
      <div class="field"><label>Domain</label><select name="domain"><option value="object">Object</option><option value="Human">Human</option><option value="Fantasy Domain">Fantasy Domain</option><option value="Background">Background</option><option value="others">Other</option></select></div>
      <div class="row small"><label><input type="checkbox" name="auto_manage_services" checked style="width:auto"> Stop/unload other heavy services before run</label></div>
      <div class="row small"><label><input type="checkbox" name="restore_services" checked style="width:auto"> Restore services that were active before run</label></div>
      <p class="hint">Verified mode is <b>model_full_load</b>. This can use ~80GiB unified/model-resident memory even for tiny runs. Start small.</p>
      <div class="row"><button class="btn" id="submitBtn">Start DomainShuttle job</button><button class="btn secondary" type="button" onclick="refresh()">Refresh</button><button class="btn danger" type="button" onclick="cancelActive()">Cancel active</button></div>
    </form>
  </section>
  <section class="card">
    <h2>Active / latest job</h2>
    <div id="active" class="status">No job loaded yet.</div>
    <h2 style="margin-top:18px">Live log</h2>
    <pre id="log"></pre>
  </section>
  <section class="card" style="grid-column:1 / -1">
    <h2>Outputs</h2>
    <div id="gallery" class="gallery"></div>
  </section>
  <section class="card" style="grid-column:1 / -1">
    <h2>Recent jobs</h2>
    <div id="jobs" class="jobs"></div>
  </section>
</main>
<script>
let currentJob = null;
function clsStatus(s){return s==='completed'?'ok':s==='failed'?'bad':s==='running'?'warn':''}
async function api(path, opts){ const r=await fetch(path, opts); if(!r.ok){throw new Error(await r.text())} return await r.json(); }
async function refresh(){
  try{
    const st=await api('/api/status');
    document.getElementById('health').textContent = st.running_job ? `RUNNING ${st.running_job}` : 'READY';
    document.getElementById('health').className = 'pill ' + (st.running_job?'warn':'ok');
    const jobs=st.jobs||[];
    if(!currentJob && jobs.length) currentJob=jobs[0].id;
    renderJobs(jobs);
    if(currentJob) await loadJob(currentJob);
  }catch(e){document.getElementById('health').textContent='ERROR'; document.getElementById('active').textContent=String(e)}
}
function renderJobs(jobs){
  document.getElementById('jobs').innerHTML = jobs.map(j=>`<div class="job"><div class="row"><b>${j.id}</b><span class="${clsStatus(j.status)}">${j.status}</span><span class="small">${j.created_at||''}</span><button class="btn secondary" onclick="currentJob='${j.id}';loadJob('${j.id}')">View</button></div><div class="small">${(j.params&&j.params.prompt||'').slice(0,180)}</div></div>`).join('') || '<div class="small">No jobs yet.</div>';
  const outs = jobs.filter(j=>j.video_url).map(j=>`<div><video src="${j.video_url}" controls muted loop></video><div class="small"><a href="${j.video_url}" target="_blank">${j.id}</a> · ${j.params.width}×${j.params.height} · ${j.params.video_length} frames · ${j.params.steps} steps</div></div>`).join('');
  document.getElementById('gallery').innerHTML = outs || '<div class="small">No generated videos yet.</div>';
}
async function loadJob(id){
  const j=await api('/api/jobs/'+id); currentJob=id;
  document.getElementById('active').innerHTML = `<div><b>${j.id}</b> <span class="${clsStatus(j.status)}">${j.status}</span></div><div class="small">${j.created_at||''} → ${j.finished_at||'running'}</div><div>${j.params.prompt}</div>${j.video_url?`<p><a href="${j.video_url}" target="_blank">Open output video</a></p><video src="${j.video_url}" controls muted loop></video>`:''}`;
  document.getElementById('log').textContent = j.log_tail || '';
}
document.getElementById('genForm').addEventListener('submit', async (ev)=>{
  ev.preventDefault();
  const btn=document.getElementById('submitBtn'); btn.disabled=true; btn.textContent='Submitting…';
  try{
    const fd=new FormData(ev.target);
    if(!fd.get('auto_manage_services')) fd.set('auto_manage_services','false'); else fd.set('auto_manage_services','true');
    if(!fd.get('restore_services')) fd.set('restore_services','false'); else fd.set('restore_services','true');
    const j=await api('/api/generate',{method:'POST',body:fd}); currentJob=j.id; await refresh();
  }catch(e){alert(String(e));}
  finally{btn.disabled=false; btn.textContent='Start DomainShuttle job';}
});
async function cancelActive(){ if(!currentJob) return; if(!confirm('Cancel active job?')) return; await api('/api/jobs/'+currentJob+'/cancel',{method:'POST'}); await refresh(); }
setInterval(refresh, 5000); refresh();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "domainshuttle-web", "version": APP_VERSION, "timestamp": time.time()}


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    with _state_lock:
        running = _running.get("job_id")
        proc = _running.get("proc")
        if running and proc is not None and proc.poll() is not None:
            running = None
    return {"ok": True, "running_job": running, "jobs": list_jobs(50), "version": APP_VERSION}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict[str, Any]:
    job = load_job(job_id)
    job["log_tail"] = tail(job.get("log_path", ""), 24000)
    return job


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str) -> dict[str, Any]:
    with _state_lock:
        if _running.get("job_id") != job_id or _running.get("proc") is None:
            raise HTTPException(status_code=409, detail="job is not the active running job")
        proc: subprocess.Popen[str] = _running["proc"]
    job = load_job(job_id)
    job["status"] = "cancelled"
    job["finished_at"] = now_iso()
    save_job(job)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    return {"ok": True, "id": job_id, "status": "cancelled"}


@app.post("/api/generate")
def api_generate(
    prompt: str = Form(...),
    width: int = Form(448),
    height: int = Form(256),
    video_length: int = Form(17),
    fps: int = Form(8),
    steps: int = Form(4),
    seed: int = Form(42),
    shift: float = Form(5.0),
    guidance_a: float = Form(4.0),
    guidance_b: float = Form(3.0),
    domain: str = Form("object"),
    auto_manage_services: str = Form("true"),
    restore_services: str = Form("true"),
    reference: UploadFile | None = File(None),
) -> JSONResponse:
    with _state_lock:
        proc = _running.get("proc")
        if proc is not None and proc.poll() is None:
            raise HTTPException(status_code=409, detail=f"job already running: {_running.get('job_id')}")

    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job_dir = JOB_ROOT / job_id
    job_input_dir = INPUT_ROOT / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)
    output_dir = OUTPUT_ROOT / f"web_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "run.log"
    job_dir.mkdir(parents=True, exist_ok=True)

    if reference is not None and reference.filename:
        suffix = Path(reference.filename).suffix.lower() or ".png"
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            suffix = ".png"
        ref_path = job_input_dir / ("reference" + suffix)
        with ref_path.open("wb") as f:
            shutil.copyfileobj(reference.file, f)
    else:
        if not SAMPLE_REFERENCE.exists():
            raise HTTPException(status_code=400, detail="no reference uploaded and sample reference missing")
        ref_path = job_input_dir / SAMPLE_REFERENCE.name
        shutil.copy2(SAMPLE_REFERENCE, ref_path)

    domain = domain if domain in {"Human", "Man", "Woman", "object", "Object", "Fantasy Domain", "Background", "others"} else "object"
    params = {
        "prompt": prompt,
        "width": safe_int(width, 448, 128, 832),
        "height": safe_int(height, 256, 128, 480),
        "video_length": safe_int(video_length, 17, 5, 81),
        "fps": safe_int(fps, 8, 1, 24),
        "steps": safe_int(steps, 4, 1, 40),
        "seed": safe_int(seed, 42, 0, 2_147_483_647),
        "shift": safe_float(shift, 5.0, 0.1, 20.0),
        "guidance_a": safe_float(guidance_a, 4.0, 0.1, 20.0),
        "guidance_b": safe_float(guidance_b, 3.0, 0.1, 20.0),
        "domain": domain,
        "auto_manage_services": str(auto_manage_services).lower() == "true",
        "restore_services": str(restore_services).lower() == "true",
        "memory_mode": "model_full_load",
        "attention": "SDPA",
    }
    input_json = job_input_dir / "input.jsonl"
    item = {"image_path": str(ref_path), "domain_code": [domain], "prompt": prompt}
    input_json.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")

    job = {
        "id": job_id,
        "created_at": now_iso(),
        "status": "queued",
        "params": params,
        "reference_path": str(ref_path),
        "reference_url": "/inputs/" + str(ref_path.relative_to(ROOT / "inputs")),
        "input_json": str(input_json),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "outputs": [],
    }
    save_job(job)
    t = threading.Thread(target=run_job_thread, args=(job_id,), daemon=True)
    t.start()
    return JSONResponse(job)


@app.get("/api/logs/{job_id}", response_class=PlainTextResponse)
def api_logs(job_id: str) -> str:
    job = load_job(job_id)
    return tail(job.get("log_path", ""), 64000)
