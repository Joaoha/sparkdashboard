#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from diffusers import QwenImagePipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("QWEN_IMAGE_ROOT", "/opt/qwen-image"))
MODEL_ID = os.environ.get("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image")
HF_HOME = Path(os.environ.get("HF_HOME", str(ROOT / "hf-cache")))
OUTPUTS_DIR = ROOT / "outputs"
MANIFEST_PATH = OUTPUTS_DIR / "manifest.json"
HOST = os.environ.get("QWEN_IMAGE_HOST", "0.0.0.0")
PORT = int(os.environ.get("QWEN_IMAGE_PORT", "7865"))
CPU_OFFLOAD = os.environ.get("QWEN_IMAGE_CPU_OFFLOAD", "0") == "1"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Qwen-Image on Spark")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

_pipe: QwenImagePipeline | None = None
_pipe_lock = threading.Lock()
_manifest_lock = threading.Lock()
_last_error: str | None = None
_last_generate_started: float | None = None
_last_generate_finished: float | None = None

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
    "512 smoke": (512, 512),
    "1024 square": (1024, 1024),
}


def _load_manifest() -> list[dict[str, Any]]:
    try:
        data = json.loads(MANIFEST_PATH.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_manifest(items: list[dict[str, Any]]) -> None:
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, indent=2, sort_keys=True))
    os.replace(tmp, MANIFEST_PATH)


def _safe_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]


def _validate_id(image_id: str) -> None:
    if not re.fullmatch(r"[0-9]{8}_[0-9]{6}_[0-9a-f]{8}", image_id or ""):
        raise HTTPException(status_code=400, detail="Invalid image id")


def _device_summary() -> dict[str, Any]:
    out: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "model_id": MODEL_ID,
        "hf_home": str(HF_HOME),
        "cpu_offload": CPU_OFFLOAD,
    }
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
        try:
            free, total = torch.cuda.mem_get_info()
            out["cuda_free_gib"] = free / 1024**3
            out["cuda_total_gib"] = total / 1024**3
        except Exception:
            pass
    return out


def get_pipe() -> QwenImagePipeline:
    global _pipe
    if _pipe is not None:
        return _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this Spark process")
        pipe = QwenImagePipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=str(HF_HOME),
        )
        if CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        _pipe = pipe
        return _pipe


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    negative_prompt: str = Field(" ", max_length=4000)
    width: int = Field(1024, ge=512, le=2048)
    height: int = Field(1024, ge=512, le=2048)
    steps: int = Field(20, ge=1, le=80)
    true_cfg_scale: float = Field(4.0, ge=0.0, le=12.0)
    seed: int = Field(42, ge=0, le=2_147_483_647)
    append_quality_suffix: bool = True


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "qwen-image",
        "model_loaded": _pipe is not None,
        "last_error": _last_error,
        "last_generate_started": _last_generate_started,
        "last_generate_finished": _last_generate_finished,
        **_device_summary(),
    }


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    with _manifest_lock:
        count = len(_load_manifest())
    return {"ok": True, "image_count": count, "aspect_ratios": ASPECT_RATIOS, "model_loaded": _pipe is not None, **_device_summary()}



@app.post("/api/unload")
def unload_model() -> dict[str, Any]:
    """Unload the resident Diffusers pipeline while keeping the web service and image library online."""
    global _pipe
    unloaded = False
    with _pipe_lock:
        if _pipe is not None:
            pipe = _pipe
            _pipe = None
            del pipe
            unloaded = True
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    return {"ok": True, "unloaded": unloaded, "model_loaded": _pipe is not None, **_device_summary()}

@app.get("/api/images")
def list_images() -> dict[str, Any]:
    with _manifest_lock:
        return {"images": _load_manifest()}


@app.delete("/api/images/{image_id}")
def delete_image(image_id: str) -> dict[str, Any]:
    _validate_id(image_id)
    image_dir = (OUTPUTS_DIR / image_id).resolve()
    if not str(image_dir).startswith(str(OUTPUTS_DIR.resolve()) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid image path")
    removed_files = 0
    with _manifest_lock:
        items = _load_manifest()
        before = len(items)
        items = [item for item in items if item.get("id") != image_id]
        _write_manifest(items)
    if image_dir.exists():
        for _root, _dirs, files in os.walk(image_dir):
            removed_files += len(files)
        shutil.rmtree(image_dir)
    return {"ok": True, "id": image_id, "manifest_removed": before != len(items), "removed_files": removed_files}


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    global _last_error, _last_generate_started, _last_generate_finished
    _last_error = None
    _last_generate_started = time.time()
    _last_generate_finished = None
    t0 = time.time()
    try:
        pipe = get_pipe()
        prompt = req.prompt.strip()
        if req.append_quality_suffix:
            prompt = prompt + ", Ultra HD, 4K, cinematic composition."
        generator = torch.Generator(device="cuda").manual_seed(req.seed)
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=req.negative_prompt or " ",
                width=req.width,
                height=req.height,
                num_inference_steps=req.steps,
                true_cfg_scale=req.true_cfg_scale,
                generator=generator,
            )
        image = result.images[0]
        image_id = _safe_id()
        image_dir = OUTPUTS_DIR / image_id
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / "image.png"
        meta_path = image_dir / "metadata.json"
        image.save(image_path)
        duration = time.time() - t0
        item = {
            "id": image_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_id": MODEL_ID,
            "image": {"local_path": str(image_path), "static_url": f"/outputs/{image_id}/image.png"},
            "metadata": {"local_path": str(meta_path), "static_url": f"/outputs/{image_id}/metadata.json"},
            "prompt": req.prompt,
            "effective_prompt": prompt,
            "negative_prompt": req.negative_prompt,
            "settings": {
                "width": req.width,
                "height": req.height,
                "steps": req.steps,
                "true_cfg_scale": req.true_cfg_scale,
                "seed": req.seed,
                "dtype": "bfloat16",
                "append_quality_suffix": req.append_quality_suffix,
                "cpu_offload": CPU_OFFLOAD,
            },
            "duration_sec": duration,
        }
        meta_path.write_text(json.dumps(item, indent=2, sort_keys=True))
        with _manifest_lock:
            items = _load_manifest()
            items.insert(0, item)
            _write_manifest(items)
        _last_generate_finished = time.time()
        return {"ok": True, "image": item}
    except Exception as exc:
        _last_error = f"{type(exc).__name__}: {exc}"
        raise HTTPException(status_code=500, detail=_last_error)


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Qwen-Image on Spark</title>
<style>
:root{--bg:#0a0807;--text:#f4f2fb;--muted:#9a98a6;--dim:#686672;--accent:#5FE3A0;--warn:#FFB35C;--danger:#ff6f7d;--border:rgba(255,255,255,.09);--radius:18px}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% -10%,rgba(95,227,160,.14),transparent 34%),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:100vh}.shell{display:grid;grid-template-columns:380px minmax(0,1fr);min-height:100vh}.side{border-right:1px solid var(--border);background:rgba(10,8,7,.86);padding:26px;position:sticky;top:0;height:100vh;overflow:auto}.main{padding:28px;max-width:1600px;width:100%;margin:0 auto}.brand{display:flex;gap:12px;align-items:center;margin-bottom:24px}.spark{width:14px;height:14px;border:2px solid var(--accent);transform:rotate(45deg);box-shadow:0 0 22px rgba(95,227,160,.5)}h1{font-size:24px;margin:0;letter-spacing:-.03em}.sub{color:var(--muted);font-size:13px;line-height:1.55}.card{border:1px solid var(--border);border-radius:var(--radius);background:rgba(255,255,255,.025);padding:18px;margin-bottom:16px;box-shadow:0 26px 80px rgba(0,0,0,.25)}label{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em;margin:14px 0 7px}textarea,input,select{width:100%;border:1px solid var(--border);background:#0d0c0b;color:var(--text);border-radius:12px;padding:11px 12px;font:inherit}textarea{min-height:150px;resize:vertical}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.btn{width:100%;border:0;border-radius:12px;padding:14px 16px;background:var(--accent);color:#06100b;font-weight:800;letter-spacing:.04em;cursor:pointer}.btn.secondary{background:rgba(95,227,160,.08);color:var(--accent);border:1px solid rgba(95,227,160,.25)}.btn:disabled{opacity:.45;cursor:not-allowed}.status{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);font-size:12px;white-space:pre-wrap}.hero{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:18px;align-items:start}.preview{min-height:620px;display:grid;place-items:center;border:1px solid var(--border);border-radius:var(--radius);background:linear-gradient(135deg,rgba(255,255,255,.03),rgba(255,255,255,.01));overflow:hidden}.preview img{max-width:100%;max-height:84vh;border-radius:12px}.library{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}.thumb{border:1px solid var(--border);border-radius:16px;background:rgba(255,255,255,.025);overflow:hidden}.thumb img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;background:#111}.thumb .body{padding:12px}.meta{font-size:11px;color:var(--dim);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;line-height:1.5}.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.pill{font-size:11px;color:var(--accent);border:1px solid rgba(95,227,160,.25);padding:6px 8px;border-radius:999px;text-decoration:none;background:rgba(95,227,160,.06);cursor:pointer}.pill.danger{color:var(--danger);border-color:rgba(255,111,125,.25);background:rgba(255,111,125,.06)}.loading{position:fixed;inset:0;background:rgba(10,8,7,.78);display:none;place-items:center;z-index:5;backdrop-filter:blur(10px)}.spinner{width:58px;height:58px;border-radius:50%;border:3px solid rgba(255,255,255,.12);border-top-color:var(--accent);animation:spin 1s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:900px){.shell{grid-template-columns:1fr}.side{position:relative;height:auto}.hero{grid-template-columns:1fr}}
</style></head><body>
<div class="loading" id="loading"><div><div class="spinner"></div><p class="status" id="loading-text" style="text-align:center;margin-top:16px">Generating...</p></div></div>
<div class="shell"><aside class="side"><div class="brand"><div class="spark"></div><div><h1>Qwen-Image</h1><div class="sub">Qwen 20B text-to-image on Spark</div></div></div>
<div class="card"><label>Prompt</label><textarea id="prompt">A cozy laneway cafe sign reading "Spark Coffee" with small Chinese text "通义千问", rainy night, cinematic lighting</textarea><label>Negative prompt</label><textarea id="negative" style="min-height:72px"> </textarea><label>Aspect ratio preset</label><select id="preset" onchange="applyPreset()"><option value="1024 square">1024 square</option><option value="512 smoke">512 smoke</option><option value="1:1">1:1 official 1328×1328</option><option value="16:9">16:9 official 1664×928</option><option value="9:16">9:16 official 928×1664</option><option value="4:3">4:3 official 1472×1140</option><option value="3:4">3:4 official 1140×1472</option><option value="3:2">3:2 official 1584×1056</option><option value="2:3">2:3 official 1056×1584</option></select><div class="row"><div><label>Width</label><input id="width" type="number" value="1024"></div><div><label>Height</label><input id="height" type="number" value="1024"></div></div><div class="row"><div><label>Steps</label><input id="steps" type="number" min="1" max="80" value="20"></div><div><label>True CFG</label><input id="cfg" type="number" min="0" max="12" step="0.1" value="4.0"></div></div><label>Seed</label><div class="row"><input id="seed" type="number" min="0" max="2147483647" value="42"><button class="btn secondary" onclick="randomSeed()">Random</button></div><label><input id="suffix" type="checkbox" checked style="width:auto"> Append official quality suffix</label><div style="height:14px"></div><button class="btn" id="generate" onclick="generate()">Generate</button></div><div class="card"><div class="status" id="status">Loading status...</div></div></aside>
<main class="main"><section class="hero"><div class="preview" id="preview"><div class="sub">Generated image will appear here and be saved with metadata/provenance.</div></div><div class="card"><h2 style="margin-top:0">Run notes</h2><div class="sub">Qwen-Image is a 20B heavyweight model. Stop Qwen vLLM, Z-Image, HiDream, and preferably Pixal3D for high-res runs. First test with 512 smoke or 1024 square.</div><div style="height:14px"></div><button class="btn secondary" onclick="refreshImages()">Refresh Library</button></div></section><h2>Saved Image Library</h2><div class="library" id="library"></div></main></div>
<script>
const presets={"1:1":[1328,1328],"16:9":[1664,928],"9:16":[928,1664],"4:3":[1472,1140],"3:4":[1140,1472],"3:2":[1584,1056],"2:3":[1056,1584],"512 smoke":[512,512],"1024 square":[1024,1024]};
function $(id){return document.getElementById(id)} function setStatus(msg){$('status').textContent=msg} function applyPreset(){const p=presets[$('preset').value]; if(p){$('width').value=p[0]; $('height').value=p[1];}} function randomSeed(){ $('seed').value=Math.floor(Math.random()*2147483647) } function showLoading(msg){$('loading-text').textContent=msg||'Generating...'; $('loading').style.display='grid'} function hideLoading(){ $('loading').style.display='none' }
async function status(){ try{ const r=await fetch('/api/status',{cache:'no-store'}); const j=await r.json(); setStatus(`model: ${j.model_id}\nloaded: ${j.model_loaded}\nimages: ${j.image_count}\ndevice: ${j.device||'--'}\ncuda free: ${j.cuda_free_gib?j.cuda_free_gib.toFixed(1)+' GiB':'--'}\ncpu offload: ${j.cpu_offload}`)}catch(e){setStatus('status failed: '+e.message)} }
async function generate(){ const body={prompt:$('prompt').value, negative_prompt:$('negative').value, width:+$('width').value, height:+$('height').value, steps:+$('steps').value, true_cfg_scale:+$('cfg').value, seed:+$('seed').value, append_quality_suffix:$('suffix').checked}; $('generate').disabled=true; showLoading('Loading Qwen-Image / generating. First run can take several minutes.'); try{ const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const j=await r.json(); if(!r.ok||!j.ok) throw new Error(j.detail||j.error||`HTTP ${r.status}`); renderPreview(j.image); await refreshImages(); setStatus(`generated ${j.image.id}\nduration: ${j.image.duration_sec.toFixed(1)}s\nseed: ${j.image.settings.seed}`); }catch(e){console.error(e); alert('Generation failed: '+e.message); setStatus('generation failed: '+e.message);} finally{hideLoading(); $('generate').disabled=false; status();}}
function renderPreview(item){ $('preview').innerHTML=`<img src="${item.image.static_url}?t=${Date.now()}" alt="generated">`; }
function card(item){ const s=item.settings||{}; return `<div class="thumb"><img src="${item.image.static_url}?t=${Date.now()}" onclick="renderPreview(window.images.find(x=>x.id==='${item.id}'))"><div class="body"><div class="meta">${item.created_at}<br>${s.width}×${s.height} · ${s.steps} steps · CFG ${s.true_cfg_scale}<br>seed ${s.seed}<br>${(item.prompt||'').slice(0,130)}</div><div class="actions"><a class="pill" href="${item.image.static_url}" target="_blank">Image</a><a class="pill" href="${item.metadata.static_url}" target="_blank">Metadata</a><button class="pill danger" onclick="deleteImage('${item.id}')">Remove</button></div></div></div>`}
async function refreshImages(){ const r=await fetch('/api/images',{cache:'no-store'}); const j=await r.json(); window.images=j.images||[]; $('library').innerHTML=window.images.length?window.images.map(card).join(''):'<div class="sub">No saved generations yet.</div>'; if(window.images.length && !$('preview').querySelector('img')) renderPreview(window.images[0]);}
async function deleteImage(id){ if(!confirm('Remove saved generated image '+id+'?')) return; const r=await fetch('/api/images/'+encodeURIComponent(id),{method:'DELETE'}); if(!r.ok) alert(await r.text()); await refreshImages(); status(); }
applyPreset(); status(); refreshImages(); setInterval(status,10000);
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
