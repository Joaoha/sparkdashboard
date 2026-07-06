#!/usr/bin/env python3
from __future__ import annotations

import base64
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
from diffusers import ZImagePipeline
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("Z_IMAGE_ROOT", "/opt/z-image"))
MODEL_ID = os.environ.get("Z_IMAGE_MODEL", "Tongyi-MAI/Z-Image")
HF_HOME = Path(os.environ.get("HF_HOME", str(ROOT / "hf-cache")))
OUTPUTS_DIR = ROOT / "outputs"
MANIFEST_PATH = OUTPUTS_DIR / "manifest.json"
FINETUNE_DIR = ROOT / "finetune" / "datasets" / "default"
FINETUNE_IMAGES_DIR = FINETUNE_DIR / "images"
FINETUNE_MANIFEST = FINETUNE_DIR / "manifest.jsonl"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FINETUNE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME.mkdir(parents=True, exist_ok=True)

HOST = os.environ.get("Z_IMAGE_HOST", "0.0.0.0")
PORT = int(os.environ.get("Z_IMAGE_PORT", "7864"))

app = FastAPI(title="Z-Image on Spark")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

_pipe: ZImagePipeline | None = None
_pipe_lock = threading.Lock()
_manifest_lock = threading.Lock()
_last_error: str | None = None
_last_generate_started: float | None = None
_last_generate_finished: float | None = None


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


def _find_image_item(image_id: str) -> dict[str, Any]:
    _validate_id(image_id)
    with _manifest_lock:
        for item in _load_manifest():
            if item.get("id") == image_id:
                return item
    meta_path = OUTPUTS_DIR / image_id / "metadata.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="Image not found")


def _load_finetune_records() -> list[dict[str, Any]]:
    if not FINETUNE_MANIFEST.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in FINETUNE_MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                records.append(rec)
        except Exception:
            continue
    return records


def _write_finetune_records(records: list[dict[str, Any]]) -> None:
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = FINETUNE_MANIFEST.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    os.replace(tmp, FINETUNE_MANIFEST)


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


def get_pipe() -> ZImagePipeline:
    global _pipe
    if _pipe is not None:
        return _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this Spark process")
        pipe = ZImagePipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=str(HF_HOME),
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")
        # Avoid safety checker assumptions; ZImagePipeline does not need custom ARM kernels.
        _pipe = pipe
        return _pipe


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str = Field("", max_length=4000)
    width: int = Field(1024, ge=512, le=2048)
    height: int = Field(1024, ge=512, le=2048)
    steps: int = Field(28, ge=1, le=80)
    guidance_scale: float = Field(4.0, ge=0.0, le=12.0)
    seed: int = Field(42, ge=0, le=2_147_483_647)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "z-image",
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
    finetune_count = len(_load_finetune_records())
    return {"ok": True, "image_count": count, "finetune_count": finetune_count, "finetune_dir": str(FINETUNE_DIR), "model_loaded": _pipe is not None, **_device_summary()}



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


@app.get("/api/images/{image_id}/metadata")
def image_metadata(image_id: str) -> dict[str, Any]:
    return {"ok": True, "image": _find_image_item(image_id)}


@app.get("/api/finetune/dataset")
def finetune_dataset() -> dict[str, Any]:
    records = _load_finetune_records()
    return {
        "ok": True,
        "count": len(records),
        "dataset_dir": str(FINETUNE_DIR),
        "images_dir": str(FINETUNE_IMAGES_DIR),
        "manifest": str(FINETUNE_MANIFEST),
        "records": records,
    }


@app.post("/api/finetune/items/{image_id}")
def add_finetune_item(image_id: str) -> dict[str, Any]:
    item = _find_image_item(image_id)
    image_path = Path(item.get("image", {}).get("local_path") or OUTPUTS_DIR / image_id / "image.png")
    meta_path = Path(item.get("metadata", {}).get("local_path") or OUTPUTS_DIR / image_id / "metadata.json")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Source image file not found")
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    dst_image = FINETUNE_IMAGES_DIR / f"{image_id}.png"
    dst_meta = FINETUNE_DIR / f"{image_id}.metadata.json"
    shutil.copyfile(image_path, dst_image)
    if meta_path.exists():
        shutil.copyfile(meta_path, dst_meta)
    else:
        dst_meta.write_text(json.dumps(item, indent=2, sort_keys=True))
    rec = {
        "id": image_id,
        "image": str(dst_image),
        "metadata": str(dst_meta),
        "caption": item.get("prompt", ""),
        "negative_prompt": item.get("negative_prompt", ""),
        "settings": item.get("settings", {}),
        "source_image": item.get("image", {}).get("local_path"),
        "source_metadata": item.get("metadata", {}).get("local_path"),
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    records = [r for r in _load_finetune_records() if r.get("id") != image_id]
    records.insert(0, rec)
    _write_finetune_records(records)
    return {"ok": True, "record": rec, "count": len(records)}


@app.delete("/api/finetune/items/{image_id}")
def remove_finetune_item(image_id: str) -> dict[str, Any]:
    _validate_id(image_id)
    records = _load_finetune_records()
    kept = [r for r in records if r.get("id") != image_id]
    _write_finetune_records(kept)
    removed_files = 0
    for path in [FINETUNE_IMAGES_DIR / f"{image_id}.png", FINETUNE_DIR / f"{image_id}.metadata.json"]:
        try:
            if path.exists():
                path.unlink()
                removed_files += 1
        except OSError:
            pass
    return {"ok": True, "id": image_id, "removed": len(records) - len(kept), "removed_files": removed_files, "count": len(kept)}


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
        generator = torch.Generator(device="cuda").manual_seed(req.seed)
        with torch.inference_mode():
            result = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt or None,
                width=req.width,
                height=req.height,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
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
            "negative_prompt": req.negative_prompt,
            "settings": {
                "width": req.width,
                "height": req.height,
                "steps": req.steps,
                "guidance_scale": req.guidance_scale,
                "seed": req.seed,
                "dtype": "bfloat16",
            },
            "duration_sec": duration,
            "finetune_staged": False,
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
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Z-Image on Spark</title>
  <style>
    :root{--bg:#0a0807;--panel:#11100f;--panel2:#151413;--text:#f4f2fb;--muted:#9a98a6;--dim:#686672;--accent:#5FE3A0;--warn:#FFB35C;--danger:#ff6f7d;--border:rgba(255,255,255,.09);--radius:18px}
    *{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at 50% -10%,rgba(95,227,160,.14),transparent 34%),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:100vh}
    .shell{display:grid;grid-template-columns:360px minmax(0,1fr);min-height:100vh}.side{border-right:1px solid var(--border);background:rgba(10,8,7,.86);padding:26px;position:sticky;top:0;height:100vh;overflow:auto}.main{padding:28px;max-width:1500px;width:100%;margin:0 auto}.brand{display:flex;gap:12px;align-items:center;margin-bottom:24px}.spark{width:14px;height:14px;border:2px solid var(--accent);transform:rotate(45deg);box-shadow:0 0 22px rgba(95,227,160,.5)}h1{font-size:24px;margin:0;letter-spacing:-.03em}.sub{color:var(--muted);font-size:13px;line-height:1.55}.card{border:1px solid var(--border);border-radius:var(--radius);background:rgba(255,255,255,.025);padding:18px;margin-bottom:16px;box-shadow:0 26px 80px rgba(0,0,0,.25)}label{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em;margin:14px 0 7px}textarea,input,select{width:100%;border:1px solid var(--border);background:#0d0c0b;color:var(--text);border-radius:12px;padding:11px 12px;font:inherit}textarea{min-height:160px;resize:vertical}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.btn{width:100%;border:0;border-radius:12px;padding:14px 16px;background:var(--accent);color:#06100b;font-weight:800;letter-spacing:.04em;cursor:pointer}.btn.secondary{background:rgba(95,227,160,.08);color:var(--accent);border:1px solid rgba(95,227,160,.25)}.btn.danger{background:rgba(255,111,125,.08);color:var(--danger);border:1px solid rgba(255,111,125,.28)}.btn:disabled{opacity:.45;cursor:not-allowed}.status{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);font-size:12px;white-space:pre-wrap}.hero{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:18px;align-items:start}.preview{min-height:560px;display:grid;place-items:center;border:1px solid var(--border);border-radius:var(--radius);background:linear-gradient(135deg,rgba(255,255,255,.03),rgba(255,255,255,.01));overflow:hidden}.preview img{max-width:100%;max-height:82vh;border-radius:12px}.library{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}.thumb{border:1px solid var(--border);border-radius:16px;background:rgba(255,255,255,.025);overflow:hidden}.thumb img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;background:#111}.thumb .body{padding:12px}.meta{font-size:11px;color:var(--dim);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;line-height:1.5}.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.pill{font-size:11px;color:var(--accent);border:1px solid rgba(95,227,160,.25);padding:6px 8px;border-radius:999px;text-decoration:none;background:rgba(95,227,160,.06);cursor:pointer}.pill.warn{color:var(--warn);border-color:rgba(255,179,92,.28);background:rgba(255,179,92,.07)}.meta-panel{margin-top:12px;padding:12px;border:1px solid var(--border);border-radius:12px;background:rgba(0,0,0,.18)}.meta-panel pre{white-space:pre-wrap;word-break:break-word;font-size:11px;color:var(--muted);margin:8px 0 0}.thumb img{cursor:pointer}.pill.danger{color:var(--danger);border-color:rgba(255,111,125,.25);background:rgba(255,111,125,.06)}.loading{position:fixed;inset:0;background:rgba(10,8,7,.78);display:none;place-items:center;z-index:5;backdrop-filter:blur(10px)}.spinner{width:58px;height:58px;border-radius:50%;border:3px solid rgba(255,255,255,.12);border-top-color:var(--accent);animation:spin 1s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:900px){.shell{grid-template-columns:1fr}.side{position:relative;height:auto}.hero{grid-template-columns:1fr}}
  </style>
</head>
<body>
<div class="loading" id="loading"><div><div class="spinner"></div><p class="status" id="loading-text" style="text-align:center;margin-top:16px">Generating...</p></div></div>
<div class="shell">
  <aside class="side">
    <div class="brand"><div class="spark"></div><div><h1>Z-Image</h1><div class="sub">Tongyi-MAI foundation text-to-image on Spark</div></div></div>
    <div class="card">
      <label>Prompt</label><textarea id="prompt" placeholder="Describe the image...">cinematic photo of a small glass greenhouse in a rainy eucalyptus forest, warm interior lights, mist, highly detailed, natural colors</textarea>
      <label>Negative prompt</label><textarea id="negative" style="min-height:74px" placeholder="Optional">blurry, low quality, distorted, extra fingers, watermark, text artifacts</textarea>
      <div class="row"><div><label>Width</label><select id="width"><option>512</option><option selected>1024</option><option>1536</option><option>2048</option></select></div><div><label>Height</label><select id="height"><option>512</option><option selected>1024</option><option>1536</option><option>2048</option></select></div></div>
      <div class="row"><div><label>Steps</label><input id="steps" type="number" min="1" max="80" value="28"></div><div><label>CFG</label><input id="cfg" type="number" min="0" max="12" step="0.1" value="4.0"></div></div>
      <label>Seed</label><div class="row"><input id="seed" type="number" min="0" max="2147483647" value="42"><button class="btn secondary" onclick="randomSeed()">Random</button></div>
      <div style="height:14px"></div><button class="btn" id="generate" onclick="generate()">Generate</button>
    </div>
    <div class="card"><div class="status" id="status">Loading status...</div></div>
  </aside>
  <main class="main">
    <section class="hero">
      <div class="preview" id="preview"><div class="sub">Generated image will appear here and be saved to the local library automatically.</div></div>
      <div class="card"><h2 style="margin-top:0">Regenerate / Finetune</h2><div class="sub">Click a saved thumbnail to inspect it inline. Use <b>Load Settings</b> to restore prompt, negative prompt, size, steps, CFG, and seed into the form for exact regeneration. Use <b>Add to Finetune Set</b> to stage the image + metadata into a dataset manifest.</div><div style="height:14px"></div><button class="btn secondary" onclick="refreshImages()">Refresh Library</button><div style="height:10px"></div><button class="btn secondary" onclick="refreshFinetune()">Refresh Finetune Set</button><div class="status" id="finetune-status" style="margin-top:12px">finetune set: loading...</div></div>
    </section>
    <h2>Saved Image Library</h2>
    <div class="library" id="library"></div>
  </main>
</div>
<script>
function $(id){return document.getElementById(id)}
function setStatus(msg){$('status').textContent=msg}
function randomSeed(){ $('seed').value=Math.floor(Math.random()*2147483647) }
function showLoading(msg){$('loading-text').textContent=msg||'Generating...'; $('loading').style.display='grid'}
function hideLoading(){ $('loading').style.display='none' }
async function status(){ try{ const r=await fetch('/api/status',{cache:'no-store'}); const j=await r.json(); setStatus(`model: ${j.model_id}
loaded: ${j.model_loaded}
images: ${j.image_count}
finetune items: ${j.finetune_count||0}
device: ${j.device||'--'}
cuda free: ${j.cuda_free_gib?j.cuda_free_gib.toFixed(1)+' GiB':'--'}`)}catch(e){setStatus('status failed: '+e.message)} }
function formBody(){return {prompt:$('prompt').value, negative_prompt:$('negative').value, width:+$('width').value, height:+$('height').value, steps:+$('steps').value, guidance_scale:+$('cfg').value, seed:+$('seed').value}}
async function generate(){
  const body=formBody();
  $('generate').disabled=true; showLoading('Loading model / generating. First run downloads and initializes weights.');
  try{ const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const j=await r.json(); if(!r.ok||!j.ok) throw new Error(j.detail||j.error||`HTTP ${r.status}`); renderPreview(j.image); await refreshImages(); setStatus(`generated ${j.image.id}
duration: ${j.image.duration_sec.toFixed(1)}s
seed: ${j.image.settings.seed}`); }
  catch(e){ console.error(e); alert('Generation failed: '+e.message); setStatus('generation failed: '+e.message); }
  finally{ hideLoading(); $('generate').disabled=false; status(); }
}
function normalizedPrompts(item){
  let prompt = item.prompt || '';
  let negative = item.negative_prompt || '';
  const marker = /\n\s*Negative prompt:\s*\n/i;
  const m = prompt.match(marker);
  if (m) {
    const before = prompt.slice(0, m.index).trimEnd();
    const after = prompt.slice(m.index + m[0].length).trim();
    prompt = before;
    // Legacy Z-Image records sometimes stored the real negative prompt inside
    // prompt while the separate negative_prompt field held the UI default.
    // Prefer the embedded value when present because it is the actual provenance.
    if (after) negative = after;
  }
  return {prompt, negative};
}
function fillFormFromItem(item, opts={}){
  const s=item.settings||{};
  const pn=normalizedPrompts(item);
  $('prompt').value=pn.prompt;
  $('negative').value=pn.negative;
  $('negative').dispatchEvent(new Event('input', {bubbles:true}));
  $('negative').dispatchEvent(new Event('change', {bubbles:true}));
  $('prompt').dispatchEvent(new Event('input', {bubbles:true}));
  $('prompt').dispatchEvent(new Event('change', {bubbles:true}));
  $('width').value=s.width||1024;
  $('height').value=s.height||1024;
  $('steps').value=s.steps||28;
  $('cfg').value=s.guidance_scale ?? 4.0;
  $('seed').value=opts.randomSeed?Math.floor(Math.random()*2147483647):(s.seed ?? 42);
}

async function loadSettings(id){ const item=window.images.find(x=>x.id===id) || (await (await fetch('/api/images/'+encodeURIComponent(id)+'/metadata')).json()).image; fillFormFromItem(item); renderPreview(item); window.scrollTo({top:0, behavior:'smooth'}); setStatus(`loaded settings from ${id}
seed: ${(item.settings||{}).seed}`); }
async function regenerateFrom(id, randomize=false){ const item=window.images.find(x=>x.id===id) || (await (await fetch('/api/images/'+encodeURIComponent(id)+'/metadata')).json()).image; fillFormFromItem(item,{randomSeed:randomize}); await generate(); }
function renderPreview(item){ const s=item.settings||{}; const pn=normalizedPrompts(item); $('preview').innerHTML=`<div style="width:100%"><img src="${item.image.static_url}?t=${Date.now()}" alt="generated"><div class="meta-panel"><b>${item.id}</b><pre>${escapeHtml(JSON.stringify({prompt:pn.prompt, negative_prompt:pn.negative, settings:s, duration_sec:item.duration_sec, model_id:item.model_id}, null, 2))}</pre><div class="actions"><button class="pill" onclick="loadSettings('${item.id}')">Load Settings</button><button class="pill" onclick="regenerateFrom('${item.id}', false)">Regenerate</button><button class="pill warn" onclick="regenerateFrom('${item.id}', true)">Variant Seed</button><button class="pill" onclick="addFinetune('${item.id}')">Add to Finetune Set</button><a class="pill" href="${item.image.static_url}" target="_blank">Open Image</a><a class="pill" href="${item.metadata.static_url}" target="_blank">Metadata</a></div></div></div>`; }
function card(item){ const s=item.settings||{}; return `<div class="thumb"><img src="${item.image.static_url}?t=${Date.now()}" onclick="renderPreview(window.images.find(x=>x.id==='${item.id}'))"><div class="body"><div class="meta">${item.created_at}<br>${s.width}×${s.height} · ${s.steps} steps · CFG ${s.guidance_scale}<br>seed ${s.seed}<br>${escapeHtml((item.prompt||'').slice(0,120))}</div><div class="actions"><button class="pill" onclick="loadSettings('${item.id}')">Load</button><button class="pill" onclick="regenerateFrom('${item.id}', false)">Regen</button><button class="pill warn" onclick="addFinetune('${item.id}')">Finetune</button><a class="pill" href="${item.image.static_url}" target="_blank">Image</a><a class="pill" href="${item.metadata.static_url}" target="_blank">Metadata</a><button class="pill danger" onclick="deleteImage('${item.id}')">Remove</button></div></div></div>` }
async function refreshImages(){ const r=await fetch('/api/images',{cache:'no-store'}); const j=await r.json(); window.images=j.images||[]; $('library').innerHTML=window.images.length?window.images.map(card).join(''):'<div class="sub">No saved generations yet.</div>'; if(window.images.length && !$('preview').querySelector('img')) renderPreview(window.images[0]); }
async function refreshFinetune(){
  try{
    const r=await fetch('/api/finetune/dataset',{cache:'no-store'});
    const j=await r.json();
    const rows=(j.records||[]).slice(0,12).map(rec=>`<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border)"><b>${escapeHtml(rec.id)}</b><br>${escapeHtml((rec.caption||'').slice(0,120))}<div class="actions"><button class="pill danger" onclick="removeFinetune('${rec.id}')">Remove from set</button></div></div>`).join('');
    $('finetune-status').innerHTML=`finetune set: ${j.count} item(s)<br>${escapeHtml(j.dataset_dir)}<br>manifest: ${escapeHtml(j.manifest)}${rows}`;
  }catch(e){ $('finetune-status').textContent='finetune set failed: '+e.message; }
}
async function addFinetune(id){ const r=await fetch('/api/finetune/items/'+encodeURIComponent(id),{method:'POST'}); const j=await r.json(); if(!r.ok||!j.ok){ alert(j.detail||j.error||await r.text()); return; } await refreshFinetune(); await status(); setStatus(`added ${id} to finetune set
count: ${j.count}`); }
async function removeFinetune(id){ const r=await fetch('/api/finetune/items/'+encodeURIComponent(id),{method:'DELETE'}); if(!r.ok) alert(await r.text()); await refreshFinetune(); status(); }
async function deleteImage(id){ if(!confirm('Remove saved generated image '+id+'?')) return; const r=await fetch('/api/images/'+encodeURIComponent(id),{method:'DELETE'}); if(!r.ok) alert(await r.text()); await refreshImages(); status(); }
function escapeHtml(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
status(); refreshImages(); refreshFinetune(); setInterval(status, 10000);
</script>
</body>
</html>'''


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
