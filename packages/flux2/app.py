#!/usr/bin/env python3
from __future__ import annotations

import io
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

import requests
import torch
from diffusers import Flux2Pipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import get_token
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("FLUX2_ROOT", "/opt/flux2"))
MODEL_ID = os.environ.get("FLUX2_MODEL", "diffusers/FLUX.2-dev-bnb-4bit")
HF_HOME = Path(os.environ.get("HF_HOME", str(ROOT / "hf-cache")))
OUTPUTS_DIR = ROOT / "outputs"
MANIFEST_PATH = OUTPUTS_DIR / "manifest.json"
HOST = os.environ.get("FLUX2_HOST", "0.0.0.0")
PORT = int(os.environ.get("FLUX2_PORT", "7866"))
TEXT_ENCODER_MODE = os.environ.get("FLUX2_TEXT_ENCODER_MODE", "remote").lower()  # remote or local_offload
REMOTE_TEXT_ENCODER_URL = os.environ.get("FLUX2_REMOTE_TEXT_ENCODER_URL", "https://remote-text-encoder-flux-2.huggingface.co/predict")

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FLUX.2 on Spark")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

_pipe: Flux2Pipeline | None = None
_pipe_lock = threading.Lock()
_manifest_lock = threading.Lock()
_last_error: str | None = None
_last_generate_started: float | None = None
_last_generate_finished: float | None = None

PRESETS: dict[str, tuple[int, int]] = {
    "512 smoke": (512, 512),
    "768 square": (768, 768),
    "1024 square": (1024, 1024),
    "16:9 1024w": (1024, 576),
    "9:16 576w": (576, 1024),
    "4:3 1024w": (1024, 768),
    "3:4 768w": (768, 1024),
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
        "text_encoder_mode": TEXT_ENCODER_MODE,
        "remote_text_encoder_url": REMOTE_TEXT_ENCODER_URL if TEXT_ENCODER_MODE == "remote" else None,
        "hf_token_present": bool(get_token()),
    }
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
        try:
            free, total = torch.cuda.mem_get_info()
            out["cuda_free_gib"] = free / 1024**3
            out["cuda_total_gib"] = total / 1024**3
        except Exception:
            pass
    try:
        import diffusers
        import bitsandbytes as bnb
        out["diffusers"] = diffusers.__version__
        out["bitsandbytes"] = bnb.__version__
    except Exception as exc:
        out["dependency_warning"] = f"{type(exc).__name__}: {exc}"
    return out


def _remote_text_encoder(prompts: str | list[str]) -> torch.Tensor:
    token = get_token()
    if not token:
        raise RuntimeError("HF token is required for the remote FLUX.2 text encoder")
    response = requests.post(
        REMOTE_TEXT_ENCODER_URL,
        json={"prompt": prompts},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=240,
    )
    response.raise_for_status()
    # HF example uses torch.load on the returned tensor payload.
    prompt_embeds = torch.load(io.BytesIO(response.content), map_location="cpu", weights_only=False)
    if isinstance(prompt_embeds, dict) and "prompt_embeds" in prompt_embeds:
        prompt_embeds = prompt_embeds["prompt_embeds"]
    if not torch.is_tensor(prompt_embeds):
        raise RuntimeError(f"Unexpected remote text encoder payload type: {type(prompt_embeds).__name__}")
    return prompt_embeds.to("cuda")


def get_pipe() -> Flux2Pipeline:
    global _pipe
    if _pipe is not None:
        return _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this Spark process")
        if TEXT_ENCODER_MODE == "remote":
            pipe = Flux2Pipeline.from_pretrained(MODEL_ID, text_encoder=None, torch_dtype=torch.bfloat16, cache_dir=str(HF_HOME))
            pipe.to("cuda")
        elif TEXT_ENCODER_MODE == "local_offload":
            from diffusers import AutoModel
            from transformers import Mistral3ForConditionalGeneration

            text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cpu", cache_dir=str(HF_HOME)
            )
            dit = AutoModel.from_pretrained(
                MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cpu", cache_dir=str(HF_HOME)
            )
            pipe = Flux2Pipeline.from_pretrained(
                MODEL_ID, text_encoder=text_encoder, transformer=dit, torch_dtype=torch.bfloat16, cache_dir=str(HF_HOME)
            )
            pipe.enable_model_cpu_offload()
        else:
            raise RuntimeError(f"Unsupported FLUX2_TEXT_ENCODER_MODE={TEXT_ENCODER_MODE!r}; use remote or local_offload")
        _pipe = pipe
        return _pipe


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    width: int = Field(512, ge=256, le=2048)
    height: int = Field(512, ge=256, le=2048)
    steps: int = Field(12, ge=1, le=80)
    guidance_scale: float = Field(4.0, ge=0.0, le=12.0)
    seed: int = Field(42, ge=0, le=2_147_483_647)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "flux2",
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
    return {"ok": True, "image_count": count, "presets": PRESETS, "model_loaded": _pipe is not None, **_device_summary()}



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
        generator = torch.Generator(device="cuda").manual_seed(req.seed)
        kwargs: dict[str, Any] = {
            "generator": generator,
            "num_inference_steps": req.steps,
            "guidance_scale": req.guidance_scale,
        }
        # Width/height are available on recent Diffusers Flux pipelines; ignore gracefully if not accepted.
        kwargs["width"] = req.width
        kwargs["height"] = req.height
        if TEXT_ENCODER_MODE == "remote":
            kwargs["prompt_embeds"] = _remote_text_encoder(req.prompt)
        else:
            kwargs["prompt"] = req.prompt
        try:
            with torch.inference_mode():
                result = pipe(**kwargs)
        except TypeError as type_exc:
            if "width" in kwargs or "height" in kwargs:
                kwargs.pop("width", None)
                kwargs.pop("height", None)
                with torch.inference_mode():
                    result = pipe(**kwargs)
            else:
                raise type_exc
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
            "settings": {
                "width": req.width,
                "height": req.height,
                "steps": req.steps,
                "guidance_scale": req.guidance_scale,
                "seed": req.seed,
                "dtype": "bfloat16",
                "text_encoder_mode": TEXT_ENCODER_MODE,
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


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>FLUX.2 on Spark</title>
<style>:root{--bg:#0a0807;--text:#f4f2fb;--muted:#9a98a6;--dim:#686672;--accent:#5FE3A0;--warn:#FFB35C;--danger:#ff6f7d;--border:rgba(255,255,255,.09);--radius:18px}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% -10%,rgba(95,227,160,.14),transparent 34%),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:100vh}.shell{display:grid;grid-template-columns:380px minmax(0,1fr);min-height:100vh}.side{border-right:1px solid var(--border);background:rgba(10,8,7,.86);padding:26px;position:sticky;top:0;height:100vh;overflow:auto}.main{padding:28px;max-width:1600px;width:100%;margin:0 auto}.brand{display:flex;gap:12px;align-items:center;margin-bottom:24px}.spark{width:14px;height:14px;border:2px solid var(--accent);transform:rotate(45deg);box-shadow:0 0 22px rgba(95,227,160,.5)}h1{font-size:24px;margin:0;letter-spacing:-.03em}.sub{color:var(--muted);font-size:13px;line-height:1.55}.card{border:1px solid var(--border);border-radius:var(--radius);background:rgba(255,255,255,.025);padding:18px;margin-bottom:16px;box-shadow:0 26px 80px rgba(0,0,0,.25)}label{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em;margin:14px 0 7px}textarea,input,select{width:100%;border:1px solid var(--border);background:#0d0c0b;color:var(--text);border-radius:12px;padding:11px 12px;font:inherit}textarea{min-height:150px;resize:vertical}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.btn{width:100%;border:0;border-radius:12px;padding:14px 16px;background:var(--accent);color:#06100b;font-weight:800;letter-spacing:.04em;cursor:pointer}.btn.secondary{background:rgba(95,227,160,.08);color:var(--accent);border:1px solid rgba(95,227,160,.25)}.btn:disabled{opacity:.45;cursor:not-allowed}.status{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);font-size:12px;white-space:pre-wrap}.hero{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:18px;align-items:start}.preview{min-height:620px;display:grid;place-items:center;border:1px solid var(--border);border-radius:var(--radius);background:linear-gradient(135deg,rgba(255,255,255,.03),rgba(255,255,255,.01));overflow:hidden}.preview img{max-width:100%;max-height:84vh;border-radius:12px}.library{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}.thumb{border:1px solid var(--border);border-radius:16px;background:rgba(255,255,255,.025);overflow:hidden}.thumb img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;background:#111}.thumb .body{padding:12px}.meta{font-size:11px;color:var(--dim);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;line-height:1.5}.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.pill{font-size:11px;color:var(--accent);border:1px solid rgba(95,227,160,.25);padding:6px 8px;border-radius:999px;text-decoration:none;background:rgba(95,227,160,.06);cursor:pointer}.pill.danger{color:var(--danger);border-color:rgba(255,111,125,.25);background:rgba(255,111,125,.06)}.loading{position:fixed;inset:0;background:rgba(10,8,7,.78);display:none;place-items:center;z-index:5;backdrop-filter:blur(10px)}.spinner{width:58px;height:58px;border-radius:50%;border:3px solid rgba(255,255,255,.12);border-top-color:var(--accent);animation:spin 1s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:900px){.shell{grid-template-columns:1fr}.side{position:relative;height:auto}.hero{grid-template-columns:1fr}}</style></head><body>
<div class="loading" id="loading"><div><div class="spinner"></div><p class="status" id="loading-text" style="text-align:center;margin-top:16px">Generating...</p></div></div>
<div class="shell"><aside class="side"><div class="brand"><div class="spark"></div><div><h1>FLUX.2</h1><div class="sub">FLUX.2-dev bnb-4bit on Spark</div></div></div><div class="card"><label>Prompt</label><textarea id="prompt">A macro photograph of a hermit crab using a soda can as its shell, the can clearly reads "Spark FLUX", sunlit beach, sharp detail</textarea><label>Preset</label><select id="preset" onchange="applyPreset()"><option value="512 smoke">512 smoke</option><option value="768 square">768 square</option><option value="1024 square">1024 square</option><option value="16:9 1024w">16:9 1024w</option><option value="9:16 576w">9:16 576w</option><option value="4:3 1024w">4:3 1024w</option><option value="3:4 768w">3:4 768w</option></select><div class="row"><div><label>Width</label><input id="width" type="number" value="512"></div><div><label>Height</label><input id="height" type="number" value="512"></div></div><div class="row"><div><label>Steps</label><input id="steps" type="number" min="1" max="80" value="12"></div><div><label>Guidance</label><input id="guidance" type="number" min="0" max="12" step="0.1" value="4.0"></div></div><label>Seed</label><div class="row"><input id="seed" type="number" min="0" max="2147483647" value="42"><button class="btn secondary" onclick="randomSeed()">Random</button></div><div style="height:14px"></div><button class="btn" id="generate" onclick="generate()">Generate</button></div><div class="card"><div class="status" id="status">Loading status...</div></div></aside><main class="main"><section class="hero"><div class="preview" id="preview"><div class="sub">Generated image will appear here and be saved with metadata/provenance.</div></div><div class="card"><h2 style="margin-top:0">Run notes</h2><div class="sub">Default mode uses the Hugging Face remote FLUX.2 text encoder and local 4-bit transformer. Stop other image services before real generation for memory headroom.</div><div style="height:14px"></div><button class="btn secondary" onclick="refreshImages()">Refresh Library</button></div></section><h2>Saved Image Library</h2><div class="library" id="library"></div></main></div>
<script>
const presets={"512 smoke":[512,512],"768 square":[768,768],"1024 square":[1024,1024],"16:9 1024w":[1024,576],"9:16 576w":[576,1024],"4:3 1024w":[1024,768],"3:4 768w":[768,1024]};
function $(id){return document.getElementById(id)} function setStatus(msg){$('status').textContent=msg} function applyPreset(){const p=presets[$('preset').value]; if(p){$('width').value=p[0]; $('height').value=p[1];}} function randomSeed(){ $('seed').value=Math.floor(Math.random()*2147483647) } function showLoading(msg){$('loading-text').textContent=msg||'Generating...'; $('loading').style.display='grid'} function hideLoading(){ $('loading').style.display='none' }
async function status(){ try{ const r=await fetch('/api/status',{cache:'no-store'}); const j=await r.json(); setStatus(`model: ${j.model_id}\nloaded: ${j.model_loaded}\nimages: ${j.image_count}\ndevice: ${j.device||'--'}\ncuda free: ${j.cuda_free_gib?j.cuda_free_gib.toFixed(1)+' GiB':'--'}\ntext encoder: ${j.text_encoder_mode}\nbnb: ${j.bitsandbytes||'--'}\nhf token: ${j.hf_token_present}`)}catch(e){setStatus('status failed: '+e.message)} }
async function generate(){ const body={prompt:$('prompt').value, width:+$('width').value, height:+$('height').value, steps:+$('steps').value, guidance_scale:+$('guidance').value, seed:+$('seed').value}; $('generate').disabled=true; showLoading('Loading FLUX.2 / generating. First run can take several minutes.'); try{ const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const j=await r.json(); if(!r.ok||!j.ok) throw new Error(j.detail||j.error||`HTTP ${r.status}`); renderPreview(j.image); await refreshImages(); setStatus(`generated ${j.image.id}\nduration: ${j.image.duration_sec.toFixed(1)}s\nseed: ${j.image.settings.seed}`); }catch(e){console.error(e); alert('Generation failed: '+e.message); setStatus('generation failed: '+e.message);} finally{hideLoading(); $('generate').disabled=false; status();}}
function renderPreview(item){ $('preview').innerHTML=`<img src="${item.image.static_url}?t=${Date.now()}" alt="generated">`; }
function card(item){ const s=item.settings||{}; return `<div class="thumb"><img src="${item.image.static_url}?t=${Date.now()}" onclick="renderPreview(window.images.find(x=>x.id==='${item.id}'))"><div class="body"><div class="meta">${item.created_at}<br>${s.width}×${s.height} · ${s.steps} steps · guidance ${s.guidance_scale}<br>seed ${s.seed}<br>${(item.prompt||'').slice(0,130)}</div><div class="actions"><a class="pill" href="${item.image.static_url}" target="_blank">Image</a><a class="pill" href="${item.metadata.static_url}" target="_blank">Metadata</a><button class="pill danger" onclick="deleteImage('${item.id}')">Remove</button></div></div></div>`}
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
