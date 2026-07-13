#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import ssl
import subprocess
import time
import urllib.error
import urllib.request
import urllib.parse
import secrets
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HOST = os.environ.get("SPARK_DASHBOARD_HOST", "0.0.0.0")
PORT = int(os.environ.get("SPARK_DASHBOARD_PORT", "7862"))
PUBLIC_HOST = os.environ.get("SPARK_PUBLIC_HOST", os.uname().nodename)
INSTALL_ROOT = Path(os.environ.get("SPARK_DASHBOARD_ROOT", "/opt/spark-dashboard"))
CONTROL_TOKEN_PATH = Path(os.environ.get("SPARK_CONTROL_TOKEN_PATH", str(INSTALL_ROOT / "control.token")))
BENCHMARK_PATH = Path(os.environ.get("SPARK_BENCHMARK_PATH", str(INSTALL_ROOT / "benchmarks.jsonl")))
BENCHMARK_MAX_RECORDS = 200
BENCHMARK_PROMPT = """/no_think
You are benchmarking a local LLM. Write a concise but information-dense technical note about running multiple AI model services on a unified-memory workstation. Include practical tradeoffs, scheduling concerns, and one final recommendation. Keep writing until you have given a complete answer."""
BENCHMARK_MAX_TOKENS = 192

LLM_BENCHMARK_SERVICE_KEYS = ("qwen", "ornith", "mistralmedium")
# Fixed allow-list only: the dashboard never accepts arbitrary commands.
CONTROL_ACTIONS: dict[tuple[str, str], dict[str, Any]] = {
    ("qwen", "start"): {
        "label": "Start Qwen fast text stack",
        "cmd": [str(INSTALL_ROOT / "bin/start-text-stack.sh")],
        "timeout": 900,
    },
    ("qwen", "stop"): {
        "label": "Stop Qwen vLLM",
        "cmd": ["systemctl", "--user", "stop", "qwen-nvfp4-vllm.service"],
        "timeout": 120,
    },
    ("personaplex", "start"): {
        "label": "Start PersonaPlex BNB4",
        "cmd": ["systemctl", "--user", "start", "personaplex.service"],
        "timeout": 420,
    },
    ("personaplex", "stop"): {
        "label": "Stop PersonaPlex",
        "cmd": ["systemctl", "--user", "stop", "personaplex.service"],
        "timeout": 120,
    },
    ("hidream", "start"): {
        "label": "Start HiDream O1 web UI",
        "cmd": ["systemctl", "--user", "start", "hidream-o1.service"],
        "timeout": 180,
    },
    ("hidream", "stop"): {
        "label": "Stop HiDream O1 web UI",
        "cmd": ["systemctl", "--user", "stop", "hidream-o1.service"],
        "timeout": 180,
    },
    ("hidream", "unload"): {
        "label": "Unload HiDream O1 model",
        "cmd": ["curl", "-fsS", "-X", "POST", "http://127.0.0.1:7861/api/unload"],
        "timeout": 180,
    },
    ("pixal3d", "start"): {
        "label": "Start Pixal3D web UI",
        "cmd": ["systemctl", "--user", "start", "pixal3d.service"],
        "timeout": 240,
    },
    ("pixal3d", "stop"): {
        "label": "Stop Pixal3D web UI",
        "cmd": ["systemctl", "--user", "stop", "pixal3d.service"],
        "timeout": 120,
    },
    ("zimage", "start"): {
        "label": "Start Z-Image web UI",
        "cmd": ["systemctl", "--user", "start", "z-image.service"],
        "timeout": 240,
    },
    ("zimage", "stop"): {
        "label": "Stop Z-Image web UI",
        "cmd": ["systemctl", "--user", "stop", "z-image.service"],
        "timeout": 120,
    },
    ("qwenimage", "start"): {
        "label": "Start Qwen-Image web UI",
        "cmd": ["systemctl", "--user", "start", "qwen-image.service"],
        "timeout": 240,
    },
    ("qwenimage", "stop"): {
        "label": "Stop Qwen-Image web UI",
        "cmd": ["systemctl", "--user", "stop", "qwen-image.service"],
        "timeout": 120,
    },
    ("flux2", "start"): {
        "label": "Start FLUX.2 image UI",
        "cmd": ["systemctl", "--user", "start", "flux2.service"],
        "timeout": 240,
    },
    ("flux2", "stop"): {
        "label": "Stop FLUX.2 image UI",
        "cmd": ["systemctl", "--user", "stop", "flux2.service"],
        "timeout": 120,
    },
    ("zimage", "unload"): {
        "label": "Unload Z-Image model",
        "cmd": ["curl", "-fsS", "-X", "POST", "http://127.0.0.1:7864/api/unload"],
        "timeout": 180,
    },
    ("qwenimage", "unload"): {
        "label": "Unload Qwen-Image model",
        "cmd": ["curl", "-fsS", "-X", "POST", "http://127.0.0.1:7865/api/unload"],
        "timeout": 180,
    },
    ("flux2", "unload"): {
        "label": "Unload FLUX.2 model",
        "cmd": ["curl", "-fsS", "-X", "POST", "http://127.0.0.1:7866/api/unload"],
        "timeout": 180,
    },
    ("domainshuttle", "start"): {
        "label": "Start DomainShuttle WebUI",
        "cmd": ["systemctl", "--user", "start", "domainshuttle-web.service"],
        "timeout": 180,
    },
    ("domainshuttle", "stop"): {
        "label": "Stop DomainShuttle WebUI",
        "cmd": ["systemctl", "--user", "stop", "domainshuttle-web.service"],
        "timeout": 120,
    },
    ("krea2", "start"): {
        "label": "Start Krea-2 Turbo WebUI",
        "cmd": ["systemctl", "--user", "start", "krea-2.service"],
        "timeout": 180,
    },
    ("krea2", "stop"): {
        "label": "Stop Krea-2 Turbo WebUI",
        "cmd": ["systemctl", "--user", "stop", "krea-2.service"],
        "timeout": 120,
    },
    ("krea2", "unload"): {
        "label": "Unload Krea-2 Turbo model",
        "cmd": ["curl", "-fsS", "-X", "POST", "http://127.0.0.1:7868/api/unload"],
        "timeout": 180,
    },
    ("un0", "start"): {
        "label": "Start Un-0 WebUI",
        "cmd": ["systemctl", "--user", "start", "un0-web.service"],
        "timeout": 60,
    },
    ("un0", "stop"): {
        "label": "Stop Un-0 WebUI",
        "cmd": ["systemctl", "--user", "stop", "un0-web.service"],
        "timeout": 60,
    },
    ("triposplat", "start"): {
        "label": "Start TripoSplat WebUI",
        "cmd": ["systemctl", "--user", "start", "triposplat-web.service"],
        "timeout": 60,
    },
    ("triposplat", "stop"): {
        "label": "Stop TripoSplat WebUI",
        "cmd": ["systemctl", "--user", "stop", "triposplat-web.service"],
        "timeout": 60,
    },
    ("agent3dify", "start"): {
        "label": "Start Agent3Dify WebUI",
        "cmd": ["systemctl", "--user", "start", "agent3dify-web.service"],
        "timeout": 60,
    },
    ("agent3dify", "stop"): {
        "label": "Stop Agent3Dify WebUI",
        "cmd": ["systemctl", "--user", "stop", "agent3dify-web.service"],
        "timeout": 60,
    },
    ("ornith", "start"): {
        "label": "Start Ornith vLLM",
        "cmd": ["systemctl", "--user", "start", "ornith-vllm.service"],
        "timeout": 60,
    },
    ("ornith", "stop"): {
        "label": "Stop Ornith vLLM",
        "cmd": ["systemctl", "--user", "stop", "ornith-vllm.service"],
        "timeout": 90,
    },
    ("mistralmedium", "start"): {
        "label": "Start Mistral Medium 3.5 128B",
        "cmd": ["systemctl", "--user", "start", "mistral-medium-vllm.service"],
        "timeout": 1800,
    },
    ("mistralmedium", "stop"): {
        "label": "Stop Mistral Medium 3.5",
        "cmd": ["systemctl", "--user", "stop", "mistral-medium-vllm.service"],
        "timeout": 120,
    },
}

_LOG_CACHE: dict[str, tuple[float, list[str]]] = {}
LOG_CACHE_TTL_SEC = 5.0

SERVICES = [
    {
        "key": "qwen",
        "name": "Qwen vLLM",
        "unit": "qwen-nvfp4-vllm.service",
        "kind": "text model",
        "health_url": "http://127.0.0.1:8000/v1/models",
        "public_url": f"http://{PUBLIC_HOST}:8000/v1/models",
        "frontend_url": f"http://{PUBLIC_HOST}:8000/docs",
        "accent": "accent",
        "process_hints": ["qwen-nvfp4"],
    },
    {
        "key": "ornith",
        "name": "Ornith 1.0 35B",
        "unit": "ornith-vllm.service",
        "kind": "agentic coding LLM",
        "health_url": "http://127.0.0.1:8001/v1/models",
        "public_url": f"http://{PUBLIC_HOST}:8001/v1/models",
        "frontend_url": f"http://{PUBLIC_HOST}:8001/docs",
        "accent": "warn",
        "process_hints": ["ornith-vllm", "Ornith-1.0-35B"],
    },
    {
        "key": "mistralmedium",
        "name": "Mistral Medium 3.5",
        "unit": "mistral-medium-vllm.service",
        "kind": "128B NVFP4 text model",
        "health_url": "http://127.0.0.1:8002/v1/models",
        "public_url": f"http://{PUBLIC_HOST}:8002/v1/models",
        "frontend_url": f"http://{PUBLIC_HOST}:8002/docs",
        "accent": "warn",
        "process_hints": ["mistral-medium-vllm", "Mistral-Medium-3.5-128B-NVFP4"],
    },
    {
        "key": "personaplex",
        "name": "PersonaPlex",
        "unit": "personaplex.service",
        "kind": "voice frontend",
        "health_url": "https://127.0.0.1:8998/",
        "public_url": f"https://{PUBLIC_HOST}:8998/",
        "frontend_url": f"https://{PUBLIC_HOST}:8998/",
        "accent": "info",
        "process_hints": ["personaplex-bnb4", "moshi.server"],
    },
    {
        "key": "hidream",
        "name": "HiDream O1",
        "unit": "hidream-o1.service",
        "kind": "image frontend",
        "health_url": "http://127.0.0.1:7861/",
        "public_url": f"http://{PUBLIC_HOST}:7861/",
        "frontend_url": f"http://{PUBLIC_HOST}:7861/",
        "accent": "warn",
        "process_hints": ["hidream-o1", "HiDream", "app.py --model_path"],
    },
    {
        "key": "pixal3d",
        "name": "Pixal3D",
        "unit": "pixal3d.service",
        "kind": "image-to-3D frontend",
        "health_url": "http://127.0.0.1:7863/",
        "public_url": f"http://{PUBLIC_HOST}:7863/",
        "frontend_url": f"http://{PUBLIC_HOST}:7863/",
        "accent": "good",
        "process_hints": ["Pixal3D", "app.py --low_vram", "pixal3d.service"],
    },
    {
        "key": "zimage",
        "name": "Z-Image",
        "unit": "z-image.service",
        "kind": "text-to-image frontend",
        "health_url": "http://127.0.0.1:7864/health",
        "public_url": f"http://{PUBLIC_HOST}:7864/health",
        "frontend_url": f"http://{PUBLIC_HOST}:7864/",
        "accent": "accent",
        "process_hints": ["z-image", "ZImagePipeline", "/opt/z-image/app.py"],
    },
    {
        "key": "qwenimage",
        "name": "Qwen-Image",
        "unit": "qwen-image.service",
        "kind": "text-to-image frontend",
        "health_url": "http://127.0.0.1:7865/health",
        "public_url": f"http://{PUBLIC_HOST}:7865/health",
        "frontend_url": f"http://{PUBLIC_HOST}:7865/",
        "accent": "warn",
        "process_hints": ["qwen-image", "QwenImagePipeline", "/opt/qwen-image/app.py"],
    },    {
        "key": "flux2",
        "name": "FLUX.2",
        "unit": "flux2.service",
        "kind": "text-to-image frontend",
        "health_url": "http://127.0.0.1:7866/health",
        "public_url": f"http://{PUBLIC_HOST}:7866/health",
        "frontend_url": f"http://{PUBLIC_HOST}:7866/",
        "accent": "accent",
        "process_hints": ["flux2", "Flux2Pipeline", "/opt/flux2/app.py"],
    },
    {
        "key": "domainshuttle",
        "name": "DomainShuttle",
        "unit": "domainshuttle-web.service",
        "kind": "subject-to-video frontend",
        "health_url": "http://127.0.0.1:7867/health",
        "public_url": f"http://{PUBLIC_HOST}:7867/health",
        "frontend_url": f"http://{PUBLIC_HOST}:7867/",
        "accent": "info",
        "process_hints": ["domainshuttle", "predict_r2v_batch", "/opt/domainshuttle/web/app.py"],
    },
    {
        "key": "krea2",
        "name": "Krea-2 Turbo",
        "unit": "krea-2.service",
        "kind": "text-to-image frontend",
        "health_url": "http://127.0.0.1:7868/health",
        "public_url": f"http://{PUBLIC_HOST}:7868/health",
        "frontend_url": f"http://{PUBLIC_HOST}:7868/",
        "accent": "good",
        "process_hints": ["krea-2", "Krea2Pipeline", "/opt/krea-2/app.py"],
    },
    {
        "key": "un0",
        "name": "Un-0 Kuramoto",
        "unit": "un0-web.service",
        "kind": "oscillator image generation + training UI",
        "health_url": "http://127.0.0.1:7870/health",
        "public_url": f"http://{PUBLIC_HOST}:7870/health",
        "frontend_url": f"http://{PUBLIC_HOST}:7870/",
        "accent": "good",
        "process_hints": ["un0-web", "/opt/un0/app.py", "train_cifar10.py", "inference.py"],
    },
    {
        "key": "triposplat",
        "name": "TripoSplat",
        "unit": "triposplat-web.service",
        "kind": "image-to-3D Gaussian splat frontend",
        "health_url": "http://127.0.0.1:7871/health",
        "public_url": f"http://{PUBLIC_HOST}:7871/health",
        "frontend_url": f"http://{PUBLIC_HOST}:7871/",
        "accent": "good",
        "process_hints": ["triposplat-web", "/opt/triposplat/app.py", "run_job.py", "TripoSplatPipeline"],
    },
    {
        "key": "agent3dify",
        "name": "Agent3Dify Local Qwen",
        "unit": "agent3dify-web.service",
        "kind": "2D drawing-to-CAD agent frontend",
        "health_url": "http://127.0.0.1:7869/health",
        "public_url": f"http://{PUBLIC_HOST}:7869/health",
        "frontend_url": f"http://{PUBLIC_HOST}:7869/",
        "accent": "info",
        "process_hints": ["agent3dify", "/opt/agent3dify/web/app.py", "Qwen3.6-35B-A3B-NVFP4"],
    },

]

FAVICON_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#0a0807"/>
  <path d="M32 8 L54 20.7 V43.3 L32 56 L10 43.3 V20.7 Z" fill="none" stroke="#5FE3A0" stroke-width="3.2" stroke-linejoin="round"/>
  <path d="M32 17 L46 25.1 V38.9 L32 47 L18 38.9 V25.1 Z" fill="#5FE3A0" fill-opacity="0.13" stroke="#5FE3A0" stroke-width="1.7" stroke-linejoin="round"/>
  <circle cx="32" cy="32" r="5" fill="#5FE3A0"/>
  <path d="M32 4 V13 M32 51 V60 M4 32 H13 M51 32 H60" stroke="#5FE3A0" stroke-width="2" stroke-linecap="round" opacity=".75"/>
</svg>'''

HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Spark Dashboard</title>
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="shortcut icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:#0a0807;
      --bg-elev:#0d0a09;
      --panel:rgba(255,255,255,0.015);
      --panel-hover:rgba(95,227,160,0.04);
      --border:rgba(255,255,255,0.07);
      --border-strong:rgba(255,255,255,0.12);
      --text:#ECEAF6;
      --text-bright:#F4F2FB;
      --text-muted:#9A98A6;
      --text-dim:#8A8896;
      --text-faint:#56545F;
      --accent:#5FE3A0;
      --accent-soft:rgba(95,227,160,0.16);
      --accent-ink:#0B0A12;
      --info:#5FD3FF;
      --warn:#FFB35C;
      --danger:#ff6f7d;
      --shadow:0 30px 80px rgba(0,0,0,0.45);
      --radius:16px;
    }

    @keyframes xnspin { to { transform: rotate(360deg); } }
    @keyframes xnblink { 0%,49%{opacity:1} 50%,100%{opacity:0} }
    @keyframes xnpulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.45;transform:scale(.7)} }
    @keyframes xnrise { 0%{transform:translateY(14px);opacity:0} 100%{transform:translateY(0);opacity:1} }

    * { box-sizing:border-box; }
    html, body { min-height:100%; margin:0; }
    body {
      background:var(--bg);
      color:var(--text);
      font-family:'Space Grotesk', system-ui, sans-serif;
      overflow-x:hidden;
    }
    .texture, .glow { position:fixed; inset:0; pointer-events:none; z-index:0; }
    .glow { background:radial-gradient(120% 90% at 50% 42%, rgba(95,227,160,0.10), rgba(10,8,7,0) 60%); }
    .texture {
      opacity:.35;
      background-image:linear-gradient(rgba(255,255,255,0.035) 1px,transparent 1px), linear-gradient(90deg,rgba(255,255,255,0.035) 1px,transparent 1px);
      background-size:60px 60px;
      mask-image:radial-gradient(circle at 50% 45%, #000 0%, transparent 72%);
    }
    .shell { position:relative; z-index:1; display:grid; grid-template-columns:260px minmax(0,1fr); min-height:100vh; }
    aside {
      position:sticky; top:0; height:100vh; padding:34px 26px;
      border-right:1px solid var(--border); background:rgba(10,8,7,.78); backdrop-filter: blur(12px);
    }
    .wordmark { display:flex; align-items:center; gap:12px; }
    .glyph { width:13px; height:13px; border:1.5px solid var(--accent); transform:rotate(45deg); animation:xnspin 14s linear infinite; box-shadow:0 0 14px var(--accent-soft); }
    .wordmark span { font-family:'Space Grotesk'; font-weight:600; letter-spacing:.26em; font-size:15px; color:var(--text-bright); }
    nav { margin-top:56px; display:grid; gap:10px; }
    .navitem { font-family:'IBM Plex Mono'; font-size:12.5px; letter-spacing:.06em; color:var(--text-dim); text-decoration:none; padding:12px 0 12px 16px; border-left:1px solid transparent; }
    .navitem.active { color:var(--accent); border-left-color:var(--accent); text-shadow:0 0 18px var(--accent-soft); }
    .side-footer { position:absolute; left:26px; right:26px; bottom:30px; color:var(--text-faint); font-family:'IBM Plex Mono'; font-size:11px; line-height:1.7; }
    main { padding:34px; max-width:1680px; width:100%; margin:0 auto; }
    .topbar { display:flex; align-items:center; justify-content:space-between; gap:18px; margin-bottom:26px; }
    .kicker { font-family:'IBM Plex Mono'; font-size:12px; letter-spacing:.2em; color:var(--accent); text-transform:uppercase; }
    h1 { margin:8px 0 0; color:var(--text-bright); font-size:42px; line-height:1; font-weight:500; letter-spacing:-.035em; text-wrap:pretty; }
    .top-actions { display:flex; align-items:center; gap:12px; flex-wrap:wrap; justify-content:flex-end; }
    .pill { display:inline-flex; align-items:center; gap:10px; padding:11px 16px; border:1px solid var(--border); border-radius:12px; background:rgba(255,255,255,0.02); font-family:'IBM Plex Mono'; font-size:12px; letter-spacing:.08em; color:var(--text-dim); white-space:nowrap; }
    .dot { width:7px; height:7px; border-radius:50%; background:var(--accent); box-shadow:0 0 10px var(--accent); animation:xnpulse 2s ease-in-out infinite; }
    .dot.warn { background:var(--warn); box-shadow:0 0 10px var(--warn); }
    .dot.danger { background:var(--danger); box-shadow:0 0 10px var(--danger); }
    .button { font-family:'IBM Plex Mono'; font-size:13px; letter-spacing:.08em; font-weight:600; color:var(--accent-ink); padding:13px 20px; border-radius:10px; background:var(--accent); box-shadow:0 0 30px var(--accent-soft); text-decoration:none; border:0; cursor:pointer; transition:.25s ease; }
    .button:hover { box-shadow:0 0 48px var(--accent-soft); transform:translateY(-1px); }
    .ghost { color:var(--accent); border:1px solid var(--accent-soft); background:rgba(95,227,160,0.04); box-shadow:none; }
    .grid { display:grid; gap:18px; }
    .kpis { grid-template-columns:repeat(4, minmax(0,1fr)); margin-bottom:18px; }
    .card { padding:26px; border:1px solid var(--border); border-radius:var(--radius); background:var(--panel); box-shadow:var(--shadow); transition:border-color .3s,background .3s; animation:xnrise .55s ease both; }
    .card:hover { border-color:var(--accent-soft); background:var(--panel-hover); }
    .label { font-family:'IBM Plex Mono'; font-size:10.5px; letter-spacing:.18em; color:var(--text-faint); text-transform:uppercase; }
    .value { margin-top:6px; font-size:26px; font-weight:500; color:var(--text-bright); letter-spacing:-.01em; }
    .value.good { color:var(--accent); }
    .value.info { color:var(--info); }
    .value.warn { color:var(--warn); }
    .sub { margin-top:8px; color:var(--text-dim); font-family:'IBM Plex Mono'; font-size:12px; line-height:1.5; }
    .layout { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(360px,.65fr); gap:18px; align-items:start; }
    .services { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:18px; }
    .service-card { min-height:245px; display:flex; flex-direction:column; }
    .service-head { display:flex; align-items:flex-start; justify-content:space-between; gap:14px; margin-bottom:18px; flex-wrap:wrap; }
    .service-title { margin:0; font-size:22px; line-height:1.18; font-weight:500; letter-spacing:-.015em; color:var(--text); }
    .badge { display:inline-flex; align-items:center; gap:7px; padding:7px 9px; border-radius:5px; font-family:'IBM Plex Mono'; font-size:10px; font-weight:600; letter-spacing:.09em; color:var(--accent); background:rgba(95,227,160,0.10); text-transform:uppercase; }
    .badge.info { color:var(--info); background:rgba(95,211,255,.10); }
    .badge.warn { color:var(--warn); background:rgba(255,179,92,.10); }
    .badge.dead { color:var(--danger); background:rgba(255,111,125,.10); }
    .svc-metrics { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:10px; }
    .mini { padding:12px 0; border-top:1px solid rgba(255,255,255,.06); }
    .mini .value { font-size:19px; margin-top:5px; }
    .links { display:flex; gap:10px; flex-wrap:wrap; margin-top:auto; padding-top:18px; }
    .small-link { font-family:'IBM Plex Mono'; font-size:12px; color:var(--accent); border:1px solid var(--accent-soft); border-radius:10px; padding:10px 12px; text-decoration:none; background:rgba(95,227,160,.035); cursor:pointer; }
    .log-tail { margin-top:14px; border-top:1px solid rgba(255,255,255,.06); padding-top:12px; }
    .log-tail summary { cursor:pointer; user-select:none; font-family:'IBM Plex Mono'; font-size:11px; letter-spacing:.12em; color:var(--text-dim); text-transform:uppercase; }
    .log-tail pre { margin:10px 0 0; max-height:155px; overflow:auto; white-space:pre-wrap; word-break:break-word; font-family:'IBM Plex Mono'; font-size:11px; line-height:1.45; color:#aaa7b6; background:rgba(0,0,0,.28); border:1px solid rgba(255,255,255,.055); border-radius:10px; padding:10px; }
    .panel-title { display:flex; justify-content:space-between; align-items:center; gap:16px; margin-bottom:18px; }
    .panel-title h2 { margin:0; font-size:22px; font-weight:500; letter-spacing:-.02em; }
    .orb-card { position:relative; overflow:hidden; min-height:365px; display:grid; grid-template-rows:auto 1fr; }
    canvas { width:100%; height:250px; display:block; }
    .console { display:inline-flex; align-items:center; gap:12px; padding:11px 18px; border:1px solid rgba(255,255,255,0.08); border-radius:12px; background:rgba(255,255,255,0.02); font-family:'IBM Plex Mono'; font-size:13px; color:var(--text-muted); }
    .caret { display:inline-block; width:8px; height:17px; background:var(--accent); animation:xnblink 1.05s steps(1) infinite; }
    .frontend { margin-top:18px; }
    .frontend-frame { height:520px; width:100%; border:1px solid var(--border); border-radius:16px; background:#050403; overflow:hidden; }
    iframe { width:100%; height:100%; border:0; background:#080706; }
    .tabs { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }
    .tab { font-family:'IBM Plex Mono'; font-size:12px; color:var(--text-dim); border:1px solid var(--border); background:rgba(255,255,255,.015); padding:10px 12px; border-radius:10px; cursor:pointer; }
    .tab.active { color:var(--accent); border-color:var(--accent-soft); background:rgba(95,227,160,.04); }
    .feed { max-height:360px; overflow:hidden; mask-image:linear-gradient(180deg,#000 78%,transparent); }
    .event { display:flex; align-items:center; gap:14px; padding:13px 0; border-bottom:1px solid rgba(255,255,255,0.06); font-family:'IBM Plex Mono'; font-size:13px; }
    .event-tag { flex:none; width:42px; height:20px; display:grid; place-items:center; border-radius:5px; font-size:10px; font-weight:600; letter-spacing:.5px; background:rgba(95,227,160,0.14); color:var(--accent); }
    .event-tag.info { color:var(--info); background:rgba(95,211,255,.12); }
    .event-tag.warn { color:var(--warn); background:rgba(255,179,92,.12); }
    .event-text { color:#c9c7d4; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .event-time { margin-left:auto; color:var(--text-faint); font-size:11px; }
    .bench-table { width:100%; border-collapse:collapse; font-family:'IBM Plex Mono'; font-size:12px; }
    .bench-table th { text-align:left; color:var(--text-faint); font-weight:600; padding:8px 7px; border-bottom:1px solid rgba(255,255,255,.08); letter-spacing:.08em; text-transform:uppercase; font-size:10px; }
    .bench-table td { color:#c9c7d4; padding:9px 7px; border-bottom:1px solid rgba(255,255,255,.055); vertical-align:top; }
    .bench-table .metric { color:var(--accent); font-weight:600; }
    .bench-controls { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin:10px 0 14px; }
    .bench-grid { display:grid; gap:14px; }
    .bench-result { font-family:'IBM Plex Mono'; font-size:12px; color:var(--text-muted); }
    .bar { height:7px; border-radius:999px; background:rgba(255,255,255,.055); overflow:hidden; margin-top:10px; }
    .bar span { display:block; height:100%; width:0%; background:var(--accent); box-shadow:0 0 20px var(--accent-soft); transition:width .45s ease; }
    .bar.info span { background:var(--info); box-shadow:0 0 20px rgba(95,211,255,.18); }
    .bar.warn span { background:var(--warn); box-shadow:0 0 20px rgba(255,179,92,.18); }
    code { font-family:'IBM Plex Mono'; color:var(--text-muted); font-size:12px; }

    .vllm-chat-panel { display:block; scroll-margin-top:24px; }
    .vllm-chat-panel.active { display:block; }
    .vllm-chat-log { min-height:180px; max-height:420px; overflow:auto; padding:14px; border:1px solid rgba(255,255,255,.08); border-radius:18px; background:rgba(0,0,0,.22); white-space:pre-wrap; }
    .vllm-chat-msg { margin:0 0 14px; }
    .vllm-chat-role { font-family:'IBM Plex Mono'; color:var(--accent); font-size:11px; letter-spacing:.12em; text-transform:uppercase; margin-bottom:4px; }
    .vllm-chat-content { color:var(--text); line-height:1.55; }
    .vllm-chat-input { width:100%; min-height:96px; resize:vertical; box-sizing:border-box; margin-top:12px; border:1px solid rgba(255,255,255,.10); border-radius:16px; background:rgba(0,0,0,.28); color:var(--text); padding:14px; font:14px/1.45 'IBM Plex Mono', monospace; }
    .vllm-chat-row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:10px; }
    .vllm-chat-status { color:var(--text-muted); font-family:'IBM Plex Mono'; font-size:12px; }
    @media (max-width:1200px) {
      .shell { grid-template-columns:1fr; }
      aside { position:relative; height:auto; display:flex; align-items:center; justify-content:space-between; }
      nav, .side-footer { display:none; }
      .layout, .services, .kpis { grid-template-columns:1fr; }
      main { padding:22px; }
      h1 { font-size:34px; }
    }
  </style>
</head>
<body>
  <div class="glow"></div>
  <div class="texture"></div>
  <div class="shell">
    <aside>
      <div class="wordmark"><div class="glyph"></div><span>SPARK&nbsp;DASHBOARD</span></div>
      <nav>
        <a class="navitem active" href="#overview">// OVERVIEW</a>
        <a class="navitem" href="#services">// SERVICES</a>
        <a class="navitem" href="#vllm-chat-card">// VLLM CHAT</a>
        <a class="navitem" href="#frontends">// FRONTENDS</a>
        <a class="navitem" href="#benchmarks">// BENCHMARKS</a>
        <a class="navitem" href="#telemetry">// TELEMETRY</a>
      </nav>
      <div class="side-footer">
        <div>NODE: <span id="host">--</span></div>
        <div>REFRESH: <span id="refresh-rate">2.0S</span></div>
        <div>MODE: <span id="profile-mode">SYNCING</span></div>
      </div>
    </aside>

    <main>
      <section class="topbar" id="overview">
        <div>
          <div class="kicker">// LOCAL MODEL CONTROL PLANE</div>
          <h1>Critical service telemetry</h1>
        </div>
        <div class="top-actions">
          <div class="pill"><span id="global-dot" class="dot"></span><span id="global-status">SYNCING</span></div>
          <button class="button ghost" id="refresh-now">REFRESH</button>
        </div>
      </section>

      <section class="grid kpis">
        <div class="card"><div class="label">GPU COMPUTE</div><div class="value good" id="gpu-util">--</div><div class="bar"><span id="gpu-util-bar"></span></div><div class="sub" id="gpu-sub">nvidia-smi</div></div>
        <div class="card"><div class="label">CPU UTILIZATION</div><div class="value" id="cpu-util">--</div><div class="bar info"><span id="cpu-util-bar"></span></div><div class="sub" id="cpu-sub">/proc/stat</div></div>
        <div class="card"><div class="label">UNIFIED MEMORY</div><div class="value info" id="ram-used">--</div><div class="bar info"><span id="ram-bar"></span></div><div class="sub" id="ram-sub">shared CPU/GPU memory</div></div>
        <div class="card"><div class="label">SERVICES ONLINE</div><div class="value good" id="services-online">--</div><div class="sub" id="service-sub">Qwen · PersonaPlex · HiDream · Pixal3D · TripoSplat · Z-Image · Qwen-Image · FLUX.2 · DomainShuttle · Krea-2 · Un-0</div></div>
      </section>

      <section class="layout">
        <div>
          <div class="services" id="services"></div>

          <div class="card" id="control-result-card">
            <div class="panel-title"><h2>Controls</h2><div class="label">FIXED ACTIONS ONLY</div></div>
            <div class="console"><span style="color:var(--accent);">&gt;</span><span id="control-result">Ready. Start/stop buttons call a fixed allow-list on Spark.</span><span class="caret"></span></div>
          </div>

          <div class="card vllm-chat-panel" id="vllm-chat-card">
            <div class="panel-title"><h2>vLLM chat</h2><div class="label" id="vllm-chat-label">TEXT MODEL TOOL</div></div>
            <div class="sub" id="vllm-chat-sub">Open this from a text-model card. The dashboard proxy calls the selected service's OpenAI-compatible /v1/chat/completions endpoint and ignores reasoning fields.</div>
            <div class="vllm-chat-log" id="vllm-chat-log"><div class="sub">No chat selected yet.</div></div>
            <textarea class="vllm-chat-input" id="vllm-chat-input" placeholder="Ask the selected vLLM text model... Shift+Enter for newline, Enter to send."></textarea>
            <div class="vllm-chat-row">
              <button class="small-link" id="vllm-chat-send">SEND</button>
              <button class="small-link" id="vllm-chat-clear">CLEAR</button>
              <span class="vllm-chat-status" id="vllm-chat-status">Idle.</span>
            </div>
          </div>


          <div class="card" id="benchmarks">
            <div class="panel-title"><h2>Benchmark leaderboard</h2><div class="label">LLM SPEED HISTORY</div></div>
            <div class="bench-controls">
              <button class="small-link" id="run-active-benchmark">RUN ACTIVE LLM BENCH</button>
              <button class="small-link" id="refresh-benchmarks">REFRESH LEADERBOARD</button>
              <span class="bench-result" id="benchmark-result">Ready. Benchmarks auto-target the active Qwen/Ornith vLLM service and persist to disk.</span>
            </div>
            <div class="bench-grid">
              <div>
                <div class="label" style="margin-bottom:8px;">LEADERBOARD</div>
                <div id="leaderboard-table" class="sub">No benchmark data yet.</div>
              </div>
              <div>
                <div class="label" style="margin-bottom:8px;">RECENT RUNS</div>
                <div id="benchmark-history" class="sub">No benchmark data yet.</div>
              </div>
            </div>
          </div>

          <div class="card frontend" id="frontends">
            <div class="panel-title"><h2>Frontend dock</h2><div class="label">INTERACTIVE SURFACES</div></div>
            <div class="tabs">
              <button class="tab active" data-frame="none">OVERVIEW</button>
              <button class="tab" data-frame="hidream">HIDREAM UI</button>
              <button class="tab" data-frame="pixal3d">PIXAL3D UI</button>
              <button class="tab" data-frame="zimage">Z-IMAGE UI</button>
              <button class="tab" data-frame="triposplat">TRIPOSPLAT UI</button>
              <button class="tab" data-frame="qwenimage">QWEN-IMAGE UI</button>
              <button class="tab" data-frame="flux2">FLUX.2 UI</button>
              <button class="tab" data-frame="personaplex">PERSONAPLEX UI</button>
              <button class="tab" data-frame="qwen">QWEN DOCS</button>
            </div>
            <div class="frontend-frame" id="frontend-frame">
              <div style="height:100%;display:grid;place-items:center;text-align:center;padding:34px;">
                <div>
                  <div class="console"><span style="color:var(--accent);">&gt;</span><span id="voice-line">Select a frontend tab to load it here.</span><span class="caret"></span></div>
                  <p class="sub" style="max-width:620px;margin:22px auto 0;">PersonaPlex uses a local HTTPS certificate. If the embedded frame is blocked by the browser, use the service card's OPEN UI link to accept the certificate in a full tab.</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="grid" id="telemetry">
          <div class="card orb-card">
            <div class="panel-title"><h2>System core</h2><div class="label">LIVE ORB</div></div>
            <canvas id="orb"></canvas>
            <div class="console"><span style="color:var(--accent);">&gt;</span><span data-voice>SYSTEM ONLINE.</span><span class="caret"></span></div>
          </div>
          <div class="card">
            <div class="panel-title"><h2>Activity</h2><div class="label" id="updated-at">--</div></div>
            <div class="feed" id="feed"></div>
          </div>
          <div class="card">
            <div class="panel-title"><h2>Model residency</h2><div class="label">UNIFIED MEMORY MAP</div></div>
            <div id="gpu-apps"></div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const CONTROL_TOKEN="__CONTROL_TOKEN__";
    let controlBusy = false;
    const serviceRoutes = {
      qwen: {port: 8000, frontend: '/docs', health: '/v1/models'},
      ornith: {port: 8001, frontend: '/docs', health: '/v1/models'},
      mistralmedium: {port: 8002, frontend: '/docs', health: '/v1/models'},
      personaplex: {port: 8998, frontend: '/', health: '/', protocol: 'https:'},
      hidream: {port: 7861, frontend: '/', health: '/'},
      pixal3d: {port: 7863, frontend: '/', health: '/'},
      zimage: {port: 7864, frontend: '/', health: '/health'},
      qwenimage: {port: 7865, frontend: '/', health: '/health'},
      flux2: {port: 7866, frontend: '/', health: '/health'},
      domainshuttle: {port: 7867, frontend: '/', health: '/health'},
      krea2: {port: 7868, frontend: '/', health: '/health'},
      agent3dify: {port: 7869, frontend: '/', health: '/health'},
      un0: {port: 7870, frontend: '/', health: '/health'},
      triposplat: {port: 7871, frontend: '/', health: '/health'}
    };
    function sameHostUrl(key, kind='frontend') {
      const route = serviceRoutes[key];
      if (!route) return '#';
      const url = new URL('/', window.location.href);
      url.protocol = route.protocol || 'http:';
      url.hostname = window.location.hostname;
      url.port = String(route.port);
      url.pathname = route[kind] || route.frontend || '/';
      url.search = '';
      url.hash = '';
      return url.toString();
    }
    const frontends = Object.fromEntries(Object.keys(serviceRoutes).map(k => [k, sameHostUrl(k, 'frontend')]));
    const serviceOrder = ['qwen','ornith','mistralmedium','personaplex','hidream','pixal3d','triposplat','zimage','qwenimage','flux2','domainshuttle','krea2','un0','agent3dify'];
    const accentClass = { qwen:'accent', ornith:'warn', mistralmedium:'warn', personaplex:'info', hidream:'warn', pixal3d:'good', triposplat:'good', zimage:'accent', qwenimage:'warn', flux2:'accent', domainshuttle:'info', krea2:'good', agent3dify:'info', un0:'good' };


    document.querySelectorAll('a.navitem[href^="#"]').forEach(link => {
      link.addEventListener('click', (event) => {
        event.preventDefault();
        const target = document.querySelector(link.getAttribute('href'));
        if (target) target.scrollIntoView({behavior:'smooth', block:'start'});
        document.querySelectorAll('a.navitem').forEach(a => a.classList.toggle('active', a === link));
      });
    });

    function fmtPct(x) { return Number.isFinite(x) ? `${Math.round(x)}%` : '--'; }
    function clamp(min,max,x){ return Math.max(min, Math.min(max, x)); }
    function bytesGiB(mib){ if(!Number.isFinite(mib)) return '--'; return `${(mib/1024).toFixed(1)}G`; }
    function safe(v, fallback='--'){ return (v === undefined || v === null || v === '') ? fallback : v; }

    function setBar(id, pct){ const el=document.getElementById(id); if(el) el.style.width=`${clamp(0,100,pct||0)}%`; }
    function badgeClass(state){ if(state === 'active') return ''; if(state === 'activating') return 'warn'; return 'dead'; }
    function healthText(s){ if(s.health && s.health.ok) return 'ONLINE'; if(s.systemd_state === 'active') return 'NO ENDPOINT'; return 'OFFLINE'; }

    const unloadableServices = new Set(['hidream','zimage','qwenimage','flux2','krea2']);
    const textModelServices = new Set(['qwen','ornith','mistralmedium','nemotronsuper']);

    function serviceShell(key){
      return `<article class="card service-card" id="svc-${key}">
        <div class="service-head">
          <div><h3 class="service-title" data-field="name"></h3><div class="sub" data-field="kind"></div></div>
          <div class="badge" data-field="badge"><span class="dot"></span><span data-field="health"></span></div>
        </div>
        <div class="svc-metrics">
          <div class="mini"><div class="label">systemd</div><div class="value" data-field="systemd"></div></div>
          <div class="mini"><div class="label">Model resident</div><div class="value ${accentClass[key] || ''}" data-field="gpu"></div></div>
          <div class="mini"><div class="label">endpoint</div><div class="value" data-field="endpoint"></div></div>
          <div class="mini"><div class="label" data-field="extra-label"></div><div class="value" style="font-size:15px;" data-field="extra"></div></div>
        </div>
        <div class="links">
          <a class="small-link" data-field="open" target="_blank" rel="noopener">OPEN UI</a>
          <button class="small-link" data-load="${key}">DOCK</button>
          ${textModelServices.has(key) ? `<button class="small-link" data-chat="${key}">VLLM CHAT</button>` : ''}
          <a class="small-link" data-field="health-link" target="_blank" rel="noopener">HEALTH</a>
        </div>
        <div class="links" style="margin-top:10px;">
          <button class="small-link" data-action="start" data-service="${key}">START</button>
          <button class="small-link danger-link" data-action="stop" data-service="${key}">STOP</button>
          ${unloadableServices.has(key) ? `<button class="small-link" data-action="unload" data-service="${key}">UNLOAD MODEL</button>` : ''}
        </div>
        <details class="log-tail">
          <summary>Running log tail</summary>
          <pre data-field="log"></pre>
        </details>
      </article>`;
    }

    function setField(card, name, value){
      const el = card.querySelector(`[data-field="${name}"]`);
      if (el && el.textContent !== String(value)) el.textContent = value;
    }

    function renderServices(data){
      const root=document.getElementById('services');
      if (root.dataset.initialized !== '1') {
        root.innerHTML = serviceOrder.map(serviceShell).join('');
        root.querySelectorAll('[data-load]').forEach(btn => btn.addEventListener('click', () => loadFrame(btn.dataset.load)));
        root.querySelectorAll('[data-chat]').forEach(btn => btn.addEventListener('click', () => openVllmChat(btn.dataset.chat)));
        root.querySelectorAll('[data-action]').forEach(btn => btn.addEventListener('click', () => runControl(btn.dataset.service, btn.dataset.action, btn)));
        root.dataset.initialized = '1';
      }
      serviceOrder.forEach(key => {
        const s = data.services[key] || {};
        const card = document.getElementById(`svc-${key}`);
        if (!card) return;
        const bcls = badgeClass(s.systemd_state);
        const health = healthText(s);
        const endpoint = s.health?.status_code ? `${s.health.status_code}` : safe(s.health?.error, '--');
        const gpu = s.gpu_mib ? bytesGiB(s.gpu_mib) : '--';

        setField(card, 'name', s.name || key);
        setField(card, 'kind', safe(s.kind));
        setField(card, 'health', health);
        setField(card, 'systemd', safe(s.systemd_state).toUpperCase());
        setField(card, 'gpu', gpu);
        setField(card, 'endpoint', endpoint);
        setField(card, 'extra-label', key === 'qwen' ? 'Context' : 'Unit');
        setField(card, 'extra', key === 'qwen' ? safe(s.health?.model_len) : safe(s.unit));
        setField(card, 'log', (s.log_tail && s.log_tail.length) ? s.log_tail.join('\n') : 'No recent journal lines.');

        const badge = card.querySelector('[data-field="badge"]');
        const dot = badge?.querySelector('.dot');
        if (badge) badge.className = `badge ${bcls}`;
        if (dot) dot.className = `dot ${bcls}`;
        const systemdValue = card.querySelector('[data-field="systemd"]');
        if (systemdValue) systemdValue.className = `value ${s.systemd_state==='active'?'good':'warn'}`;
        const open = card.querySelector('[data-field="open"]');
        const healthLink = card.querySelector('[data-field="health-link"]');
        const openUrl = sameHostUrl(key, 'frontend');
        const healthUrl = sameHostUrl(key, 'health');
        if (open && open.href !== openUrl) open.href = openUrl;
        if (healthLink && healthLink.href !== healthUrl) healthLink.href = healthUrl;
      });
    }

    function renderGpuApps(data){
      const root=document.getElementById('gpu-apps');
      const rows = (data.gpu.apps || []).map(app => `<div class="event"><span class="event-tag info">GPU</span><span class="event-text">${app.name}</span><span class="event-time">${bytesGiB(app.used_memory_mib)}</span></div>`).join('');
      root.innerHTML = rows || '<div class="sub">No GPU compute processes reported.</div>';
    }

    function renderFeed(data){
      const root=document.getElementById('feed');
      const events=[];
      for (const key of serviceOrder){
        const s=data.services[key]||{};
        events.push({tag:key.slice(0,2).toUpperCase(), cls: s.systemd_state==='active' ? '' : 'warn', text:`${s.name || key} ${healthText(s).toLowerCase()} · ${safe(s.systemd_state)}`, time:'now'});
      }
      if (data.gpu.utilization_gpu_pct !== null) { const residentPct = data.memory?.total_gib ? (data.gpu.memory_used_mib || 0) / (data.memory.total_gib * 1024) * 100 : null; events.push({tag:'GPU', cls:'info', text:`GPU compute ${fmtPct(data.gpu.utilization_gpu_pct)} · model resident ${Number.isFinite(residentPct) ? fmtPct(residentPct) : bytesGiB(data.gpu.memory_used_mib)}`, time:'live'}); }
      events.push({tag:'SYS', cls:'info', text:`CPU ${fmtPct(data.cpu.utilization_pct)} · RAM ${fmtPct(data.memory.used_pct)} · load ${safe(data.cpu.load_avg?.[0])}`, time:'live'});
      root.innerHTML = events.map(e => `<div class="event"><span class="event-tag ${e.cls}">${e.tag}</span><span class="event-text">${e.text}</span><span class="event-time">${e.time}</span></div>`).join('');
    }



    function sec(v){ return (v === undefined || v === null || v === '') ? '--' : (Number.isFinite(Number(v)) ? `${Number(v).toFixed(3)}s` : '--'); }
    function num2(v){ return (v === undefined || v === null || v === '') ? '--' : (Number.isFinite(Number(v)) ? Number(v).toFixed(2) : '--'); }

    function renderBenchTable(rows, mode){
      if(!rows || !rows.length) return '<div class="sub">No benchmark data yet.</div>';
      if(mode === 'leaderboard'){
        return `<table class="bench-table"><thead><tr><th>#</th><th>Model</th><th>Best tok/s</th><th>Avg tok/s</th><th>Best TTFT</th><th>Avg prefill</th><th>Runs</th><th>Latest</th></tr></thead><tbody>${rows.map((r,i)=>`<tr><td>${i+1}</td><td>${safe(r.model)}</td><td class="metric">${num2(r.best_tokens_per_sec)}</td><td>${num2(r.avg_tokens_per_sec)}</td><td>${sec(r.best_time_to_first_token_sec)}</td><td>${sec(r.avg_prompt_preprocess_sec)}</td><td>${r.runs}</td><td>${new Date((r.latest_timestamp||0)*1000).toLocaleString()}</td></tr>`).join('')}</tbody></table>`;
      }
      return `<table class="bench-table"><thead><tr><th>When</th><th>Service</th><th>Model</th><th>Tok/s</th><th>Decode tok/s</th><th>TTFT</th><th>Prefill</th><th>Queue</th><th>Tokens</th><th>Duration</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${new Date((r.timestamp||0)*1000).toLocaleString()}</td><td>${safe(r.service)}</td><td>${safe(r.model)}</td><td class="metric">${num2(r.tokens_per_sec)}</td><td>${num2(r.decode_tokens_per_sec)}</td><td>${sec(r.time_to_first_token_sec)}</td><td>${sec(r.prompt_preprocess_sec)}</td><td>${sec(r.queue_time_sec)}</td><td>${safe(r.completion_tokens)}</td><td>${Number(r.duration_sec||0).toFixed(2)}s</td></tr>`).join('')}</tbody></table>`;
    }


    let vllmChatService = null;
    let vllmChatMessages = [];

    function appendVllmChat(role, content){
      const log = document.getElementById('vllm-chat-log');
      if (!log) return;
      if (log.querySelector('.sub')) log.innerHTML = '';
      const item = document.createElement('div');
      item.className = 'vllm-chat-msg';
      const r = document.createElement('div');
      r.className = 'vllm-chat-role';
      r.textContent = role;
      const c = document.createElement('div');
      c.className = 'vllm-chat-content';
      c.textContent = content || '[no visible content]';
      item.appendChild(r);
      item.appendChild(c);
      log.appendChild(item);
      log.scrollTop = log.scrollHeight;
    }

    function openVllmChat(service){
      if(!textModelServices.has(service)) return;
      vllmChatService = service;
      vllmChatMessages = [];
      const card = document.getElementById('vllm-chat-card');
      const label = document.getElementById('vllm-chat-label');
      const sub = document.getElementById('vllm-chat-sub');
      const sName = document.querySelector(`#svc-${service} [data-field="name"]`)?.textContent || service;
      if(card) card.classList.add('active');
      if(label) label.textContent = `SERVICE: ${service}`;
      if(sub) sub.textContent = `${sName} via ${sameHostUrl(service, 'health').replace('/v1/models','/v1/chat/completions')} — reasoning fields are ignored.`;
      const log = document.getElementById('vllm-chat-log');
      if(log) log.innerHTML = '<div class="sub">New chat. Visible assistant content only; reasoning fields are discarded.</div>';
      const input = document.getElementById('vllm-chat-input');
      if(input) input.focus();
      card?.scrollIntoView({behavior:'smooth', block:'start'});
    }

    async function sendVllmChat(){
      const input = document.getElementById('vllm-chat-input');
      const status = document.getElementById('vllm-chat-status');
      const btn = document.getElementById('vllm-chat-send');
      if(!vllmChatService){ if(status) status.textContent='Open VLLM CHAT from a text model card first.'; return; }
      const text = (input?.value || '').trim();
      if(!text) return;
      if(input) input.value = '';
      appendVllmChat('you', text);
      const outbound = [...vllmChatMessages, {role:'user', content:text}];
      if(status) status.textContent = 'Waiting for model...';
      if(btn) btn.disabled = true;
      try {
        const res = await fetch('/api/vllm-chat', {
          method:'POST',
          headers:{'Content-Type':'application/json','X-Spark-Control-Token':CONTROL_TOKEN},
          body:JSON.stringify({service:vllmChatService, messages:outbound, max_tokens:4096, temperature:0.2})
        });
        const data = await res.json();
        if(!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
        vllmChatMessages = outbound;
        if(data.content){
          appendVllmChat('assistant', data.content);
          vllmChatMessages.push({role:'assistant', content:data.content});
        } else {
          appendVllmChat('assistant', '[No visible content returned even after the dashboard requested visible final content. Try a shorter prompt or ask for a concise final answer.]');
        }
        if(status) status.textContent = `${data.service_name || data.service} · ${data.model || 'model'} · ${data.usage?.completion_tokens ?? '--'} output tokens`;
      } catch (err) {
        console.error(err);
        appendVllmChat('error', err.message || String(err));
        if(status) status.textContent = `Error: ${err.message || err}`;
      } finally {
        if(btn) btn.disabled = false;
        input?.focus();
      }
    }

    function clearVllmChat(){
      vllmChatMessages = [];
      const log = document.getElementById('vllm-chat-log');
      if(log) log.innerHTML = '<div class="sub">Chat cleared.</div>';
      const status = document.getElementById('vllm-chat-status');
      if(status) status.textContent = 'Idle.';
      document.getElementById('vllm-chat-input')?.focus();
    }

    async function refreshBenchmarks(){
      try {
        const res = await fetch('/api/benchmarks', {cache:'no-store'});
        const data = await res.json();
        document.getElementById('leaderboard-table').innerHTML = renderBenchTable(data.leaderboard || [], 'leaderboard');
        document.getElementById('benchmark-history').innerHTML = renderBenchTable((data.records || []).slice(0,10), 'history');
      } catch (err) {
        console.error(err);
        const el=document.getElementById('benchmark-result');
        if(el) el.textContent = `Benchmark refresh failed: ${err.message || err}`;
      }
    }

    async function runBenchmark(service){
      const label = service === 'active' ? 'the active LLM service (Qwen or Ornith)' : service;
      const ok = window.confirm(`Run benchmark for ${label}?\n\nThis sends a fixed chat-completions request and stores tokens/sec in the local leaderboard.`);
      if(!ok) return;
      const btn=document.getElementById(service === 'active' ? 'run-active-benchmark' : `run-${service}-benchmark`);
      const result=document.getElementById('benchmark-result');
      if(btn) btn.disabled=true;
      if(result) result.textContent = `Benchmarking ${label}; waiting for model output...`;
      try {
        const res = await fetch('/api/benchmark', {
          method:'POST',
          headers:{'Content-Type':'application/json','X-Spark-Control-Token':CONTROL_TOKEN},
          body:JSON.stringify({service})
        });
        const data = await res.json();
        if(!res.ok || !data.ok) throw new Error(data.error || data.stderr || `HTTP ${res.status}`);
        if(result) result.textContent = `${data.service_name || data.service} / ${data.model}: ${num2(data.tokens_per_sec)} tok/s · TTFT ${sec(data.time_to_first_token_sec)} · prefill ${sec(data.prompt_preprocess_sec)} · ${data.completion_tokens} tokens in ${Number(data.duration_sec).toFixed(2)}s`;
        await refreshBenchmarks();
      } catch (err) {
        console.error(err);
        if(result) result.textContent = `Benchmark failed: ${err.message || err}`;
      } finally {
        if(btn) btn.disabled=false;
      }
    }

    async function runControl(service, action, btn){
      if (controlBusy) return;
      const label = `${action.toUpperCase()} ${service.replace('_','-')}`;
      const ok = window.confirm(`${label}?\n\nThis will run the fixed Spark dashboard action for this service/profile.`);
      if (!ok) return;
      controlBusy = true;
      document.querySelectorAll('[data-action]').forEach(b => b.disabled = true);
      const result = document.getElementById('control-result');
      if (result) result.textContent = `${label} requested; waiting for Spark...`;
      try {
        const res = await fetch('/api/control', {
          method: 'POST',
          headers: {'Content-Type':'application/json', 'X-Spark-Control-Token': CONTROL_TOKEN},
          body: JSON.stringify({service, action})
        });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || data.stderr || `HTTP ${res.status}`);
        const tail = (data.stdout || '').trim().split('\n').slice(-2).join(' · ');
        if (result) result.textContent = `${data.label}: OK${tail ? ' · ' + tail : ''}`;
      } catch (err) {
        console.error(err);
        if (result) result.textContent = `${label} failed: ${err.message || err}`;
      } finally {
        controlBusy = false;
        document.querySelectorAll('[data-action]').forEach(b => b.disabled = false);
        refresh();
      }
    }

    async function refresh(){
      try {
        const res = await fetch('/api/status', {cache:'no-store'});
        const data = await res.json();
        window.lastStatus = data;
        document.getElementById('host').textContent = data.host || 'spark';
        document.getElementById('updated-at').textContent = new Date(data.timestamp * 1000).toLocaleTimeString();
        const online = serviceOrder.filter(k => data.services[k]?.systemd_state === 'active' && data.services[k]?.health?.ok).length;
        document.getElementById('services-online').textContent = `${online}/${serviceOrder.length}`;
        const qwenLen = data.services.qwen?.health?.model_len || 0;
        const highContext = qwenLen >= 262144;
        const mode = highContext ? 'HIGH CONTEXT' : 'COEXISTENCE';
        const modeEl = document.getElementById('profile-mode');
        if (modeEl) modeEl.textContent = mode;
        const reserve = data.memory?.available_gib;
        document.getElementById('service-sub').textContent = highContext ? (online === serviceOrder.length ? `262k qwen + image/voice online; reserve ${reserve?.toFixed ? reserve.toFixed(1) : reserve}GiB` : `qwen context ${qwenLen}; active ${online}/${serviceOrder.length}; reserve ${reserve?.toFixed ? reserve.toFixed(1) : reserve}GiB`) : (online === serviceOrder.length ? 'all service endpoints nominal' : 'attention required');
        const gd = document.getElementById('global-dot');
        gd.className = `dot ${online===serviceOrder.length?'':online>0?'warn':'danger'}`;
        document.getElementById('global-status').textContent = highContext && data.services.qwen?.systemd_state === 'active' ? 'QWEN HIGH-CONTEXT ONLINE' : (online===serviceOrder.length ? 'ALL SYSTEMS ONLINE' : `${online}/${serviceOrder.length} SYSTEMS ONLINE`);

        document.getElementById('gpu-util').textContent = fmtPct(data.gpu.utilization_gpu_pct);
        const residentPct = data.memory?.total_gib ? (data.gpu.memory_used_mib || 0) / (data.memory.total_gib * 1024) * 100 : null;
        document.getElementById('gpu-sub').textContent = `${bytesGiB(data.gpu.memory_used_mib)} model-resident · ${Number.isFinite(residentPct) ? Math.round(residentPct) + '% of unified' : 'unified memory'}`;
        setBar('gpu-util-bar', data.gpu.utilization_gpu_pct || 0);
        document.getElementById('cpu-util').textContent = fmtPct(data.cpu.utilization_pct);
        document.getElementById('cpu-sub').textContent = `load ${data.cpu.load_avg ? data.cpu.load_avg.join(' · ') : '--'}`;
        setBar('cpu-util-bar', data.cpu.utilization_pct || 0);
        document.getElementById('ram-used').textContent = fmtPct(data.memory.used_pct);
        document.getElementById('ram-sub').textContent = `${data.memory.used_gib?.toFixed(1)}G / ${data.memory.total_gib?.toFixed(1)}G unified used`;
        setBar('ram-bar', data.memory.used_pct || 0);
        renderServices(data);
        renderGpuApps(data);
        renderFeed(data);
        updateOrbEnergy(online, data.gpu.utilization_gpu_pct || 0);
      } catch (err) {
        document.getElementById('global-status').textContent = 'DASHBOARD API OFFLINE';
        document.getElementById('global-dot').className = 'dot danger';
        console.error(err);
      }
    }

    function iframeHtml(key, url){
      return `<iframe src="${url}" title="${key} frontend" allow="microphone; camera; autoplay; clipboard-read; clipboard-write"></iframe>`;
    }

    function loadPersonaPlexDock(){
      const root=document.getElementById('frontend-frame');
      root.innerHTML = iframeHtml('personaplex', frontends.personaplex);
    }

    function loadFrame(key){
      document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.frame === key));
      const root=document.getElementById('frontend-frame');
      if(key === 'none') { root.innerHTML = '<div style="height:100%;display:grid;place-items:center;text-align:center;padding:34px;"><div><div class="console"><span style="color:var(--accent);">&gt;</span><span>Select a frontend tab to load it here.</span><span class="caret"></span></div><p class="sub" style="max-width:620px;margin:22px auto 0;">Use OPEN UI if browser certificate or framing policy blocks an embedded surface.</p></div></div>'; return; }
      const url = frontends[key];
      root.innerHTML = iframeHtml(key, url);
    }
    document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => loadFrame(tab.dataset.frame)));
    document.getElementById('refresh-now').addEventListener('click', refresh);
    document.getElementById('vllm-chat-send')?.addEventListener('click', sendVllmChat);
    document.getElementById('vllm-chat-clear')?.addEventListener('click', clearVllmChat);
    document.getElementById('vllm-chat-input')?.addEventListener('keydown', (event) => { if(event.key === 'Enter' && !event.shiftKey){ event.preventDefault(); sendVllmChat(); } });
    document.getElementById('run-active-benchmark')?.addEventListener('click', () => runBenchmark('active'));
    document.getElementById('refresh-benchmarks')?.addEventListener('click', refreshBenchmarks);

    const voiceLines = ['SYSTEM ONLINE.', 'Coexistence profile nominal.', 'Monitoring Qwen, PersonaPlex, HiDream, Pixal3D, Z-Image, Qwen-Image.', 'Ollama offline. Unified memory map stable.'];
    let vi=0, vc=0, deleting=false;
    function typeLoop(){
      const el=document.querySelector('[data-voice]');
      const line=voiceLines[vi];
      if(!deleting){ vc++; el.textContent=line.slice(0,vc); if(vc>=line.length){ deleting=true; setTimeout(typeLoop,1900); return; } }
      else { vc-=2; el.textContent=line.slice(0,Math.max(0,vc)); if(vc<=0){ deleting=false; vi=(vi+1)%voiceLines.length; } }
      setTimeout(typeLoop, deleting ? 18 : 36 + Math.random()*42);
    }

    let orbEnergy = .55;
    function updateOrbEnergy(online, gpu){ orbEnergy = clamp(.25, 1.25, .35 + online*.18 + gpu/180); }
    function startOrb(){
      const c=document.getElementById('orb'); const ctx=c.getContext('2d'); let raf; let t=0; let mx=0,my=0;
      function resize(){ const r=c.getBoundingClientRect(); const d=Math.min(devicePixelRatio||1,2); c.width=Math.max(1,Math.floor(r.width*d)); c.height=Math.max(1,Math.floor(r.height*d)); ctx.setTransform(d,0,0,d,0,0); }
      window.addEventListener('resize', resize); resize();
      c.addEventListener('mousemove', e=>{ const r=c.getBoundingClientRect(); mx=(e.clientX-r.left-r.width/2)/r.width; my=(e.clientY-r.top-r.height/2)/r.height; });
      c.addEventListener('mouseleave', ()=>{mx=0;my=0;});
      function draw(){
        const w=c.clientWidth,h=c.clientHeight; ctx.clearRect(0,0,w,h); t+=0.012;
        const cx=w/2+mx*12, cy=h/2+my*9, R=Math.min(w,h)*0.28;
        ctx.globalCompositeOperation='lighter';
        const g=ctx.createRadialGradient(cx,cy,0,cx,cy,R*2.3); g.addColorStop(0,`rgba(95,227,160,${0.24*orbEnergy})`); g.addColorStop(.45,`rgba(95,227,160,${0.08*orbEnergy})`); g.addColorStop(1,'rgba(95,227,160,0)'); ctx.fillStyle=g; ctx.beginPath(); ctx.arc(cx,cy,R*2.35,0,Math.PI*2); ctx.fill();
        for(let ring of [1,.74]){ ctx.beginPath(); for(let i=0;i<220;i++){ const a=i/220*Math.PI*2; const wave=Math.sin(a*7+t*5)*4*orbEnergy+Math.sin(a*13-t*2)*2; const rr=R*ring+wave; const x=cx+Math.cos(a)*rr, y=cy+Math.sin(a)*rr; i?ctx.lineTo(x,y):ctx.moveTo(x,y); } ctx.closePath(); ctx.strokeStyle=`rgba(95,227,160,${ring===1?.36:.22})`; ctx.lineWidth=1; ctx.stroke(); }
        for(let i=0;i<88;i++){ const a=i/88*Math.PI*2+t*.35; const amp=(Math.sin(i*.47+t*3)+1)*.5; const r1=R*1.08, r2=R*(1.14+amp*.22*orbEnergy); ctx.beginPath(); ctx.moveTo(cx+Math.cos(a)*r1,cy+Math.sin(a)*r1); ctx.lineTo(cx+Math.cos(a)*r2,cy+Math.sin(a)*r2); ctx.strokeStyle=`rgba(95,227,160,${.08+amp*.38})`; ctx.lineWidth=1; ctx.stroke(); }
        for(let i=0;i<72;i++){ const a=i/72*Math.PI*2-t*.22; const len=i%6===0?10:5; ctx.beginPath(); ctx.moveTo(cx+Math.cos(a)*(R*1.62),cy+Math.sin(a)*(R*1.62)); ctx.lineTo(cx+Math.cos(a)*(R*1.62+len),cy+Math.sin(a)*(R*1.62+len)); ctx.strokeStyle='rgba(236,234,246,.16)'; ctx.stroke(); }
        ctx.beginPath(); ctx.arc(cx,cy,R*.42,0,Math.PI*2); ctx.fillStyle=`rgba(95,227,160,${.11+.04*Math.sin(t*5)})`; ctx.fill(); ctx.strokeStyle='rgba(95,227,160,.65)'; ctx.stroke();
        ctx.globalCompositeOperation='source-over'; raf=requestAnimationFrame(draw);
      }
      draw(); document.addEventListener('visibilitychange', ()=>{ if(!document.hidden){ cancelAnimationFrame(raf); draw(); }});
    }

    startOrb(); typeLoop(); refresh(); refreshBenchmarks(); setInterval(refresh, 2000); setInterval(refreshBenchmarks, 15000);
  </script>
</body>
</html>'''


def run(cmd: list[str], timeout: float = 3.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def http_probe(url: str, timeout: float = 2.5) -> dict[str, Any]:
    ctx = None
    if url.startswith("https://"):
        ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "spark-dashboard/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            body = r.read(64_000)
            out: dict[str, Any] = {"ok": 200 <= r.status < 400, "status_code": r.status, "content_type": r.headers.get("content-type", "")}
            if url.endswith("/v1/models"):
                try:
                    data = json.loads(body.decode("utf-8", errors="ignore"))
                    if data.get("data"):
                        out["model_id"] = data["data"][0].get("id")
                        out["model_len"] = data["data"][0].get("max_model_len")
                except Exception:
                    pass
            return out
    except urllib.error.HTTPError as e:
        return {"ok": False, "status_code": e.code, "error": str(e)}
    except Exception as e:
        return {"ok": False, "status_code": None, "error": str(e)[:160]}


def int_or_zero(value: str | None) -> int:
    try:
        if value is None or value == "" or value == "[not set]":
            return 0
        return int(value)
    except Exception:
        return 0


def systemd_state(unit: str, user: bool = True) -> dict[str, Any]:
    base = ["systemctl"] + (["--user"] if user else [])
    rc, active = run(base + ["is-active", unit], timeout=2)
    rc2, enabled = run(base + ["is-enabled", unit], timeout=2)
    rc3, show = run(base + ["show", unit, "--property=MainPID,ActiveEnterTimestamp,MemoryCurrent,CPUUsageNSec"], timeout=2)
    fields: dict[str, str] = {}
    for line in show.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k] = v
    return {
        "active": active if rc == 0 else active or "inactive",
        "enabled": enabled if rc2 == 0 else enabled or "unknown",
        "main_pid": int(fields.get("MainPID") or 0),
        "active_enter": fields.get("ActiveEnterTimestamp") or "",
        "memory_current": int_or_zero(fields.get("MemoryCurrent")),
        "cpu_nsec": int_or_zero(fields.get("CPUUsageNSec")),
    }



def journal_tail(unit: str, lines: int = 8) -> list[str]:
    now = time.time()
    cached = _LOG_CACHE.get(unit)
    if cached and now - cached[0] < LOG_CACHE_TTL_SEC:
        return cached[1]
    rc, out = run([
        "journalctl", "--user", "-u", unit,
        "-n", str(lines), "--no-pager", "--output=short-iso",
    ], timeout=2.5)
    if rc != 0 and not out:
        result = ["journal unavailable"]
    else:
        result = []
        for raw in out.splitlines()[-lines:]:
            line = re.sub(r"\s+", " ", raw).strip()
            if len(line) > 260:
                line = line[:257] + "…"
            if line:
                result.append(line)
    _LOG_CACHE[unit] = (now, result)
    return result

def parse_free() -> dict[str, Any]:
    rc, out = run(["free", "-m"], timeout=2)
    total = used = avail = None
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            total = float(parts[1])
            used = float(parts[2])
            avail = float(parts[6]) if len(parts) > 6 else None
            break
    if not total:
        return {"total_gib": None, "used_gib": None, "available_gib": None, "used_pct": None}
    return {
        "total_gib": total / 1024,
        "used_gib": used / 1024,
        "available_gib": (avail / 1024) if avail is not None else None,
        "used_pct": round(used / total * 100, 1),
    }


def read_cpu_times() -> list[int]:
    first = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    return [int(x) for x in first]


def cpu_usage() -> dict[str, Any]:
    try:
        a = read_cpu_times()
        time.sleep(0.08)
        b = read_cpu_times()
        idle_a = a[3] + (a[4] if len(a) > 4 else 0)
        idle_b = b[3] + (b[4] if len(b) > 4 else 0)
        total_a = sum(a)
        total_b = sum(b)
        total = total_b - total_a
        idle = idle_b - idle_a
        pct = 0.0 if total <= 0 else (1 - idle / total) * 100
    except Exception:
        pct = None
    try:
        load = [round(float(x), 2) for x in os.getloadavg()]
    except Exception:
        load = None
    return {"utilization_pct": round(pct, 1) if pct is not None else None, "load_avg": load, "cores": os.cpu_count()}


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "utilization_gpu_pct": None,
        "utilization_memory_pct": None,
        "memory_total_mib": None,
        "memory_used_mib": None,
        "memory_free_mib": None,
        "temperature_c": None,
        "power_w": None,
        "apps": [],
    }
    rc, out = run([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ], timeout=3)
    if rc == 0 and out:
        vals = [x.strip() for x in out.splitlines()[0].split(",")]
        keys = ["utilization_gpu_pct", "utilization_memory_pct", "temperature_c", "power_w", "memory_total_mib", "memory_used_mib", "memory_free_mib"]
        for k, v in zip(keys, vals):
            if v and v != "[N/A]" and v.upper() != "N/A":
                try:
                    info[k] = float(v)
                except Exception:
                    pass
    rc, out = run([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], timeout=3)
    apps = []
    if rc == 0:
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3 and parts[0] and parts[0] != "[N/A]":
                try:
                    mib = float(parts[2])
                except Exception:
                    mib = None
                apps.append({"pid": int(parts[0]), "name": parts[1], "used_memory_mib": mib})
    info["apps"] = apps
    if info["memory_used_mib"] is None and apps:
        total_used = sum(a["used_memory_mib"] or 0 for a in apps)
        info["memory_used_mib"] = total_used
    if info["memory_total_mib"] and info["memory_used_mib"] is not None:
        info["memory_used_pct"] = round(info["memory_used_mib"] / info["memory_total_mib"] * 100, 1)
    else:
        info["memory_used_pct"] = None
    return info


def service_gpu_mib(service: dict[str, Any], apps: list[dict[str, Any]], active_state: str | None = None) -> float | None:
    hints = [h.lower() for h in service.get("process_hints", [])]
    total = 0.0
    for app in apps:
        name = app.get("name", "").lower()
        if any(h in name for h in hints):
            total += app.get("used_memory_mib") or 0
    # vLLM engine processes appear only as VLLM::EngineCore in nvidia-smi.
    # Assign generic VLLM::EngineCore residency to active Qwen when Qwen is the vLLM tenant.
    if service["key"] == "qwen" and total == 0 and active_state == "active":
        for app in apps:
            if "vllm::enginecore" in app.get("name", "").lower():
                total += app.get("used_memory_mib") or 0
    return total or None



def read_benchmark_records(limit: int = BENCHMARK_MAX_RECORDS) -> list[dict[str, Any]]:
    if not BENCHMARK_PATH.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line in BENCHMARK_PATH.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return []
    records.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
    return records[:limit]


def benchmark_leaderboard(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        model = str(r.get("model") or "unknown")
        grouped.setdefault(model, []).append(r)
    rows = []
    for model, items in grouped.items():
        speeds = [float(x.get("tokens_per_sec") or 0) for x in items if x.get("tokens_per_sec")]
        ttfts = [float(x.get("time_to_first_token_sec")) for x in items if x.get("time_to_first_token_sec") is not None]
        prefills = [float(x.get("prompt_preprocess_sec")) for x in items if x.get("prompt_preprocess_sec") is not None]
        queues = [float(x.get("queue_time_sec")) for x in items if x.get("queue_time_sec") is not None]
        if not speeds:
            continue
        latest = max(items, key=lambda x: x.get("timestamp", 0))
        rows.append({
            "model": model,
            "runs": len(speeds),
            "best_tokens_per_sec": max(speeds),
            "avg_tokens_per_sec": sum(speeds) / len(speeds),
            "latest_tokens_per_sec": float(latest.get("tokens_per_sec") or 0),
            "best_time_to_first_token_sec": min(ttfts) if ttfts else None,
            "avg_time_to_first_token_sec": (sum(ttfts) / len(ttfts)) if ttfts else None,
            "avg_prompt_preprocess_sec": (sum(prefills) / len(prefills)) if prefills else None,
            "avg_queue_time_sec": (sum(queues) / len(queues)) if queues else None,
            "latest_time_to_first_token_sec": latest.get("time_to_first_token_sec"),
            "latest_prompt_preprocess_sec": latest.get("prompt_preprocess_sec"),
            "latest_timestamp": latest.get("timestamp", 0),
            "latest_service": latest.get("service"),
        })
    rows.sort(key=lambda r: r["best_tokens_per_sec"], reverse=True)
    return rows



def metrics_url_for_service(svc: dict[str, Any]) -> str:
    return svc["health_url"].replace("/v1/models", "/metrics")


def metric_snapshot(metrics_url: str, model_id: str) -> dict[str, float]:
    wanted = [
        "vllm:request_prefill_time_seconds",
        "vllm:request_queue_time_seconds",
        "vllm:time_to_first_token_seconds",
    ]
    out = {f"{name}_sum": 0.0 for name in wanted}
    out.update({f"{name}_count": 0.0 for name in wanted})
    try:
        req = urllib.request.Request(metrics_url, method="GET", headers={"User-Agent": "spark-dashboard-benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            text = r.read(2_000_000).decode("utf-8", errors="ignore")
    except Exception:
        return out
    for raw in text.splitlines():
        if not raw or raw.startswith("#"):
            continue
        if "model_name=\"" in raw and f'model_name="{model_id}"' not in raw:
            continue
        try:
            lhs, val = raw.rsplit(" ", 1)
            value = float(val)
        except Exception:
            continue
        for name in wanted:
            for suffix in ("_sum", "_count"):
                key = f"{name}{suffix}"
                if lhs.startswith(key + "{") or lhs == key:
                    out[key] += value
    return out


def metric_avg_delta(before: dict[str, float], after: dict[str, float], metric: str) -> tuple[float | None, float]:
    count_delta = (after.get(metric + "_count") or 0.0) - (before.get(metric + "_count") or 0.0)
    sum_delta = (after.get(metric + "_sum") or 0.0) - (before.get(metric + "_sum") or 0.0)
    if count_delta <= 0:
        return None, count_delta
    return max(0.0, sum_delta / count_delta), count_delta


def resolve_benchmark_service(service_key: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    requested = (service_key or "active").strip().lower()
    if requested in ("active", "current", "auto"):
        healthy: list[tuple[dict[str, Any], dict[str, Any]]] = []
        unhealthy: list[str] = []
        for key in LLM_BENCHMARK_SERVICE_KEYS:
            svc = next((x for x in SERVICES if x["key"] == key), None)
            if not svc:
                continue
            models = http_probe(svc["health_url"], timeout=5)
            if models.get("ok") and models.get("model_id"):
                healthy.append((svc, models))
            else:
                unhealthy.append(f"{key}: {models.get('error') or models.get('status') or 'offline'}")
        if len(healthy) == 1:
            svc, models = healthy[0]
            return svc, models, svc["key"]
        if not healthy:
            raise RuntimeError("no active benchmarkable LLM service found; checked " + "; ".join(unhealthy))
        names = ", ".join(f"{svc['key']}={models.get('model_id')}" for svc, models in healthy)
        raise RuntimeError("multiple active LLM services found; use an explicit service key: " + names)
    if requested not in LLM_BENCHMARK_SERVICE_KEYS:
        raise ValueError("benchmark service must be active, qwen, ornith, or mistralmedium")
    svc = next((x for x in SERVICES if x["key"] == requested), None)
    if not svc:
        raise ValueError("unknown service")
    models = http_probe(svc["health_url"], timeout=5)
    if not models.get("ok") or not models.get("model_id"):
        raise RuntimeError(f"service {requested} is not healthy: {models}")
    return svc, models, requested




def ensure_visible_answer_messages(service_key: str, messages: list[dict[str, str]]) -> None:
    """Coax reasoning-capable text models to put the final answer in visible content.

    The dashboard deliberately ignores reasoning fields. Nemotron/Qwen-family models can
    otherwise spend the entire token budget in hidden reasoning and return empty content.
    """
    if service_key not in ("qwen", "ornith", "nemotronsuper"):
        return
    prefix = (
        "/no_think\n"
        "Answer directly in visible final assistant content. "
        "Do not leave the final answer only in hidden reasoning.\n\n"
    )
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") != "user":
            continue
        content = messages[idx].get("content") or ""
        if not content.lstrip().lower().startswith("/no_think"):
            messages[idx] = {**messages[idx], "content": prefix + content}
        return

def run_vllm_chat(payload: dict[str, Any]) -> dict[str, Any]:
    service_key = str(payload.get("service", "")).strip().lower()
    if service_key not in LLM_BENCHMARK_SERVICE_KEYS:
        raise ValueError("vLLM chat is only available for text model services: " + ", ".join(LLM_BENCHMARK_SERVICE_KEYS))
    svc = next((x for x in SERVICES if x["key"] == service_key), None)
    if not svc:
        raise ValueError("unknown service")
    models = http_probe(svc["health_url"], timeout=5)
    if not models.get("ok") or not models.get("model_id"):
        raise RuntimeError(f"service {service_key} is not healthy: {models}")
    model_id = str(payload.get("model") or models["model_id"])
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list")
    messages: list[dict[str, str]] = []
    for raw in raw_messages[-24:]:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role", "user"))
        if role not in ("system", "user", "assistant"):
            role = "user"
        content = str(raw.get("content", ""))[:12000]
        if content:
            messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("messages contained no content")
    ensure_visible_answer_messages(service_key, messages)
    try:
        max_tokens = int(payload.get("max_tokens") or 2048)
    except Exception:
        max_tokens = 2048
    max_tokens = max(1, min(max_tokens, 8192))
    try:
        temperature = float(payload.get("temperature", 0.2))
    except Exception:
        temperature = 0.2
    temperature = max(0.0, min(temperature, 2.0))
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if service_key in ("qwen", "ornith"):
        body["chat_template_kwargs"] = {"enable_thinking": False}
    chat_url = svc["health_url"].replace("/v1/models", "/v1/chat/completions")
    req = urllib.request.Request(
        chat_url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "spark-dashboard-vllm-chat/1.0"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            response = json.loads(r.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:4000]
        raise RuntimeError(f"vLLM HTTP {e.code}: {detail}") from e
    message = ((response.get("choices") or [{}])[0].get("message") or {})
    content = message.get("content") or ""
    return {
        "ok": True,
        "service": service_key,
        "service_name": svc["name"],
        "model": model_id,
        "model_len": models.get("model_len"),
        "content": content,
        "ignored_reasoning": bool(message.get("reasoning")),
        "usage": response.get("usage") or {},
        "duration_sec": round(time.time() - started, 3),
    }

def run_llm_benchmark(service_key: str) -> dict[str, Any]:
    svc, models, service_key = resolve_benchmark_service(service_key)
    model_id = str(models["model_id"])
    chat_url = svc["health_url"].replace("/v1/models", "/v1/chat/completions")
    metrics_url = metrics_url_for_service(svc)
    metrics_before = metric_snapshot(metrics_url, model_id)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": BENCHMARK_PROMPT}],
        "max_tokens": BENCHMARK_MAX_TOKENS,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # Qwen/Ornith use Qwen chat templates where hidden thinking should be disabled.
    # Mistral tokenizers reject chat_template_kwargs with HTTP 400.
    if service_key in ("qwen", "ornith"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(chat_url, data=json.dumps(payload).encode("utf-8"), method="POST", headers={"Content-Type": "application/json", "Accept": "text/event-stream", "User-Agent": "spark-dashboard-benchmark/1.0"})
    started = time.time()
    first_chunk_time: float | None = None
    first_token_time: float | None = None
    usage: dict[str, Any] = {}
    text_parts: list[str] = []
    chunk_count = 0
    with urllib.request.urlopen(req, timeout=240) as r:
        for raw_line in r:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if payload_text == "[DONE]":
                break
            now = time.time()
            if first_chunk_time is None:
                first_chunk_time = now
            try:
                event = json.loads(payload_text)
            except Exception:
                continue
            chunk_count += 1
            if event.get("usage"):
                usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning") or ""
                if piece:
                    if first_token_time is None:
                        first_token_time = now
                    text_parts.append(str(piece))
    duration = max(time.time() - started, 0.001)
    # Give vLLM a brief moment to flush Prometheus counters for the completed request.
    time.sleep(0.15)
    metrics_after = metric_snapshot(metrics_url, model_id)
    prefill_avg, prefill_samples = metric_avg_delta(metrics_before, metrics_after, "vllm:request_prefill_time_seconds")
    queue_avg, queue_samples = metric_avg_delta(metrics_before, metrics_after, "vllm:request_queue_time_seconds")
    vllm_ttft_avg, ttft_samples = metric_avg_delta(metrics_before, metrics_after, "vllm:time_to_first_token_seconds")
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (completion_tokens + prompt_tokens))
    if completion_tokens <= 0:
        completion_tokens = max(1, len("".join(text_parts).split()))
    client_first_chunk_sec = (first_chunk_time - started) if first_chunk_time else None
    client_ttft_sec = (first_token_time - started) if first_token_time else client_first_chunk_sec
    decode_window = max(duration - (client_ttft_sec or 0), 0.001)
    decode_tokens = max(completion_tokens - 1, 1)
    memory = parse_free()
    gpu = gpu_info()
    record = {
        "timestamp": started,
        "timestamp_iso": datetime.fromtimestamp(started).isoformat(),
        "service": service_key,
        "service_name": svc["name"],
        "model": model_id,
        "model_len": models.get("model_len"),
        "duration_sec": round(duration, 4),
        "client_first_chunk_sec": round(client_first_chunk_sec, 4) if client_first_chunk_sec is not None else None,
        "time_to_first_token_sec": round(client_ttft_sec, 4) if client_ttft_sec is not None else None,
        "vllm_time_to_first_token_sec": round(vllm_ttft_avg, 4) if vllm_ttft_avg is not None else None,
        "prompt_preprocess_sec": round(prefill_avg, 4) if prefill_avg is not None else None,
        "queue_time_sec": round(queue_avg, 4) if queue_avg is not None else None,
        "metrics_samples": {
            "prefill": prefill_samples,
            "queue": queue_samples,
            "ttft": ttft_samples,
        },
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": round(completion_tokens / duration, 4),
        "decode_tokens_per_sec": round(decode_tokens / decode_window, 4),
        "stream_chunks": chunk_count,
        "prompt_name": "spark-fixed-technical-note-v2-nothink",
        "max_tokens": BENCHMARK_MAX_TOKENS,
        "temperature": 0.2,
        "memory_available_gib": memory.get("available_gib"),
        "gpu_compute_pct": gpu.get("utilization_gpu_pct"),
        "model_resident_mib": service_gpu_mib(svc, gpu.get("apps", []), "active"),
    }
    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_PATH.open("a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
    return {"ok": True, **record}

def collect_status() -> dict[str, Any]:
    gpu = gpu_info()
    services: dict[str, Any] = {}
    for svc in SERVICES:
        state = systemd_state(svc["unit"], user=True)
        services[svc["key"]] = {
            "name": svc["name"],
            "key": svc["key"],
            "unit": svc["unit"],
            "kind": svc["kind"],
            "public_url": svc["public_url"],
            "frontend_url": svc["frontend_url"],
            "systemd_state": state["active"],
            "enabled": state["enabled"],
            "main_pid": state["main_pid"],
            "active_enter": state["active_enter"],
            "memory_current_bytes": state["memory_current"],
            "health": http_probe(svc["health_url"]),
            "gpu_mib": service_gpu_mib(svc, gpu.get("apps", []), state["active"]),
            "log_tail": journal_tail(svc["unit"]),
        }
    return {
        "timestamp": time.time(),
        "host": os.uname().nodename,
        "public_host": PUBLIC_HOST,
        "cpu": cpu_usage(),
        "memory": parse_free(),
        "gpu": gpu,
        "services": services,
    }




def control_token() -> str:
    try:
        if CONTROL_TOKEN_PATH.exists():
            token = CONTROL_TOKEN_PATH.read_text().strip()
            if token:
                return token
        token = secrets.token_urlsafe(24)
        CONTROL_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONTROL_TOKEN_PATH.write_text(token + "\n")
        try:
            CONTROL_TOKEN_PATH.chmod(0o600)
        except OSError:
            pass
        return token
    except Exception:
        if not hasattr(control_token, "_fallback"):
            setattr(control_token, "_fallback", secrets.token_urlsafe(24))
        return getattr(control_token, "_fallback")


def run_control_action(service: str, action: str) -> dict[str, Any]:
    spec = CONTROL_ACTIONS.get((service, action))
    if not spec:
        raise ValueError(f"unsupported action: {service}/{action}")
    started = time.time()
    proc = subprocess.run(
        spec["cmd"],
        text=True,
        capture_output=True,
        timeout=spec.get("timeout", 300),
    )
    return {
        "ok": proc.returncode == 0,
        "service": service,
        "action": action,
        "label": spec["label"],
        "returncode": proc.returncode,
        "duration_sec": round(time.time() - started, 2),
        "stdout": (proc.stdout or "")[-6000:],
        "stderr": (proc.stderr or "")[-6000:],
    }

class Handler(BaseHTTPRequestHandler):
    server_version = "SparkDashboard/1.0"

    def clean_path(self) -> str:
        return urllib.parse.urlsplit(self.path).path

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self) -> None:
        body = HTML.replace("__CONTROL_TOKEN__", control_token()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_svg(self, svg: str) -> None:
        body = svg.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
        self.send_header("Cache-Control", "public, max-age=86400")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self) -> None:
        path = self.clean_path()
        if path in ("/", "/index.html", "/health", "/api/status", "/api/benchmarks", "/favicon.svg"):
            self.send_response(200)
            ctype = "image/svg+xml; charset=utf-8" if path == "/favicon.svg" else ("application/json; charset=utf-8" if path.startswith("/api") or path == "/health" else "text/html; charset=utf-8")
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        path = self.clean_path()
        if path not in ("/api/control", "/api/benchmark", "/api/vllm-chat"):
            self.send_error(404)
            return
        if self.headers.get("X-Spark-Control-Token", "") != control_token():
            self.send_json({"ok": False, "error": "forbidden"}, status=403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            payload = json.loads(self.rfile.read(min(length, 65536)) or b"{}")
            if path == "/api/control":
                service = str(payload.get("service", ""))
                action = str(payload.get("action", ""))
                result = run_control_action(service, action)
                status = 200 if result.get("ok") else 500
                self.send_json(result, status=status)
            elif path == "/api/benchmark":
                service = str(payload.get("service", "qwen"))
                self.send_json(run_llm_benchmark(service), status=200)
            else:
                self.send_json(run_vllm_chat(payload), status=200)
        except subprocess.TimeoutExpired as e:
            self.send_json({"ok": False, "error": "action timed out", "stdout": e.stdout or "", "stderr": e.stderr or ""}, status=504)
        except Exception as e:
            self.send_json({"ok": False, "error": str(e)}, status=400)

    def do_GET(self) -> None:
        path = self.clean_path()
        if path in ("/", "/index.html"):
            self.send_html()
        elif path == "/favicon.svg":
            self.send_svg(FAVICON_SVG)
        elif path == "/api/status":
            self.send_json(collect_status())
        elif path == "/api/benchmarks":
            records = read_benchmark_records()
            self.send_json({"records": records, "leaderboard": benchmark_leaderboard(records), "path": str(BENCHMARK_PATH)})
        elif path == "/health":
            self.send_json({"ok": True, "service": "spark-dashboard", "timestamp": time.time()})
        else:
            self.send_json({"error": "not found"}, status=404)


def main() -> None:
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Spark dashboard listening on http://{HOST}:{PORT}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
