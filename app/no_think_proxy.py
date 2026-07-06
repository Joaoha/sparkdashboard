#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = os.environ.get("VLLM_PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("VLLM_PROXY_PORT", "8011"))
UPSTREAM = os.environ.get("VLLM_PROXY_UPSTREAM", "http://127.0.0.1:8001").rstrip("/")
NAME = os.environ.get("VLLM_PROXY_NAME", "vllm-no-think-proxy")
TIMEOUT = float(os.environ.get("VLLM_PROXY_TIMEOUT", "600"))

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}


def _prefix_no_think(content: Any) -> Any:
    if isinstance(content, str):
        stripped = content.lstrip()
        if stripped.startswith("/no_think"):
            return content
        return "/no_think\n" + content
    if isinstance(content, list):
        updated = []
        inserted = False
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("text", "input_text") and isinstance(item.get("text"), str):
                if not inserted:
                    text = item["text"]
                    if not text.lstrip().startswith("/no_think"):
                        item = dict(item)
                        item["text"] = "/no_think\n" + text
                    inserted = True
                updated.append(item)
            else:
                updated.append(item)
        if not inserted:
            updated.insert(0, {"type": "text", "text": "/no_think"})
        return updated
    return content


def mutate_chat_payload(raw: bytes) -> bytes:
    try:
        payload = json.loads(raw or b"{}")
    except Exception:
        return raw
    if not isinstance(payload, dict):
        return raw
    ctk = payload.setdefault("chat_template_kwargs", {})
    if isinstance(ctk, dict):
        ctk["enable_thinking"] = False
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = _prefix_no_think(msg.get("content", ""))
                break
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "VLLMNoThinkProxy/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {self.client_address[0]} {fmt % args}\n")

    def _proxy(self) -> None:
        path = self.path
        if path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "service": NAME, "upstream": UPSTREAM}).encode())
            return
        body = b""
        if self.command in ("POST", "PUT", "PATCH"):
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length)
            clean_path = path.split("?", 1)[0]
            if clean_path.endswith("/chat/completions"):
                body = mutate_chat_payload(body)
        url = UPSTREAM + path
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP and k.lower() != "host"}
        if body:
            headers["Content-Length"] = str(len(body))
            headers["Content-Type"] = headers.get("Content-Type", "application/json")
        req = urllib.request.Request(url, data=body if self.command != "GET" else None, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() not in HOP_BY_HOP:
                        self.send_header(k, v)
                self.end_headers()
                while True:
                    chunk = r.read(64 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(e), "upstream": UPSTREAM}).encode())

    def do_GET(self) -> None: self._proxy()
    def do_HEAD(self) -> None: self._proxy()
    def do_POST(self) -> None: self._proxy()
    def do_OPTIONS(self) -> None: self._proxy()


def main() -> None:
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"{NAME} listening on {HOST}:{PORT}, upstream={UPSTREAM}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
