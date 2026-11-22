"""API FastAPI — détection de phishing multimodale avec logs temps réel (SSE).

Lancement (depuis la racine du projet) :
    python -m uvicorn backend.main:app --reload --port 8000

Flux :
    1. POST /api/analyze            (multipart : url, html, screenshot) -> {job_id}
    2. GET  /api/analyze/{id}/stream  (text/event-stream)  -> events log / result / failed / done
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .pipeline import analyze, load_models, log

HISTORY_FILE = Path(__file__).resolve().parent / "history.json"
HISTORY_MAX = 200
JOB_TTL = 600  # s
MAX_HTML = 10 * 1024 * 1024
MAX_IMAGE = 15 * 1024 * 1024

state: dict = {"models": None, "error": None}
jobs: dict[str, dict] = {}
history_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state["models"] = await asyncio.to_thread(load_models)
        print(f"[OK] Modèles chargés (image sur {state['models'].device})")
    except Exception as exc:
        state["error"] = str(exc)
        print(f"[ERREUR] {exc}")
    yield


app = FastAPI(title="Phishing Detector — Multimodal", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- history
def read_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def write_history(items: list[dict]) -> None:
    tmp = HISTORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(items[:HISTORY_MAX], ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(HISTORY_FILE)


def add_history(entry: dict) -> None:
    with history_lock:
        write_history([entry] + read_history())


# --------------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    m = state["models"]
    return {
        "ready": m is not None,
        "error": state["error"],
        "device": m.device if m else None,
        "image_model": bool(m and m.resnet is not None),
        "meta": m.meta if m else None,
    }


@app.post("/api/analyze")
async def create_job(
    url: str = Form(""),
    html: UploadFile | None = File(None),
    screenshot: UploadFile | None = File(None),
):
    if state["models"] is None:
        raise HTTPException(503, state["error"] or "Modèles en cours de chargement")
    url = url.strip()
    html_bytes = await html.read() if html else None
    img_bytes = await screenshot.read() if screenshot else None
    if not (url or html_bytes or img_bytes):
        raise HTTPException(400, "Fournissez au moins une URL, un HTML ou un screenshot")
    if html_bytes and len(html_bytes) > MAX_HTML:
        raise HTTPException(413, "Fichier HTML trop volumineux (max 10 Mo)")
    if img_bytes and len(img_bytes) > MAX_IMAGE:
        raise HTTPException(413, "Screenshot trop volumineux (max 15 Mo)")

    now = time.time()
    for jid in [j for j, v in jobs.items() if now - v["created"] > JOB_TTL]:
        jobs.pop(jid, None)

    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "url": url, "html": html_bytes or None, "image": img_bytes or None, "created": now,
        "html_name": html.filename if html_bytes else None,
        "image_name": screenshot.filename if img_bytes else None,
    }
    return {"job_id": job_id}


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/analyze/{job_id}/stream")
def stream(job_id: str):
    job = jobs.pop(job_id, None)
    if job is None:
        raise HTTPException(404, "Analyse inconnue ou déjà consommée")

    # Générateur synchrone : Starlette l'exécute dans un threadpool, chaque yield est flushé.
    def events():
        yield ": connected\n\n"
        try:
            for ev in analyze(state["models"], job["url"], job["html"], job["image"]):
                if ev["event"] == "result":
                    entry = {
                        "id": job_id,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "html_name": job["html_name"],
                        "image_name": job["image_name"],
                        **ev["data"],
                    }
                    add_history(entry)
                    yield sse("result", entry)
                else:
                    yield sse(ev["event"], ev["data"])
        except Exception as exc:
            yield sse("log", log("error", f"Erreur interne : {exc}")["data"])
            yield sse("failed", {"message": str(exc)})
        yield sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


@app.get("/api/history")
def get_history():
    return read_history()


@app.delete("/api/history")
def clear_history():
    with history_lock:
        write_history([])
    return {"ok": True}


@app.delete("/api/history/{entry_id}")
def delete_history(entry_id: str):
    with history_lock:
        items = read_history()
        write_history([h for h in items if h.get("id") != entry_id])
    return {"ok": True}
