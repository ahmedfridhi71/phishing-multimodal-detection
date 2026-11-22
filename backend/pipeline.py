"""Chargement des modèles et pipeline d'analyse multimodale (générateur d'événements)."""
from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT / "models"
IMAGE_MODEL = PROJECT / "data" / "processed" / "mod3_resnet50.pt"

THRESHOLD = 0.5
MAX_HTML_BYTES = 2_000_000  # identique à step_2_2_0_html_text.py
MAX_TEXT = 20_000
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- events
def log(level: str, message: str) -> dict:
    """level : ok | progress | error | info"""
    return {"event": "log", "data": {"level": level, "message": message, "ts": time.time()}}


# --------------------------------------------------------------------------- models
@dataclass
class Models:
    url: dict
    html_text: dict
    html_struct: dict
    fusion: dict
    resnet: object | None
    device: str
    meta: dict


def load_models() -> Models:
    missing = [f for f in ("mod1_url.joblib", "mod2a_html_text.joblib",
                           "mod2b_html_struct.joblib", "fusion.joblib")
               if not (MODELS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Modèles manquants dans {MODELS_DIR}: {missing}. Lancez d'abord : python save_models.py")

    meta_path = MODELS_DIR / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    resnet, device = None, "cpu"
    try:
        import torch
        import torch.nn as nn
        from torchvision import models as tvm

        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = tvm.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, 1)
        net.load_state_dict(torch.load(IMAGE_MODEL, map_location=device, weights_only=True))
        resnet = net.to(device).eval()
    except Exception as exc:  # le backend reste utilisable sans le modèle image
        print(f"[WARN] Modèle image non chargé : {exc}")

    return Models(
        url=joblib.load(MODELS_DIR / "mod1_url.joblib"),
        html_text=joblib.load(MODELS_DIR / "mod2a_html_text.joblib"),
        html_struct=joblib.load(MODELS_DIR / "mod2b_html_struct.joblib"),
        fusion=joblib.load(MODELS_DIR / "fusion.joblib"),
        resnet=resnet,
        device=device,
        meta=meta,
    )


# --------------------------------------------------------------------------- features
def parse_html(raw: bytes) -> dict:
    """Même extraction que src/step_2_2_0_html_text.py."""
    raw = raw[:MAX_HTML_BYTES]
    soup = BeautifulSoup(raw.decode("utf-8-sig", errors="replace"), "html.parser")
    title = soup.title.get_text(" ", strip=True)[:300] if soup.title else ""
    inputs = soup.find_all("input")
    feats = {
        "title": title,
        "n_forms": len(soup.find_all("form")),
        "n_password": sum(1 for i in inputs if (i.get("type") or "").lower() == "password"),
        "n_inputs": len(inputs),
        "n_links": len(soup.find_all("a")),
        "n_scripts": len(soup.find_all("script")),
        "n_iframes": len(soup.find_all("iframe")),
    }
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    feats["text"] = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:MAX_TEXT]
    return feats


def predict_image(m: Models, raw: bytes) -> float:
    import torch
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    with torch.no_grad():
        out = m.resnet(tf(img).unsqueeze(0).to(m.device)).squeeze()
        return float(torch.sigmoid(out).item())


# --------------------------------------------------------------------------- pipeline
def analyze(m: Models, url: str, html: bytes | None, image: bytes | None) -> Iterator[dict]:
    """Génère des événements SSE : log (ok/progress/error/info) puis result."""
    t0 = time.perf_counter()
    scores: dict[str, float | None] = {
        "p_url": None, "p_html_texte": None, "p_html_structure": None, "p_image": None}
    details: dict = {}

    yield log("info", f"Analyse démarrée — {url or '(URL vide)'}")

    # --- URL
    if url:
        yield log("progress", "Mod 1 · Vectorisation TF-IDF des n-grammes de caractères (2-5)...")
        X = m.url["vectorizer"].transform([url])
        scores["p_url"] = float(m.url["clf"].predict_proba(X)[0, 1])
        yield log("ok", f"Mod 1 · URL analysée — p(phishing) = {scores['p_url']:.3f}  [{X.nnz} n-grammes]")
    else:
        yield log("error", "Mod 1 · Aucune URL fournie — modalité ignorée")

    # --- HTML
    if html:
        yield log("progress", f"Mod 2 · Parsing HTML ({len(html) / 1024:.1f} Ko)...")
        try:
            feats = parse_html(html)
            details["html"] = {k: v for k, v in feats.items() if k != "text"}
            details["html"]["text_chars"] = len(feats["text"])
            yield log("ok", f"Mod 2 · HTML parsé — titre : « {feats['title'][:60] or '(aucun)'} », "
                            f"{len(feats['text'])} caractères de texte visible")

            yield log("progress", "Mod 2A · Classification du texte (TF-IDF mots + LinearSVC)...")
            doc = f"{feats['title']} {feats['text']}".strip()
            if len(doc) < 20:
                yield log("info", "Mod 2A · Texte visible très court — score peu fiable")
            Xt = m.html_text["vectorizer"].transform([doc])
            scores["p_html_texte"] = float(m.html_text["clf"].predict_proba(Xt)[0, 1])
            yield log("ok", f"Mod 2A · Texte HTML — p(phishing) = {scores['p_html_texte']:.3f}")

            yield log("progress", "Mod 2B · Analyse de la structure (formulaires, mots de passe, iframes...)...")
            cols = m.html_struct["columns"]
            Xs = np.log1p(np.array([[feats[c] for c in cols]], dtype=float))
            scores["p_html_structure"] = float(m.html_struct["clf"].predict_proba(Xs)[0, 1])
            struct = ", ".join(f"{c[2:]}={feats[c]}" for c in cols)
            yield log("ok", f"Mod 2B · Structure ({struct}) — p(phishing) = {scores['p_html_structure']:.3f}")
        except Exception as exc:
            yield log("error", f"Mod 2 · Échec de l'analyse HTML : {exc}")
    else:
        yield log("error", "Mod 2 · Aucun fichier HTML fourni — modalité ignorée")

    # --- Image
    if image:
        if m.resnet is None:
            yield log("error", "Mod 3 · Modèle image indisponible (torch/poids non chargés)")
        else:
            yield log("progress", f"Mod 3 · Inférence ResNet50 sur {m.device.upper()} (224×224)...")
            try:
                scores["p_image"] = predict_image(m, image)
                yield log("ok", f"Mod 3 · Screenshot analysé — p(phishing) = {scores['p_image']:.3f}")
            except Exception as exc:
                yield log("error", f"Mod 3 · Image illisible : {exc}")
    else:
        yield log("error", "Mod 3 · Aucun screenshot fourni — modalité ignorée")

    # --- Fusion
    available = {k: v for k, v in scores.items() if v is not None}
    if not available:
        yield log("error", "Fusion impossible : aucune modalité exploitable")
        yield {"event": "failed", "data": {"message": "Aucune modalité exploitable"}}
        return

    yield log("progress", "Fusion · Combinaison des experts...")
    features = m.fusion["features"]
    if len(available) == len(features):
        X = np.array([[scores[f] for f in features]])
        p_final = float(m.fusion["clf"].predict_proba(X)[0, 1])
        method = "Régression logistique (4 experts)"
    else:
        p_final = float(np.mean(list(available.values())))
        method = f"Moyenne simple ({len(available)}/{len(features)} experts)"
        yield log("info", "Fusion · Modalités manquantes → repli sur la moyenne simple")

    is_phishing = p_final >= THRESHOLD
    confidence = p_final if is_phishing else 1 - p_final
    elapsed = time.perf_counter() - t0
    verdict = "PHISHING" if is_phishing else "LÉGITIME"
    yield log("error" if is_phishing else "ok",
              f"Verdict : {verdict} — p(phishing) = {p_final:.3f}, confiance {confidence:.1%} ({elapsed:.2f} s)")

    yield {"event": "result", "data": {
        "url": url,
        "label": "phishing" if is_phishing else "legitimate",
        "probability": round(p_final, 4),
        "confidence": round(confidence, 4),
        "scores": {k: (round(v, 4) if v is not None else None) for k, v in scores.items()},
        "fusion_method": method,
        "details": details,
        "elapsed": round(elapsed, 3),
    }}
