import re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from bs4 import BeautifulSoup

PROJECT = Path(r"C:\phishing_project")
READY = PROJECT / "data" / "processed" / "phish360_ready.parquet"
PARTS = PROJECT / "data" / "processed" / "html_parts"
FINAL = PROJECT / "data" / "processed" / "phish360_html_text.parquet"
PARTS.mkdir(parents=True, exist_ok=True)

MAX_BYTES = 2_000_000   # on lit au plus 2 Mo par fichier
MAX_TEXT = 20_000       # on garde au plus 20 000 caractères de texte visible
CHUNK = 500             # sauvegarde tous les 500 sites
WORKERS = 4

def extract(args):
    folder, path = args
    try:
        with open(path, "rb") as f:
            raw = f.read(MAX_BYTES)
    except OSError:
        return {"folder_name": folder, "html_ok": False}
    html = raw.decode("utf-8-sig", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True)[:300] if soup.title else ""
    inputs = soup.find_all("input")
    row = {
        "folder_name": folder, "html_ok": True,
        "html_tronque": len(raw) >= MAX_BYTES,
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
    row["text"] = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:MAX_TEXT]
    return row

df = pd.read_parquet(READY, columns=["folder_name", "html_path"])
done = set()
for p in PARTS.glob("part_*.parquet"):
    done |= set(pd.read_parquet(p, columns=["folder_name"])["folder_name"])
todo = df[~df["folder_name"].isin(done)]
print("Sites au total:", len(df), "| déjà faits:", len(done), "| à faire:", len(todo))

jobs = list(zip(todo["folder_name"], todo["html_path"]))
t0 = time.time()
with ThreadPoolExecutor(WORKERS) as ex:
    for start in range(0, len(jobs), CHUNK):
        rows = list(ex.map(extract, jobs[start:start + CHUNK]))
        pd.DataFrame(rows).to_parquet(PARTS / f"part_{len(done) + start:06d}.parquet", index=False)
        fait = min(start + CHUNK, len(jobs))
        print(f"{fait}/{len(jobs)} faits | {round(time.time() - t0)} s")

parts = [pd.read_parquet(p) for p in sorted(PARTS.glob("part_*.parquet"))]
out = pd.concat(parts, ignore_index=True)
if out["folder_name"].nunique() == len(df):
    out.to_parquet(FINAL, index=False)
    print("\nHTML illisibles:", int((~out["html_ok"]).sum()), "| tronqués à 2 Mo:", int(out["html_tronque"].fillna(False).sum()))
    print("Texte vide (moins de 20 caractères):", int((out["text"].fillna("").str.len() < 20).sum()))
    print("Fichier:", FINAL)
    print("=== ETAPE 2.2.0 OK ✅ ===")
else:
    print("Pas encore complet:", out["folder_name"].nunique(), "/", len(df), "-> relancez la même commande")
