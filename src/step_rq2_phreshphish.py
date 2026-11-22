import json
from pathlib import Path
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

PROJECT = Path(r"C:\phishing_project")
SEED = 42

print("=== RQ2 : Generalisation sur PhreshPhish ===")

# 1. Charger Phish360 train
df = pd.read_parquet(PROJECT / "data" / "processed" / "phish360_ready.parquet")
train = df[df["split"] == "train"].reset_index(drop=True)
html_text = pd.read_parquet(PROJECT / "data" / "processed" / "phish360_html_text.parquet")
train_html = train.merge(html_text, on="folder_name")
train_html["doc"] = (train_html["title"].fillna("") + " " + train_html["text"].fillna("")).str.strip()
print("Train Phish360:", len(train), "sites")

# 2. Charger PhreshPhish
pp = pd.read_parquet(PROJECT / "data" / "processed" / "phreshphish_test_part.parquet")
pp["label_bin"] = (pp["label"] == "phish").astype(int)
print("PhreshPhish:", len(pp), "sites | Phishing:", pp["label_bin"].sum())

# 3. Extraire texte HTML de PhreshPhish
print("Extraction HTML PhreshPhish...")
def extract_text(html):
    if not html or pd.isna(html): return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script","style","noscript"]): t.decompose()
        return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:20000]
    except: return ""

def extract_title(html):
    if not html or pd.isna(html): return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        return soup.title.get_text(" ", strip=True)[:300] if soup.title else ""
    except: return ""

pp["text_pp"]  = pp["html"].apply(extract_text)
pp["title_pp"] = pp["html"].apply(extract_title)
pp["doc"]      = (pp["title_pp"].fillna("") + " " + pp["text_pp"].fillna("")).str.strip()
print("Extraction terminee")

def mesures(nom, y, p):
    pred = (p >= 0.5).astype(int)
    r = {"sites": int(len(y)), "phishing": int(y.sum()),
         "precision": round(precision_score(y, pred, zero_division=0), 3),
         "rappel":    round(recall_score(y, pred, zero_division=0), 3),
         "F1":        round(f1_score(y, pred, zero_division=0), 3),
         "AUC":       round(roc_auc_score(y, p), 3)}
    print(f"  {nom:<35}", r)
    return r

res = {}

# 4. Mod 1 : URL
print("\n--- Mod 1 : URL ---")
vec1 = TfidfVectorizer(analyzer="char", ngram_range=(2,5), min_df=3,
                       sublinear_tf=True, max_features=300000)
Xtr1 = vec1.fit_transform(train["url"])
clf1 = LogisticRegression(max_iter=1000, solver="liblinear", C=10.0, random_state=SEED)
clf1.fit(Xtr1, train["label"])
ppp1 = clf1.predict_proba(vec1.transform(pp["url"]))[:, 1]
res["mod1"] = mesures("Mod 1 : URL", pp["label_bin"], ppp1)

# 5. Mod 2A : HTML texte
print("\n--- Mod 2A : HTML texte ---")
vec2 = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.9,
                       sublinear_tf=True, max_features=200000, strip_accents="unicode")
Xtr2 = vec2.fit_transform(train_html["doc"])
clf2 = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, random_state=SEED))
clf2.fit(Xtr2, train_html["label"])
ppp2 = clf2.predict_proba(vec2.transform(pp["doc"]))[:, 1]
res["mod2a"] = mesures("Mod 2A : HTML", pp["label_bin"], ppp2)

# 6. Fusion URL + HTML
print("\n--- Fusion : URL + HTML ---")
p_fusion = (ppp1 + ppp2) / 2
res["fusion"] = mesures("Fusion : URL + HTML", pp["label_bin"], p_fusion)

# 7. Conclusion
print("\n=== CONCLUSION RQ2 ===")
print(f"Mod 1  URL   : F1 = {res['mod1']['F1']}")
print(f"Mod 2A HTML  : F1 = {res['mod2a']['F1']}")
print(f"Fusion       : F1 = {res['fusion']['F1']}")

(PROJECT/"reports"/"step_rq2_phreshphish_report.json").write_text(
    json.dumps(res, indent=2), encoding="utf-8")

# 8. Summary CSV
rows = [
    {"Dataset": "PhreshPhish (RQ2)", "Modele": "Mod 1 : URL",        **{k: res["mod1"][k]    for k in ["F1","precision","rappel","AUC","sites","phishing"]}},
    {"Dataset": "PhreshPhish (RQ2)", "Modele": "Mod 2A : HTML",       **{k: res["mod2a"][k]   for k in ["F1","precision","rappel","AUC","sites","phishing"]}},
    {"Dataset": "PhreshPhish (RQ2)", "Modele": "Fusion : URL + HTML", **{k: res["fusion"][k]  for k in ["F1","precision","rappel","AUC","sites","phishing"]}},
]
df_out = pd.DataFrame(rows)
print("\n", df_out.to_string(index=False))
df_out.to_csv(PROJECT/"reports"/"summary_rq2_phreshphish.csv", index=False, encoding="utf-8-sig")
print("\nTableau sauvegarde dans reports\\summary_rq2_phreshphish.csv")
print("=== RQ2 PHRESHPHISH OK ✅ ===")
