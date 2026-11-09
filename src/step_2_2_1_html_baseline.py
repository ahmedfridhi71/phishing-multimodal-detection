import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

PROJECT = Path(r"C:\phishing_project")
SEED = 42

ready = pd.read_parquet(PROJECT / "data" / "processed" / "phish360_ready.parquet",
                        columns=["folder_name", "split", "label", "gros_domaine"])
html = pd.read_parquet(PROJECT / "data" / "processed" / "phish360_html_text.parquet")
df = ready.merge(html, on="folder_name", how="left")
assert len(df) == 10587 and df["html_ok"].all(), "Fusion HTML incorrecte"
df["doc"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

train = df[df["split"] == "train"].reset_index(drop=True)
val   = df[df["split"] == "val"].reset_index(drop=True)
print("train:", len(train), "| val:", len(val))

vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.9,
                      sublinear_tf=True, max_features=200000, strip_accents="unicode")
Xtr = vec.fit_transform(train["doc"])
Xva = vec.transform(val["doc"])
print("Features (mots et paires):", Xtr.shape[1])

MODELES = {
    "LogReg_C1"   : LogisticRegression(max_iter=1000, solver="liblinear", C=1.0,  random_state=SEED),
    "LogReg_C0.1" : LogisticRegression(max_iter=1000, solver="liblinear", C=0.1,  random_state=SEED),
    "LogReg_C10"  : LogisticRegression(max_iter=1000, solver="liblinear", C=10.0, random_state=SEED),
    "LinearSVC"   : CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, random_state=SEED)),
    "GradBoost"   : GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED),
}

def mesures(nom, y, p):
    pred = (p >= 0.5).astype(int)
    r = {
        "sites": int(len(y)), "phishing": int(y.sum()),
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "rappel": round(recall_score(y, pred, zero_division=0), 3),
        "F1": round(f1_score(y, pred, zero_division=0), 3),
        "AUC": round(roc_auc_score(y, p), 3) if y.nunique() == 2 else None,
    }
    print(f"  {nom:<30}", r)
    return r

res = {}
meilleur_f1  = 0
meilleur_nom = ""
meilleur_p   = None
m = (val["gros_domaine"] == 0).values

print("\n" + "="*60)
for nom, clf in MODELES.items():
    print(f"\n--- {nom} ---")
    clf.fit(Xtr, train["label"])
    p_tr = clf.predict_proba(Xtr)[:, 1]
    p_va = clf.predict_proba(Xva)[:, 1]
    res[f"{nom}_train"]         = mesures("train (comparer)", train["label"], p_tr)
    res[f"{nom}_val"]           = mesures("val : tous",        val["label"],   p_va)
    res[f"{nom}_val_sans_gros"] = mesures("val : sans gros",   val.loc[m, "label"], p_va[m])
    f1_val = res[f"{nom}_val"]["F1"]
    if f1_val > meilleur_f1:
        meilleur_f1  = f1_val
        meilleur_nom = nom
        meilleur_p   = p_va

print("\n" + "="*60)
print("MEILLEUR MODELE SUR VAL :", meilleur_nom, "| F1 =", meilleur_f1)
print("="*60)

# Structure (Mod 2B)
cols = ["n_forms", "n_password", "n_inputs", "n_links", "n_scripts", "n_iframes"]
Str = np.log1p(train[cols].fillna(0))
Sva = np.log1p(val[cols].fillna(0))
MODELES_B = {
    "LogReg_C1"  : LogisticRegression(max_iter=1000, solver="liblinear", C=1.0,  random_state=SEED),
    "LogReg_C0.1": LogisticRegression(max_iter=1000, solver="liblinear", C=0.1,  random_state=SEED),
    "LogReg_C10" : LogisticRegression(max_iter=1000, solver="liblinear", C=10.0, random_state=SEED),
    "GradBoost"  : GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED),
}
meilleur_f1_b  = 0
meilleur_nom_b = ""
meilleur_p_b   = None
print("\n--- Modele B : structure ---")
for nom, clf in MODELES_B.items():
    clf.fit(Str, train["label"])
    pS = clf.predict_proba(Sva)[:, 1]
    res[f"struct_{nom}_val"]           = mesures(f"B_{nom} val : tous",      val["label"],        pS)
    res[f"struct_{nom}_val_sans_gros"] = mesures(f"B_{nom} val : sans gros", val.loc[m, "label"], pS[m])
    f1_b = res[f"struct_{nom}_val"]["F1"]
    if f1_b > meilleur_f1_b:
        meilleur_f1_b  = f1_b
        meilleur_nom_b = nom
        meilleur_p_b   = pS

print("\nMEILLEUR MODELE B :", meilleur_nom_b, "| F1 =", meilleur_f1_b)

out = val[["folder_name"]].copy()
out["p_html_texte"]     = meilleur_p
out["p_html_structure"] = meilleur_p_b
out.to_parquet(PROJECT / "data" / "processed" / "mod2_html_val_probs.parquet", index=False)
res["meilleur_A"] = {"nom": meilleur_nom,   "F1_val": meilleur_f1}
res["meilleur_B"] = {"nom": meilleur_nom_b, "F1_val": meilleur_f1_b}
(PROJECT / "reports" / "step_2_2_1_report.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("\n=== ETAPE 2.2.1 OK ✅ ===")
