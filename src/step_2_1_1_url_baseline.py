import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

PROJECT = Path(r"C:\phishing_project")
SEED = 42

df = pd.read_parquet(PROJECT / "data" / "processed" / "phish360_ready.parquet")
train = df[df["split"] == "train"].reset_index(drop=True)
val   = df[df["split"] == "val"].reset_index(drop=True)
print("train:", len(train), "| val:", len(val))

# 1) Features : suites de 2 à 5 caractères de l URL
vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3,
                      sublinear_tf=True, max_features=300000)
Xtr = vec.fit_transform(train["url"])
Xva = vec.transform(val["url"])
print("Features (suites de caractères):", Xtr.shape[1])

# 2) Plusieurs modèles à comparer
MODELES = {
    "LogReg_C1"   : LogisticRegression(max_iter=1000, solver="liblinear", C=1.0,  random_state=SEED),
    "LogReg_C0.1" : LogisticRegression(max_iter=1000, solver="liblinear", C=0.1,  random_state=SEED),
    "LogReg_C10"  : LogisticRegression(max_iter=1000, solver="liblinear", C=10.0, random_state=SEED),
    "LinearSVC"   : CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, random_state=SEED)),
    "GradBoost"   : GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED),
}

# 3) Fonction de mesure
def mesures(nom, y, p):
    pred = (p >= 0.5).astype(int)
    r = {
        "sites"    : int(len(y)),
        "phishing" : int(y.sum()),
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "rappel"   : round(recall_score(y, pred, zero_division=0), 3),
        "F1"       : round(f1_score(y, pred, zero_division=0), 3),
        "AUC"      : round(roc_auc_score(y, p), 3) if y.nunique() == 2 else None,
    }
    print(f"  {nom:<30}", r)
    return r

# 4) Entraîner et noter chaque modèle
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

# 5) Résumé
print("\n" + "="*60)
print("MEILLEUR MODELE SUR VAL :", meilleur_nom, "| F1 =", meilleur_f1)
print("="*60)

# 6) Ce que le meilleur modèle a appris (si LogReg)
clf_best = MODELES[meilleur_nom]
if hasattr(clf_best, "coef_"):
    noms  = np.array(vec.get_feature_names_out())
    ordre = np.argsort(clf_best.coef_[0])
    print("\n15 suites les plus LEGITIMES :", [repr(s) for s in noms[ordre[:15]]])
    print("15 suites les plus PHISHING  :", [repr(s) for s in noms[ordre[-15:]]])

# 7) Sauvegarde du meilleur modèle
sortie = val[["folder_name"]].copy()
sortie["p_url"] = meilleur_p
sortie.to_parquet(PROJECT / "data" / "processed" / "mod1_url_val_probs.parquet", index=False)
res["meilleur"] = {"nom": meilleur_nom, "F1_val": meilleur_f1}
(PROJECT / "reports" / "step_2_1_1_report.json").write_text(
    json.dumps(res, indent=2), encoding="utf-8")

print("\n=== ETAPE 2.1.1 OK ✅ ===")
