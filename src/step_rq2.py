import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

PROJECT = Path(r"C:\phishing_project")
SEED = 42

print("=== RQ2 : Generalisation inter-datasets ===")
print("Entrainement : Phish360 | Test : Phishing dataset")

# 1. Charger Phish360 (train complet)
print("\n1. Chargement Phish360...")
df = pd.read_parquet(PROJECT / "data" / "processed" / "phish360_ready.parquet")
train = df[df["split"] == "train"].reset_index(drop=True)
print("Train Phish360:", len(train), "sites | Phishing:", train["label"].sum())

# 2. Charger le Phishing dataset
print("\n2. Chargement Phishing dataset...")
rq2 = pd.read_excel(r"C:\Users\ahmed\Desktop\Phishing dataset\Phishing dataset\url\merged_features_final.xlsx",
                    usecols=["url", "label"])
rq2["label_bin"] = (rq2["label"] == "phishing").astype(int)
print("Phishing dataset:", len(rq2), "sites")
print("Labels:", rq2["label"].value_counts().to_dict())

# 3. Entrainer Mod 1 sur Phish360 train
print("\n3. Entrainement Mod 1 sur Phish360 train...")
vec = TfidfVectorizer(analyzer="char", ngram_range=(2,5), min_df=3,
                      sublinear_tf=True, max_features=300000)
Xtr = vec.fit_transform(train["url"])
clf = LogisticRegression(max_iter=1000, solver="liblinear", C=10.0, random_state=SEED)
clf.fit(Xtr, train["label"])
print("Entrainement termine")

# 4. Tester sur Phish360 test (reference)
print("\n4. Note de reference sur Phish360 test...")
test_p360 = df[df["split"] == "test"].reset_index(drop=True)
Xte_p360 = vec.transform(test_p360["url"])
pte_p360 = clf.predict_proba(Xte_p360)[:, 1]
pred_p360 = (pte_p360 >= 0.5).astype(int)
f1_p360   = f1_score(test_p360["label"], pred_p360, zero_division=0)
auc_p360  = roc_auc_score(test_p360["label"], pte_p360)
print(f"  Phish360 test  : F1 = {round(f1_p360,3)} | AUC = {round(auc_p360,3)}")

# 5. Tester sur Phishing dataset (RQ2)
print("\n5. Test sur Phishing dataset (RQ2)...")
Xrq2 = vec.transform(rq2["url"])
prq2 = clf.predict_proba(Xrq2)[:, 1]
pred_rq2 = (prq2 >= 0.5).astype(int)
f1_rq2   = f1_score(rq2["label_bin"], pred_rq2, zero_division=0)
auc_rq2  = roc_auc_score(rq2["label_bin"], prq2)
prec_rq2 = precision_score(rq2["label_bin"], pred_rq2, zero_division=0)
rap_rq2  = recall_score(rq2["label_bin"], pred_rq2, zero_division=0)
print(f"  Phishing dataset : F1 = {round(f1_rq2,3)} | Precision = {round(prec_rq2,3)} | Rappel = {round(rap_rq2,3)} | AUC = {round(auc_rq2,3)}")

# 6. Conclusion RQ2
print("\n=== CONCLUSION RQ2 ===")
print(f"F1 sur Phish360 test    : {round(f1_p360,3)}")
print(f"F1 sur Phishing dataset : {round(f1_rq2,3)}")
chute = round(f1_p360 - f1_rq2, 3)
print(f"Chute de generalisation : {chute} points")
if chute <= 0.05:
    print("-> Le modele generalise BIEN (chute <= 5 points)")
elif chute <= 0.10:
    print("-> Generalisation ACCEPTABLE (chute <= 10 points)")
else:
    print("-> Generalisation FAIBLE (chute > 10 points)")

# 7. Sauvegarder
res = {
    "phish360_test":      {"F1": round(f1_p360,3),  "AUC": round(auc_p360,3)},
    "phishing_dataset":   {"F1": round(f1_rq2,3),   "AUC": round(auc_rq2,3),
                           "precision": round(prec_rq2,3), "rappel": round(rap_rq2,3),
                           "sites": len(rq2), "phishing": int(rq2["label_bin"].sum())},
    "chute_generalisation": chute,
}
(PROJECT / "reports" / "step_rq2_report.json").write_text(
    json.dumps(res, indent=2), encoding="utf-8")
print("\n=== RQ2 OK ✅ ===")
