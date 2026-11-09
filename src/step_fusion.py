import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

PROJECT = Path(r"C:\phishing_project")
SEED    = 42
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

df = pd.read_parquet(PROJECT / "data" / "processed" / "phish360_ready.parquet")
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df   = df[df["split"] == "val"].reset_index(drop=True)
m = (val_df["gros_domaine"] == 0).values

def mesures(nom, y, p):
    pred = (p >= 0.5).astype(int)
    r = {"sites": int(len(y)), "phishing": int(y.sum()),
         "precision": round(precision_score(y, pred, zero_division=0), 3),
         "rappel":    round(recall_score(y, pred, zero_division=0), 3),
         "F1":        round(f1_score(y, pred, zero_division=0), 3),
         "AUC":       round(roc_auc_score(y, p), 3)}
    print(f"  {nom:<30}", r)
    return r

res = {}
print("\n=== MOD 1 : URL ===")
vec1 = TfidfVectorizer(analyzer="char", ngram_range=(2,5), min_df=3,
                       sublinear_tf=True, max_features=300000)
Xtr1 = vec1.fit_transform(train_df["url"])
Xva1 = vec1.transform(val_df["url"])
clf1 = LogisticRegression(max_iter=1000, solver="liblinear", C=10.0, random_state=SEED)
clf1.fit(Xtr1, train_df["label"])
ptr1 = clf1.predict_proba(Xtr1)[:, 1]
pva1 = clf1.predict_proba(Xva1)[:, 1]
res["mod1_val"] = mesures("val (tous)", val_df["label"], pva1)

print("\n=== MOD 2A : HTML texte ===")
html = pd.read_parquet(PROJECT / "data" / "processed" / "phish360_html_text.parquet")
tr_html = train_df.merge(html, on="folder_name")
va_html = val_df.merge(html, on="folder_name")
tr_html["doc"] = (tr_html["title"].fillna("") + " " + tr_html["text"].fillna("")).str.strip()
va_html["doc"] = (va_html["title"].fillna("") + " " + va_html["text"].fillna("")).str.strip()
vec2 = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.9,
                       sublinear_tf=True, max_features=200000, strip_accents="unicode")
Xtr2 = vec2.fit_transform(tr_html["doc"])
Xva2 = vec2.transform(va_html["doc"])
clf2 = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, random_state=SEED))
clf2.fit(Xtr2, tr_html["label"])
ptr2 = clf2.predict_proba(Xtr2)[:, 1]
pva2 = clf2.predict_proba(Xva2)[:, 1]
res["mod2a_val"] = mesures("val (tous)", va_html["label"], pva2)

print("\n=== MOD 2B : structure ===")
cols = ["n_forms","n_password","n_inputs","n_links","n_scripts","n_iframes"]
Str2b = np.log1p(tr_html[cols].fillna(0))
Sva2b = np.log1p(va_html[cols].fillna(0))
clf2b = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED)
clf2b.fit(Str2b, tr_html["label"])
ptr2b = clf2b.predict_proba(Str2b)[:, 1]
pva2b = clf2b.predict_proba(Sva2b)[:, 1]
res["mod2b_val"] = mesures("val (tous)", va_html["label"], pva2b)

print("\n=== MOD 3 : image ===")
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
tf_img = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])
class PhishDS(Dataset):
    def __init__(self, df, tf):
        self.df = df; self.tf = tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try: img = Image.open(r["image_path"]).convert("RGB")
        except: img = Image.new("RGB", (224,224), (255,255,255))
        return self.tf(img), int(r["label"])

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(PROJECT/"data"/"processed"/"mod3_resnet50.pt", map_location=DEVICE))
model = model.to(DEVICE); model.eval()

def get_probs(ds_df):
    loader = DataLoader(PhishDS(ds_df, tf_img), batch_size=32, shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            out = model(imgs.to(DEVICE)).squeeze(1)
            probs.extend(torch.sigmoid(out).cpu().numpy())
    return np.array(probs)

ptr3 = get_probs(train_df)
pva3 = get_probs(val_df)
res["mod3_val"] = mesures("val (tous)", val_df["label"].values, pva3)

print("\n=== FUSION (entrainee sur TRAIN, notee sur VAL) ===")
X_train = np.column_stack([ptr1, ptr2, ptr2b, ptr3])
X_val   = np.column_stack([pva1, pva2, pva2b, pva3])
y_train = train_df["label"].values
y_val   = val_df["label"].values

print("\n--- Moyenne simple ---")
p_moy = X_val.mean(axis=1)
res["fusion_moy_val"]           = mesures("val (tous)",      y_val,    p_moy)
res["fusion_moy_val_sans_gros"] = mesures("val (sans gros)", y_val[m], p_moy[m])

print("\n--- Moyenne pondérée ---")
poids = np.array([0.925, 0.955, 0.871, 0.920]); poids /= poids.sum()
p_pond = X_val @ poids
res["fusion_pond_val"]           = mesures("val (tous)",      y_val,    p_pond)
res["fusion_pond_val_sans_gros"] = mesures("val (sans gros)", y_val[m], p_pond[m])

print("\n--- Régression logistique (train → val) ---")
clf_f = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train, y_train)
p_lr  = clf_f.predict_proba(X_val)[:, 1]
res["fusion_lr_val"]           = mesures("val (tous)",      y_val,    p_lr)
res["fusion_lr_val_sans_gros"] = mesures("val (sans gros)", y_val[m], p_lr[m])

print("\n--- GradBoost (train → val) ---")
clf_gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED)
clf_gb.fit(X_train, y_train)
p_gb = clf_gb.predict_proba(X_val)[:, 1]
res["fusion_gb_val"]           = mesures("val (tous)",      y_val,    p_gb)
res["fusion_gb_val_sans_gros"] = mesures("val (sans gros)", y_val[m], p_gb[m])

print("\n--- Meilleur expert seul (reference) ---")
print("  Mod 2A HTML LinearSVC : F1 = 0.955")

(PROJECT/"reports"/"step_fusion_report.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("\n=== ETAPE FUSION OK ✅ ===")
