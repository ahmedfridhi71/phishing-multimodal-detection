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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

PROJECT = Path(r"C:\phishing_project")
SEED    = 42
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

df = pd.read_parquet(PROJECT / "data" / "processed" / "phish360_ready.parquet")
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df   = df[df["split"] == "val"].reset_index(drop=True)
test_df  = df[df["split"] == "test"].reset_index(drop=True)
m_val  = (val_df["gros_domaine"] == 0).values
m_test = (test_df["gros_domaine"] == 0).values
print("train:", len(train_df), "| val:", len(val_df), "| test:", len(test_df))

def mesures(nom, y, p):
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    r = {"sites": int(len(y)), "phishing": int(y.sum()),
         "precision": round(precision_score(y, pred, zero_division=0), 3),
         "rappel":    round(recall_score(y, pred, zero_division=0), 3),
         "F1":        round(f1_score(y, pred, zero_division=0), 3),
         "AUC":       round(roc_auc_score(y, p), 3),
         "TP": int(cm[1,1]), "FP": int(cm[0,1]),
         "FN": int(cm[1,0]), "TN": int(cm[0,0])}
    print(f"  {nom:<35}", r)
    return r

res = {}

# MOD 1 : URL
print("\n=== MOD 1 : URL ===")
vec1 = TfidfVectorizer(analyzer="char", ngram_range=(2,5), min_df=3,
                       sublinear_tf=True, max_features=300000)
Xtr1 = vec1.fit_transform(train_df["url"])
Xva1 = vec1.transform(val_df["url"])
Xte1 = vec1.transform(test_df["url"])
clf1 = LogisticRegression(max_iter=1000, solver="liblinear", C=10.0, random_state=SEED)
clf1.fit(Xtr1, train_df["label"])
ptr1 = clf1.predict_proba(Xtr1)[:, 1]
pva1 = clf1.predict_proba(Xva1)[:, 1]
pte1 = clf1.predict_proba(Xte1)[:, 1]
res["mod1_val"]            = mesures("val (tous)",       val_df["label"].values,  pva1)
res["mod1_val_sans_gros"]  = mesures("val (sans gros)",  val_df["label"].values[m_val], pva1[m_val])
res["mod1_test"]           = mesures("TEST (tous)",      test_df["label"].values, pte1)
res["mod1_test_sans_gros"] = mesures("TEST (sans gros)", test_df["label"].values[m_test], pte1[m_test])

# MOD 2A : HTML texte
print("\n=== MOD 2A : HTML texte ===")
html = pd.read_parquet(PROJECT / "data" / "processed" / "phish360_html_text.parquet")
tr_html = train_df.merge(html, on="folder_name")
va_html = val_df.merge(html, on="folder_name")
te_html = test_df.merge(html, on="folder_name")
for d in [tr_html, va_html, te_html]:
    d["doc"] = (d["title"].fillna("") + " " + d["text"].fillna("")).str.strip()
vec2 = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.9,
                       sublinear_tf=True, max_features=200000, strip_accents="unicode")
Xtr2 = vec2.fit_transform(tr_html["doc"])
Xva2 = vec2.transform(va_html["doc"])
Xte2 = vec2.transform(te_html["doc"])
clf2 = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, random_state=SEED))
clf2.fit(Xtr2, tr_html["label"])
ptr2 = clf2.predict_proba(Xtr2)[:, 1]
pva2 = clf2.predict_proba(Xva2)[:, 1]
pte2 = clf2.predict_proba(Xte2)[:, 1]
res["mod2a_val"]            = mesures("val (tous)",       va_html["label"].values,  pva2)
res["mod2a_val_sans_gros"]  = mesures("val (sans gros)",  va_html["label"].values[m_val], pva2[m_val])
res["mod2a_test"]           = mesures("TEST (tous)",      te_html["label"].values,  pte2)
res["mod2a_test_sans_gros"] = mesures("TEST (sans gros)", te_html["label"].values[m_test], pte2[m_test])

# MOD 2B : structure
print("\n=== MOD 2B : structure ===")
cols = ["n_forms","n_password","n_inputs","n_links","n_scripts","n_iframes"]
Str2b = np.log1p(tr_html[cols].fillna(0))
Sva2b = np.log1p(va_html[cols].fillna(0))
Ste2b = np.log1p(te_html[cols].fillna(0))
clf2b = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED)
clf2b.fit(Str2b, tr_html["label"])
ptr2b = clf2b.predict_proba(Str2b)[:, 1]
pva2b = clf2b.predict_proba(Sva2b)[:, 1]
pte2b = clf2b.predict_proba(Ste2b)[:, 1]
res["mod2b_val"]            = mesures("val (tous)",       va_html["label"].values,  pva2b)
res["mod2b_val_sans_gros"]  = mesures("val (sans gros)",  va_html["label"].values[m_val], pva2b[m_val])
res["mod2b_test"]           = mesures("TEST (tous)",      te_html["label"].values,  pte2b)
res["mod2b_test_sans_gros"] = mesures("TEST (sans gros)", te_html["label"].values[m_test], pte2b[m_test])

# MOD 3 : image
print("\n=== MOD 3 : image ===")
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
tf_img = transforms.Compose([transforms.Resize((224,224)),
                              transforms.ToTensor(),
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
model.load_state_dict(torch.load(PROJECT/"data"/"processed"/"mod3_resnet50.pt",
                                  map_location=DEVICE, weights_only=False))
model = model.to(DEVICE); model.eval()

def get_probs(ds_df):
    loader = DataLoader(PhishDS(ds_df, tf_img), batch_size=32, shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            out = model(imgs.to(DEVICE)).squeeze(1)
            probs.extend(torch.sigmoid(out).detach().cpu().numpy())
    return np.array(probs)

ptr3 = get_probs(train_df)
pva3 = get_probs(val_df)
pte3 = get_probs(test_df)
res["mod3_val"]            = mesures("val (tous)",       val_df["label"].values,  pva3)
res["mod3_val_sans_gros"]  = mesures("val (sans gros)",  val_df["label"].values[m_val], pva3[m_val])
res["mod3_test"]           = mesures("TEST (tous)",      test_df["label"].values, pte3)
res["mod3_test_sans_gros"] = mesures("TEST (sans gros)", test_df["label"].values[m_test], pte3[m_test])

# FUSION
print("\n=== FUSION (entrainee sur TRAIN, notee sur VAL et TEST) ===")
X_train = np.column_stack([ptr1, ptr2, ptr2b, ptr3])
X_val   = np.column_stack([pva1, pva2, pva2b, pva3])
X_test  = np.column_stack([pte1, pte2, pte2b, pte3])
y_train = train_df["label"].values
y_val   = val_df["label"].values
y_test  = test_df["label"].values

print("\n--- Moyenne simple ---")
p_moy_val  = X_val.mean(axis=1)
p_moy_test = X_test.mean(axis=1)
res["fusion_moy_val"]            = mesures("val (tous)",       y_val,         p_moy_val)
res["fusion_moy_val_sans_gros"]  = mesures("val (sans gros)",  y_val[m_val],  p_moy_val[m_val])
res["fusion_moy_test"]           = mesures("TEST (tous)",      y_test,        p_moy_test)
res["fusion_moy_test_sans_gros"] = mesures("TEST (sans gros)", y_test[m_test],p_moy_test[m_test])

print("\n--- LogReg (train → val et test) ---")
clf_f = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train, y_train)
p_lr_val  = clf_f.predict_proba(X_val)[:, 1]
p_lr_test = clf_f.predict_proba(X_test)[:, 1]
res["fusion_lr_val"]            = mesures("val (tous)",       y_val,         p_lr_val)
res["fusion_lr_val_sans_gros"]  = mesures("val (sans gros)",  y_val[m_val],  p_lr_val[m_val])
res["fusion_lr_test"]           = mesures("TEST (tous)",      y_test,        p_lr_test)
res["fusion_lr_test_sans_gros"] = mesures("TEST (sans gros)", y_test[m_test],p_lr_test[m_test])

print("\n=== RESUME FINAL ===")
print("Meilleur expert seul (Mod 2A) val  :", res["mod2a_val"]["F1"])
print("Meilleur expert seul (Mod 2A) TEST :", res["mod2a_test"]["F1"])
print("Fusion LogReg val  :", res["fusion_lr_val"]["F1"])
print("Fusion LogReg TEST :", res["fusion_lr_test"]["F1"])

(PROJECT/"reports"/"step_test_final_report.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("\n=== TEST FINAL OK ✅ ===")
