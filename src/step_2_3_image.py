import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

PROJECT = Path(r"C:\phishing_project")
SEED    = 42
DEVICE  = torch.device("cuda")
BATCH   = 32
EPOCHS  = 5
LR      = 1e-4

torch.manual_seed(SEED)
print("GPU:", torch.cuda.get_device_name(0))

df = pd.read_parquet(PROJECT / "data" / "processed" / "phish360_ready.parquet",
                     columns=["folder_name", "label", "split", "gros_domaine", "image_path"])
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df   = df[df["split"] == "val"].reset_index(drop=True)
print("train:", len(train_df), "| val:", len(val_df))

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
tf_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
tf_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class PhishDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (255, 255, 255))
        return self.transform(img), int(row["label"])

train_loader = DataLoader(PhishDataset(train_df, tf_train), batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(PhishDataset(val_df,   tf_val),   batch_size=BATCH, shuffle=False, num_workers=0)

model = models.resnet50(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    loss_sum = 0
    all_labels, all_probs = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.float().to(DEVICE)
            if train: optimizer.zero_grad()
            out = model(imgs).squeeze(1)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            all_probs.extend(torch.sigmoid(out).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    p = np.array(all_probs)
    y = np.array(all_labels, dtype=int)
    pred = (p >= 0.5).astype(int)
    return {
        "loss": round(loss_sum / len(y), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "rappel": round(recall_score(y, pred, zero_division=0), 3),
        "F1": round(f1_score(y, pred, zero_division=0), 3),
        "AUC": round(roc_auc_score(y, p), 3),
        "probs": p, "labels": y,
    }

res = {}
best_f1, best_probs = 0, None
print("\nEntrainement ResNet50 sur GPU...")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr = run_epoch(train_loader, train=True)
    va = run_epoch(val_loader,   train=False)
    print(f"Epoch {epoch}/{EPOCHS} ({round(time.time()-t0)}s) | train F1={tr['F1']} | val F1={va['F1']} AUC={va['AUC']}")
    res[f"epoch_{epoch}_train"] = {k: v for k, v in tr.items() if k not in ("probs","labels")}
    res[f"epoch_{epoch}_val"]   = {k: v for k, v in va.items() if k not in ("probs","labels")}
    if va["F1"] > best_f1:
        best_f1    = va["F1"]
        best_probs = va["probs"]
        best_ep    = epoch
        torch.save(model.state_dict(), PROJECT / "data" / "processed" / "mod3_resnet50.pt")

print(f"\nMeilleure epoch : {best_ep} | F1 val = {best_f1}")

m = (val_df["gros_domaine"] == 0).values
y_all = val_df["label"].values
pred_all = (best_probs >= 0.5).astype(int)
pred_sg  = (best_probs[m] >= 0.5).astype(int)
res["val_complet"] = {
    "sites": int(len(y_all)), "phishing": int(y_all.sum()),
    "precision": round(precision_score(y_all, pred_all, zero_division=0), 3),
    "rappel": round(recall_score(y_all, pred_all, zero_division=0), 3),
    "F1": round(f1_score(y_all, pred_all, zero_division=0), 3),
    "AUC": round(roc_auc_score(y_all, best_probs), 3),
}
res["val_sans_gros"] = {
    "sites": int(m.sum()), "phishing": int(val_df.loc[m, "label"].sum()),
    "precision": round(precision_score(y_all[m], pred_sg, zero_division=0), 3),
    "rappel": round(recall_score(y_all[m], pred_sg, zero_division=0), 3),
    "F1": round(f1_score(y_all[m], pred_sg, zero_division=0), 3),
    "AUC": round(roc_auc_score(y_all[m], best_probs[m]), 3),
}
print("\n--- Resultats finaux (meilleure epoch) ---")
print("val (tous)      :", res["val_complet"])
print("val (sans gros) :", res["val_sans_gros"])

out = val_df[["folder_name"]].copy()
out["p_image"] = best_probs
out.to_parquet(PROJECT / "data" / "processed" / "mod3_image_val_probs.parquet", index=False)
res["meilleur"] = {"epoch": best_ep, "F1_val": best_f1}
(PROJECT / "reports" / "step_2_3_report.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("\n=== ETAPE 2.3 OK ? ===")
