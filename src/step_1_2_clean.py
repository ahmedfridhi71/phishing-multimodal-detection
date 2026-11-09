import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT = Path(r"C:\phishing_project")
INDEX = Path(r"C:\p360work\phish360_index.parquet")
FLAGS = PROJECT / "data" / "processed" / "phish360_flags.parquet"
OUT = PROJECT / "data" / "processed" / "phish360_clean.parquet"
REPORT = PROJECT / "reports" / "step_1_2_report.json"
SEED = 42
N_FOLDS = 10   # 1 part test (10 %), 1 part val (10 %), 8 parts train (80 %)

rapport = {}

# 1) Chargement et vérification des chiffres officiels
df = pd.read_parquet(INDEX)
assert len(df) == 10748 and int(df["label"].sum()) == 4332, "Chiffres de départ inattendus"
rapport["depart"] = {"sites": len(df), "phishing": int(df["label"].sum())}

# 2) Sites inutilisables (fichier absent ou illisible)
flags = pd.read_parquet(FLAGS)
df = df.merge(flags, on="folder_name", how="left")
assert df["utilisable"].notna().all(), "Des sites n'ont pas de drapeau"
exclus = df[~df["utilisable"]]
print("Sites exclus (fichier absent ou illisible):", len(exclus), "| classes:", exclus["label"].value_counts().to_dict())
rapport["exclus_fichiers"] = {"sites": len(exclus), "phishing": int(exclus["label"].sum())}
df = df[df["utilisable"]].drop(columns="utilisable")
assert len(df) == 10701 and int(df["label"].sum()) == 4285, "Après exclusion, on attendait 10701 sites dont 4285 phishing"

# 3) Une ligne par URL (on garde la première, par ordre de dossier)
df["url"] = df["url"].str.strip()
df = df.sort_values("folder_name").reset_index(drop=True)
avant = len(df)
df = df.drop_duplicates(subset="url", keep="first").reset_index(drop=True)
print("Doublons d'URL supprimés:", avant - len(df), "| sites restants:", len(df))
rapport["doublons_supprimes"] = int(avant - len(df))

# 4) Splits par domaine (fld), mêmes proportions de phishing
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
df["split"] = "train"
for k, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["fld"])):
    if k == 0:
        df.loc[df.index[idx], "split"] = "test"
    elif k == 1:
        df.loc[df.index[idx], "split"] = "val"

tab = df.groupby(["split", "label"]).size().unstack(fill_value=0)
tab.columns = ["legit", "phish"]
tab["part_phishing"] = (tab["phish"] / (tab["legit"] + tab["phish"])).round(3)
print("\nRépartition par split:")
print(tab.to_string())
rapport["repartition"] = tab.reset_index().to_dict(orient="records")

# 5) Vérifications automatiques : le script s'arrête si une règle est violée
by = {s: set(g["fld"]) for s, g in df.groupby("split")}
communs = {f"{a}/{b}": len(by[a] & by[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
print("\nDomaines communs entre splits:", communs)
rapport["domaines_communs"] = communs
assert sum(communs.values()) == 0, "FUITE : des domaines sont partagés entre splits"
assert df["url"].duplicated().sum() == 0, "Il reste des URLs en double"
assert tab["part_phishing"].max() - tab["part_phishing"].min() < 0.03, "Les proportions de phishing diffèrent trop"
assert set(df["split"]) == {"train", "val", "test"}

# 6) Sauvegarde
df = df.drop(columns=["label_txt", "split_orig"])
df.to_parquet(OUT, index=False)
REPORT.write_text(json.dumps(rapport, indent=2), encoding="utf-8")
print("\nFichier propre:", OUT)
print("Colonnes:", list(df.columns))
print("JAMAIS en entrée d'un modèle: label, folder_name, image_path, split, fld")
print("\n=== ETAPE 1.2.4 OK ✅ ===")