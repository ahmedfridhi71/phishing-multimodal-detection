from pathlib import Path
import pandas as pd

PROJECT = Path(r"C:\phishing_project")
SEUIL_GROS = 20

df = pd.read_parquet(PROJECT / "data" / "processed" / "phish360_clean.parquet")
assert len(df) == 10587, "Le fichier propre devrait avoir 10587 sites"

taille = df["fld"].map(df["fld"].value_counts())
df["gros_domaine"] = (taille >= SEUIL_GROS).astype(int)

tab = df.groupby(["split", "gros_domaine"]).size().unstack(fill_value=0)
print(tab.to_string())
assert int(tab.loc["test", 1]) == 144 and int(tab.loc["val", 1]) == 63 and int(tab.loc["train", 1]) == 262, "Chiffres inattendus"

out = PROJECT / "data" / "processed" / "phish360_ready.parquet"
df.to_parquet(out, index=False)
print("Fichier écrit:", out)
print("=== ETAPE 1.2.7 OK ✅ ===")
