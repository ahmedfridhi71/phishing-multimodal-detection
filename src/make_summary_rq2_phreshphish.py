import json
from pathlib import Path
import pandas as pd

R = Path(r"C:\phishing_project\reports")

# Charger les deux rapports RQ2
rq2_old  = json.loads((R / "step_rq2_report.json").read_text(encoding="utf-8"))
rq2_new  = json.loads((R / "step_rq2_phreshphish_report.json").read_text(encoding="utf-8")) if (R / "step_rq2_phreshphish_report.json").exists() else {}

rows = [
    # Phishing dataset (URL seulement)
    {"Dataset": "Phish360 test (référence)", "Modele": "Mod 1 : URL",
     "F1": rq2_old["phish360_test"]["F1"], "Precision": "-", "Rappel": "-",
     "AUC": rq2_old["phish360_test"]["AUC"], "Sites": 1059, "Phishing": 421},
    {"Dataset": "Phishing dataset (RQ2)", "Modele": "Mod 1 : URL",
     "F1": rq2_old["phishing_dataset"]["F1"],
     "Precision": rq2_old["phishing_dataset"]["precision"],
     "Rappel": rq2_old["phishing_dataset"]["rappel"],
     "AUC": rq2_old["phishing_dataset"]["AUC"],
     "Sites": rq2_old["phishing_dataset"]["sites"],
     "Phishing": rq2_old["phishing_dataset"]["phishing"]},
]

# PhreshPhish (URL + HTML)
if rq2_new:
    for cle, modele in [
        ("mod1_phreshphish",    "Mod 1 : URL"),
        ("mod2a_phreshphish",   "Mod 2A : HTML"),
        ("fusion_phreshphish",  "Fusion : URL + HTML"),
    ]:
        if cle in rq2_new:
            d = rq2_new[cle]
            rows.append({
                "Dataset":   "PhreshPhish (RQ2)",
                "Modele":    modele,
                "F1":        d["F1"],
                "Precision": d["precision"],
                "Rappel":    d["rappel"],
                "AUC":       d["AUC"],
                "Sites":     d["sites"],
                "Phishing":  d["phishing"],
            })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(R / "summary_rq2_phreshphish.csv", index=False, encoding="utf-8-sig")
print("\nTableau sauvegarde dans reports\\summary_rq2_phreshphish.csv")
print("=== SUMMARY RQ2 PHRESHPHISH OK ✅ ===")
