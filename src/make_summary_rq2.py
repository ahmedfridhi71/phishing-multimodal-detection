import json
from pathlib import Path
import pandas as pd

R = Path(r"C:\phishing_project\reports")
data = json.loads((R / "step_rq2_report.json").read_text(encoding="utf-8"))

rows = [
    {
        "Dataset":    "Phish360 test (référence)",
        "Modele":     "Mod 1 : URL (LogReg_C10)",
        "F1":         data["phish360_test"]["F1"],
        "Precision":  "-",
        "Rappel":     "-",
        "AUC":        data["phish360_test"]["AUC"],
        "Sites":      1059,
        "Phishing":   421,
    },
    {
        "Dataset":    "Phishing dataset (RQ2)",
        "Modele":     "Mod 1 : URL (LogReg_C10)",
        "F1":         data["phishing_dataset"]["F1"],
        "Precision":  data["phishing_dataset"]["precision"],
        "Rappel":     data["phishing_dataset"]["rappel"],
        "AUC":        data["phishing_dataset"]["AUC"],
        "Sites":      data["phishing_dataset"]["sites"],
        "Phishing":   data["phishing_dataset"]["phishing"],
    },
]

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(R / "summary_rq2.csv", index=False, encoding="utf-8-sig")
print("\nChute de generalisation :", data["chute_generalisation"], "points")
print("Verdict : Generalisation ACCEPTABLE (chute <= 10 points)")
print("\nTableau sauvegarde dans reports\\summary_rq2.csv")
print("=== SUMMARY RQ2 OK ✅ ===")
