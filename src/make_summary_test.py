import json
from pathlib import Path
import pandas as pd

R = Path(r"C:\phishing_project\reports")
data = json.loads((R / "step_test_final_report.json").read_text(encoding="utf-8"))

SOURCES = [
    ("mod1_test",            "Mod 1 : URL (LogReg_C10)",      "TEST (tous)"),
    ("mod1_test_sans_gros",  "Mod 1 : URL (LogReg_C10)",      "TEST (sans gros)"),
    ("mod2a_test",           "Mod 2A : HTML (LinearSVC)",      "TEST (tous)"),
    ("mod2a_test_sans_gros", "Mod 2A : HTML (LinearSVC)",      "TEST (sans gros)"),
    ("mod2b_test",           "Mod 2B : structure (GradBoost)", "TEST (tous)"),
    ("mod2b_test_sans_gros", "Mod 2B : structure (GradBoost)", "TEST (sans gros)"),
    ("mod3_test",            "Mod 3 : image (ResNet50)",       "TEST (tous)"),
    ("mod3_test_sans_gros",  "Mod 3 : image (ResNet50)",       "TEST (sans gros)"),
    ("fusion_moy_test",           "Fusion : moyenne simple",   "TEST (tous)"),
    ("fusion_moy_test_sans_gros", "Fusion : moyenne simple",   "TEST (sans gros)"),
    ("fusion_lr_test",            "Fusion : LogReg",           "TEST (tous)"),
    ("fusion_lr_test_sans_gros",  "Fusion : LogReg",           "TEST (sans gros)"),
]

rows = []
for cle, modele, evaluation in SOURCES:
    if cle in data:
        d = data[cle]
        rows.append({
            "Modele":     modele,
            "Evaluation": evaluation,
            "F1":         d["F1"],
            "Precision":  d["precision"],
            "Rappel":     d["rappel"],
            "AUC":        d["AUC"],
            "Sites":      d["sites"],
            "Phishing":   d["phishing"],
            "TP":         d["TP"],
            "FP":         d["FP"],
            "FN":         d["FN"],
            "TN":         d["TN"],
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(R / "summary_test.csv", index=False, encoding="utf-8-sig")
print("\nTableau sauvegarde dans reports\\summary_test.csv")
print("=== SUMMARY TEST OK ✅ ===")
