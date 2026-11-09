import json
from pathlib import Path
import pandas as pd

R = Path(r"C:\phishing_project\reports")

def meilleur(fichier, prefixe):
    p = R / fichier
    if not p.exists():
        print("Absent:", fichier); return []
    data = json.loads(p.read_text(encoding="utf-8"))
    m = data.get("meilleur", data.get("meilleur_A", {}))
    nom = m.get("nom", "")
    rows = []
    for suffix, label in [("_train", "train"), ("_val", "val (tous)"), ("_val_sans_gros", "val (sans gros)")]:
        cle = f"{nom}{suffix}"
        if cle in data:
            d = data[cle]
            rows.append({"Modele": prefixe + f" ({nom})", "Evaluation": label,
                         "F1": d["F1"], "Precision": d["precision"],
                         "Rappel": d["rappel"], "AUC": d["AUC"],
                         "Sites": d["sites"], "Phishing": d["phishing"]})
    return rows

def meilleur_b(fichier):
    p = R / fichier
    if not p.exists(): return []
    data = json.loads(p.read_text(encoding="utf-8"))
    m = data.get("meilleur_B", {})
    nom = m.get("nom", "")
    rows = []
    for suffix, label in [("_val", "val (tous)"), ("_val_sans_gros", "val (sans gros)")]:
        cle = f"struct_{nom}{suffix}"
        if cle in data:
            d = data[cle]
            rows.append({"Modele": f"Mod 2B : structure ({nom})", "Evaluation": label,
                         "F1": d["F1"], "Precision": d["precision"],
                         "Rappel": d["rappel"], "AUC": d["AUC"],
                         "Sites": d["sites"], "Phishing": d["phishing"]})
    return rows

def mod3(fichier):
    p = R / fichier
    if not p.exists(): return []
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for cle, label in [("val_complet", "val (tous)"), ("val_sans_gros", "val (sans gros)")]:
        if cle in data:
            d = data[cle]
            ep = data.get("meilleur", {}).get("epoch", "?")
            rows.append({"Modele": f"Mod 3 : image (ResNet50 ep{ep})", "Evaluation": label,
                         "F1": d["F1"], "Precision": d["precision"],
                         "Rappel": d["rappel"], "AUC": d["AUC"],
                         "Sites": d["sites"], "Phishing": d["phishing"]})
    return rows

rows = []
rows += meilleur("step_2_1_1_report.json", "Mod 1 : URL")
rows += meilleur("step_2_2_1_report.json", "Mod 2A : texte HTML")
rows += meilleur_b("step_2_2_1_report.json")
rows += mod3("step_2_3_report.json")

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(R / "summary_val.csv", index=False, encoding="utf-8-sig")
print("\nTableau sauvegarde dans reports\\summary_val.csv")
print("=== SYNTHESE OK ✅ ===")
