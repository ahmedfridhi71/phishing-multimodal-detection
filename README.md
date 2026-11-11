# Détection de Phishing Multimodale

Projet académique — Module "Projet Big Data et Deep Learning"
TEK-UP Tunis — Ahmed Fridhi

## Description

Ce projet détecte automatiquement les sites de phishing en combinant trois modalités :
- **URL** : analyse de l'adresse du site
- **HTML** : analyse du texte et de la structure de la page
- **Image** : analyse de la capture d'écran

## Questions de recherche

| Question | Réponse |
|---|---|
| **RQ1** : La fusion multimodale est-elle meilleure que chaque modalité seule ? | ✅ Oui, Fusion F1 = 0,983 vs meilleur expert seul F1 = 0,973 |
| **RQ2** : Le modèle généralise-t-il à de nouveaux datasets ? | ⚠️ Acceptable (0,787) mais chute temporelle (0,623 sur 2024-2025) |
| **RQ3** : Quel est le coût de calcul avec Spark ? | ✅ 2 workers plus rapides que 1, gain visible sur grands volumes |

## Dataset principal : Phish360

- 10 748 sites (6 416 légitimes, 4 332 phishing)
- Modalités : URL + HTML + capture d'écran
- Source : Université Hacettepe, Turquie (2022)
- Référence : Almakhamreh & Bozkir, Applied Sciences 16(2), 751, 2026

## Structure du projet



## Résultats principaux (RQ1)

### Sur val

| Modèle | F1 | AUC |
|---|---|---|
| Mod 1 : URL (LogReg_C10) | 0,925 | 0,987 |
| Mod 2A : HTML (LinearSVC) | 0,955 | 0,993 |
| Mod 2B : Structure (GradBoost) | 0,871 | 0,954 |
| Mod 3 : Image (ResNet50) | 0,920 | 0,977 |
| Fusion moyenne simple | 0,974 | 0,998 |
| **Fusion LogReg** | **0,975** | **0,999** |

### Sur test (résultats finaux)

| Modèle | F1 | AUC |
|---|---|---|
| Mod 1 : URL | 0,882 | 0,972 |
| Mod 2A : HTML | 0,973 | 0,997 |
| Mod 2B : Structure | 0,889 | 0,952 |
| Mod 3 : Image | 0,899 | 0,978 |
| Fusion moyenne simple | 0,976 | 0,996 |
| **Fusion LogReg** | **0,983** | **0,997** |

## Résultats RQ2 (Généralisation)

| Dataset | Période | Modèle | F1 |
|---|---|---|---|
| Phish360 test | 2022 | Mod 1 URL | 0,882 |
| Phishing dataset | 2022-2023 | Mod 1 URL | 0,787 |
| PhreshPhish | 2024-2025 | Mod 1 URL | 0,623 |
| PhreshPhish | 2024-2025 | Mod 2A HTML | 0,537 |
| PhreshPhish | 2024-2025 | Fusion URL+HTML | 0,581 |

## Installation

```bash
# Créer l environnement virtuel
python -m venv .venv
.venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt

# GPU (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Ordre d exécution

```bash
python src/step_1_2_clean.py
python src/step_1_2_7_flag.py
python src/step_2_2_0_html_text.py
python src/step_2_1_1_url_baseline.py
python src/step_2_2_1_html_baseline.py
python src/step_2_3_image.py
python src/step_fusion.py
python src/step_test_final.py
python src/step_rq2.py
python src/step_rq2_phreshphish.py
python src/step_rq3.py
```

## Limites

- Pas de date dans Phish360 → test temporel difficile
- weebly.com représente 33% du phishing du test
- Mod 3 nécessite un GPU (NVIDIA RTX 4050 utilisé)
- PhreshPhish (2024-2025) montre une chute de performance → le phishing évolue

## Références

- Almakhamreh & Bozkir (2026). CrossPhire: Multimodal Phishing Detection. Applied Sciences, 16(2), 751.
- PhreshPhish : https://huggingface.co/datasets/phreshphish/phreshphish
