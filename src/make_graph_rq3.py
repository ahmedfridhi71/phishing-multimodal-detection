import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PROJECT = Path(r"C:\phishing_project")
df = pd.read_csv(PROJECT / "reports" / "summary_rq3.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("RQ3 : Passage à l'échelle avec Apache Spark", fontsize=14, fontweight="bold")

couleurs = {1: "#2196F3", 2: "#4CAF50", 4: "#FF5722"}
volumes  = [10, 25, 50, 100]
x        = range(len(volumes))

# --- Graphique 1 : Temps selon le volume et les workers ---
ax1 = axes[0]
for w in [1, 2, 4]:
    temps = [df[(df["workers"]==w) & (df["volume_pct"]==v)]["temps_s"].values[0] for v in volumes]
    ax1.plot(volumes, temps, marker="o", color=couleurs[w], linewidth=2, label=f"{w} worker(s)")
ax1.set_title("Temps de traitement selon le volume")
ax1.set_xlabel("Volume (%)")
ax1.set_ylabel("Temps (secondes)")
ax1.set_xticks(volumes)
ax1.set_xticklabels([f"{v}%\n({v*600:,} sites)" for v in volumes])
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Graphique 2 : Comparaison workers à 100% ---
ax2 = axes[1]
workers_list = [1, 2, 4]
temps_100 = [df[(df["workers"]==w) & (df["volume_pct"]==100)]["temps_s"].values[0] for w in workers_list]
bars = ax2.bar([f"{w} worker(s)" for w in workers_list], temps_100,
               color=[couleurs[w] for w in workers_list], width=0.5, edgecolor="white")
for bar, t in zip(bars, temps_100):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{t}s", ha="center", va="bottom", fontweight="bold")
ax2.set_title("Temps à 100% du volume (60 000 sites)")
ax2.set_ylabel("Temps (secondes)")
ax2.set_ylim(0, max(temps_100) * 1.3)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(PROJECT / "reports" / "rq3_graphique.png", dpi=150, bbox_inches="tight")
plt.show()
print("Graphique sauvegarde dans reports\\rq3_graphique.png")
print("=== GRAPHIQUE RQ3 OK ✅ ===")
