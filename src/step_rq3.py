import json, time
from pathlib import Path
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

PROJECT = Path(r"C:\phishing_project")
EXCEL   = r"C:\Users\ahmed\Desktop\Phishing dataset\Phishing dataset\url\merged_features_final.xlsx"

print("=== RQ3 : Big Data avec Spark ===")

# 1. Charger les données avec pandas
print("\n1. Chargement du Phishing dataset...")
df_pd = pd.read_excel(EXCEL, usecols=["url", "label"])
df_pd["label_bin"] = (df_pd["label"] == "phishing").astype(int)
df_pd = df_pd[["url", "label_bin"]].dropna()
print(f"Total : {len(df_pd)} sites")

# 2. Volumes à tester
VOLUMES  = [0.10, 0.25, 0.50, 1.00]
WORKERS  = [1, 2, 4]
res      = {"mesures": [], "total_sites": len(df_pd)}

# 3. Boucle sur les workers
for n_workers in WORKERS:
    print(f"\n--- {n_workers} worker(s) ---")

    spark = (SparkSession.builder
             .appName(f"RQ3_phishing_{n_workers}w")
             .master(f"local[{n_workers}]")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", str(n_workers * 4))
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    for vol in VOLUMES:
        n = int(len(df_pd) * vol)
        sample = df_pd.sample(n=n, random_state=42).reset_index(drop=True)

        t0 = time.time()

        # Convertir en Spark DataFrame
        sdf = spark.createDataFrame(sample)

        # Pipeline : Tokenizer → HashingTF → IDF → LogReg
        tok   = Tokenizer(inputCol="url", outputCol="words")
        htf   = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
        idf   = IDF(inputCol="rawFeatures", outputCol="features")
        lr    = LogisticRegression(labelCol="label_bin", featuresCol="features",
                                   maxIter=20, regParam=0.1)
        pipe  = Pipeline(stages=[tok, htf, idf, lr])

        # Train / test split
        train_sdf, test_sdf = sdf.randomSplit([0.8, 0.2], seed=42)
        model  = pipe.fit(train_sdf)
        preds  = model.transform(test_sdf)

        # Évaluation
        eval_ = BinaryClassificationEvaluator(labelCol="label_bin",
                                               metricName="areaUnderROC")
        auc   = round(eval_.evaluate(preds), 3)
        duree = round(time.time() - t0, 2)

        print(f"  Volume {int(vol*100):3d}% ({n:6d} sites) | {n_workers} workers | {duree:6.2f}s | AUC={auc}")
        res["mesures"].append({
            "workers": n_workers, "volume_pct": int(vol*100),
            "sites": n, "temps_s": duree, "AUC": auc
        })

    spark.stop()

# 4. Tableau de synthèse
print("\n=== TABLEAU DE SYNTHESE ===")
df_res = pd.DataFrame(res["mesures"])
pivot  = df_res.pivot_table(index="volume_pct", columns="workers",
                             values="temps_s", aggfunc="mean")
print(pivot.to_string())

# 5. Sauvegarde
df_res.to_csv(PROJECT / "reports" / "summary_rq3.csv", index=False, encoding="utf-8-sig")
(PROJECT / "reports" / "step_rq3_report.json").write_text(
    json.dumps(res, indent=2), encoding="utf-8")
print("\nTableau sauvegarde dans reports\\summary_rq3.csv")
print("=== RQ3 OK ✅ ===")
