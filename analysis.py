"""Reproducible real-estate buyer segmentation. Dataset is synthetic practice data."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

DATA_FILE = Path("real_estate_buyer_synthetic_dataset.csv")

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Place {DATA_FILE.name} in the project root.")
    df = pd.read_csv(DATA_FILE)
    df = df.drop_duplicates().drop_duplicates(subset=["client_id"], keep="first").copy()
    df = df.dropna(subset=["region"]).copy()
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age"] = ((pd.Timestamp("2026-11-11") - df["date_of_birth"]).dt.days / 365.25).round()
    df.loc[~df["age"].between(18, 100), "age"] = np.nan
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    cat = ["client_type","gender","country","region","acquisition_purpose","loan_applied","referral_channel"]
    num = ["age","satisfaction_score"]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), num),
        ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    X = prep.fit_transform(df[cat+num])
    rows = []
    for k in range(2,8):
        m = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = m.fit_predict(X)
        rows.append({"k":k,"inertia":m.inertia_,"silhouette":silhouette_score(X,labels)})
    evaluation = pd.DataFrame(rows)
    best_k = int(evaluation.loc[evaluation.silhouette.idxmax(),"k"])
    df["kmeans_cluster"] = KMeans(n_clusters=best_k,random_state=42,n_init=10).fit_predict(X)
    dense = X.toarray() if hasattr(X,"toarray") else X
    df["hierarchical_cluster"] = AgglomerativeClustering(n_clusters=best_k).fit_predict(dense)
    evaluation.to_csv("kmeans_cluster_evaluation.csv",index=False)
    df.to_csv("buyer_segments_output.csv",index=False)
    profile = df.groupby("kmeans_cluster").agg(
        buyers=("client_id","count"), median_age=("age","median"),
        investment_share=("acquisition_purpose",lambda s:(s=="Investment").mean()),
        loan_share=("loan_applied",lambda s:(s=="Yes").mean()),
        satisfaction=("satisfaction_score","mean"))
    profile.to_csv("kmeans_cluster_profiles.csv")
    print("Rows analysed:",len(df),"Best k:",best_k)
    print(evaluation.to_string(index=False))
    print(profile.to_string())

if __name__ == "__main__":
    main()
