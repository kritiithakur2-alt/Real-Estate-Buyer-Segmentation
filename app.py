"""Interactive Streamlit dashboard for the synthetic real-estate buyer dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Real Estate Buyer Intelligence",page_icon="🏘️",layout="wide")
st.title("Real Estate Buyer Intelligence")
st.caption("Buyer segmentation and investment profiling | Synthetic practice data — illustrative only")
uploaded = st.sidebar.file_uploader("Upload buyer dataset (CSV)",type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    candidates = ["real_estate_buyer_synthetic_dataset.csv","real_estate_buyer_synthetic_dataset(1).csv"]
    path = next((p for p in candidates if Path(p).exists()),None)
    if path is None:
        st.info("Upload the project CSV in the sidebar, or place it beside app.py.")
        st.stop()
    df = pd.read_csv(path)
required = ["client_id","client_type","gender","country","region","date_of_birth","acquisition_purpose","loan_applied","referral_channel","satisfaction_score"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Missing columns: "+", ".join(missing)); st.stop()
df = df.drop_duplicates().drop_duplicates(subset=["client_id"],keep="first").dropna(subset=["region"]).copy()
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"],errors="coerce")
df["age"] = ((pd.Timestamp("2026-11-11")-df["date_of_birth"]).dt.days/365.25).round()
df.loc[~df["age"].between(18,100),"age"] = np.nan
df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"],errors="coerce")
cat = ["client_type","gender","country","region","acquisition_purpose","loan_applied","referral_channel"]
num = ["age","satisfaction_score"]
prep = ColumnTransformer([
 ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
 ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),cat)])
X = prep.fit_transform(df[cat+num])
k = st.sidebar.slider("Number of buyer clusters",2,7,3)
df["cluster"] = KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(X)
score = silhouette_score(X,df["cluster"])
c1,c2,c3,c4 = st.columns(4)
c1.metric("Buyers analysed",f"{len(df):,}")
c2.metric("Clusters",k)
c3.metric("Average satisfaction",f"{df.satisfaction_score.mean():.2f}")
c4.metric("Silhouette score",f"{score:.3f}")
left,right = st.columns(2)
with left:
    purpose = df["acquisition_purpose"].fillna("Unknown").value_counts().rename_axis("Purpose").reset_index(name="Buyers")
    st.plotly_chart(px.bar(purpose,x="Purpose",y="Buyers",title="Buyers by acquisition purpose"),use_container_width=True)
with right:
    region = df["region"].value_counts().rename_axis("Region").reset_index(name="Buyers")
    st.plotly_chart(px.pie(region,names="Region",values="Buyers",title="Geographic buyer distribution"),use_container_width=True)
profile = df.groupby("cluster").agg(buyers=("client_id","count"),median_age=("age","median"),investment_share=("acquisition_purpose",lambda s:(s=="Investment").mean()*100),loan_share=("loan_applied",lambda s:(s=="Yes").mean()*100),satisfaction=("satisfaction_score","mean")).reset_index()
st.subheader("Cluster profiles")
st.dataframe(profile.round(2),use_container_width=True,hide_index=True)
st.plotly_chart(px.scatter(df,x="age",y="satisfaction_score",color=df["cluster"].astype(str),hover_data=["client_id","region","acquisition_purpose"],title="Buyer clusters by age and satisfaction"),use_container_width=True)
regions = st.multiselect("Filter by region",sorted(df.region.dropna().unique()))
view = df[df.region.isin(regions)] if regions else df
st.subheader("Buyer records")
st.dataframe(view,use_container_width=True,hide_index=True)
st.download_button("Download segmented buyer data",view.to_csv(index=False).encode("utf-8"),"buyer_segments_filtered.csv","text/csv")
st.caption("A low silhouette score means groups overlap. Treat segments as exploratory, not definitive market categories or investment advice. Dataset is synthetic, not actual Parcl customer data.")
