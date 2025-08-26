import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

API_KEY = "OUVGhxqikMYfcQ5ks7shZSUEaELGxX0ttLuzSubW"
API_URL = "https://api.nasa.gov/neo/rest/v1/feed"
DATA_DIR = "nasa_data"
DATA_FILE = os.path.join(DATA_DIR, "data.json")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_nasa_data(start_date, end_date):
   
   url = f"{API_URL}?start_date={start_date}&end_date={end_date}&api_key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
        st.success(f"Data fetched and saved for {start_date} to {end_date}")
    else:
        st.error(f"Error fetching data: {response.status_code}")

def load_data():
   
    with open(DATA_FILE, "r") as f:
        raw = json.load(f)
    records = []
    for date, objects in raw["near_earth_objects"].items():
        for obj in objects:
            est_diameter = obj["estimated_diameter"]["kilometers"]
            close_data = obj["close_approach_data"][0]
            records.append({
                "name": obj["name"],
                "velocity_kmph": float(close_data["relative_velocity"]["kilometers_per_hour"]),
                "diameter": (est_diameter["estimated_diameter_min"] + est_diameter["estimated_diameter_max"]) / 2,
                "distance": float(close_data["miss_distance"]["kilometers"]),
                "date": date,
                "hazard": int(obj["is_potentially_hazardous_asteroid"])
            })
    return pd.DataFrame(records)

def preprocess_data(df):
    """Add risk column and scale features."""
    df["risk"] = df["diameter"] * df["velocity_kmph"] / df["distance"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[["velocity_kmph", "diameter", "distance", "risk"]])
    return df, scaled_features

def perform_kmeans(scaled_features, max_clusters=10):
    """Find best K using silhouette score and return KMeans model."""
    score_list = []
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        score = silhouette_score(scaled_features, kmeans.labels_)
        score_list.append((k, score))
    best_k, best_score = max(score_list, key=lambda x: x[1])
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    kmeans.fit(scaled_features)
    return kmeans, best_k

def detect_anomalies(scaled_features, contamination=0.2):
    """Return anomaly predictions (-1 for anomaly)."""
    forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    forest.fit(scaled_features)
    return forest.predict(scaled_features)

def plot_3d(df, df_anomaly):
    """Plot 3D scatter of NEO clusters and anomalies using Plotly."""
    df_normal = df[df["Anomaly"] != -1].copy()
    df_normal["cluster"] = df_normal["cluster"].astype(str)

    fig = px.scatter_3d(
        df_normal,
        x="velocity_kmph",
        y="diameter",
        z="distance",
        color="cluster",
        opacity=0.7,
        title="🪐 OrbTrack: NEO Clusters vs Anomalies"
    )

    anomaly_trace = go.Scatter3d(
        x=df_anomaly["velocity_kmph"],
        y=df_anomaly["diameter"],
        z=df_anomaly["distance"],
        mode="markers",
        name="Anomalies",
        marker=dict(size=6, color="red", line=dict(width=1, color="black")),
        opacity=1.0
    )
    fig.add_trace(anomaly_trace)

    fig.update_layout(
        scene=dict(
            xaxis_title="Velocity (km/h)",
            yaxis_title="Diameter (km)",
            zaxis_title="Distance (km)"
        ),
        legend=dict(
            title="Legend",
            itemsizing="constant",
            x=0.85,
            y=0.95
        ),
        template="plotly_white"
    )
    return fig

#Streamlit App 
st.title("🪐 AsteRisk: NASA Near-Earth Object Analyzer")


st.sidebar.header("Fetch NEO Data")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")
if st.sidebar.button("Fetch Data"):
    fetch_nasa_data(str(start_date), str(end_date))


df = load_data()
df, scaled = preprocess_data(df)

# Clustering & anomaly detection
kmeans_model, best_k = perform_kmeans(scaled)
df["cluster"] = kmeans_model.labels_
df["Anomaly"] = detect_anomalies(scaled)
df_anomaly = df[df["Anomaly"] == -1]


st.sidebar.header("Filter NEOs")


if not df.empty:
    vel_min, vel_max = float(df["velocity_kmph"].min()), float(df["velocity_kmph"].max())
    dist_min, dist_max = float(df["distance"].min()), float(df["distance"].max())
    diam_min, diam_max = float(df["diameter"].min()), float(df["diameter"].max())

   
    vel_min, vel_max = max(0.0, vel_min * 0.9), vel_max * 1.1
    dist_min, dist_max = max(0.0, dist_min * 0.9), dist_max * 1.1
    diam_min, diam_max = max(0.0, diam_min * 0.9), diam_max * 1.1

    vel_range = st.sidebar.slider(
        "Velocity (kmph)", vel_min, vel_max, (vel_min, vel_max), step=(vel_max - vel_min) / 100
    )
    dist_range = st.sidebar.slider(
        "Distance (km)", dist_min, dist_max, (dist_min, dist_max), step=(dist_max - dist_min) / 100
    )
    diam_range = st.sidebar.slider(
        "Diameter (km)", diam_min, diam_max, (diam_min, diam_max), step=(diam_max - diam_min) / 100
    )

   
    df_filtered = df[
        (df["velocity_kmph"] >= vel_range[0]) & (df["velocity_kmph"] <= vel_range[1]) &
        (df["distance"] >= dist_range[0]) & (df["distance"] <= dist_range[1]) &
        (df["diameter"] >= diam_range[0]) & (df["diameter"] <= diam_range[1])
    ]
    df_anomaly_filtered = df_filtered[df_filtered["Anomaly"] == -1]

else:
    st.warning("No NEO data available. Please fetch data first.")
    df_filtered, df_anomaly_filtered = df.copy(), df.copy()


st.subheader("Filtered NEO Data")
st.dataframe(df_filtered)

st.subheader("3D Cluster & Anomaly Visualization")
fig = plot_3d(df_filtered, df_anomaly_filtered)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top 5 Risky NEOs")
top_risks = df_filtered.sort_values(by="risk", ascending=False).head(5)
st.table(top_risks[["name", "velocity_kmph", "diameter", "distance", "risk", "hazard"]])
