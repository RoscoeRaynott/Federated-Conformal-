# Streamlit + Flower + OpenRouter: Federated Conformal Clustering with Real Gene Names

import streamlit as st
import pandas as pd
import numpy as np
import random
import flwr as fl
import threading
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
import requests

# -----------------------
# Simulate Federated Data
# -----------------------
def load_real_gene_names(n_features=50):
    # Sample actual gene names from a static list (for demo purposes)
    known_genes = ["TP53", "EGFR", "VEGFA", "IL6", "TNF", "BRCA1", "APOE", "ESR1", "CRP", "IFNG",
                   "CDKN2A", "MTHFR", "AKT1", "BRAF", "CTNNB1", "FTO", "INS", "JAK2", "KIT", "MAPK1",
                   "MTOR", "NOS3", "PIK3CA", "PTEN", "RAF1", "TGFB1", "TGFBR2", "TSC1", "TSC2", "PDGFRA",
                   "NFKB1", "SMAD4", "CCND1", "FGFR1", "ABL1", "EGF", "CXCL8", "IL10", "RELA", "STAT3",
                   "CXCL10", "CDK4", "CDK6", "CASP3", "CASP8", "CD19", "CD8A", "FOXP3", "GATA3", "MYC"]
    return known_genes[:n_features]

def simulate_federated_data(n_centers=3, n_samples=100, n_features=50):
    genes = load_real_gene_names(n_features)
    data_centers = []
    for i in range(n_centers):
        mean_shift = np.random.normal(loc=0.0, scale=1.5, size=n_features)
        X = np.random.normal(loc=mean_shift, scale=1.0, size=(n_samples, n_features))
        df = pd.DataFrame(X, columns=genes)
        df["Center"] = f"Center_{i+1}"
        data_centers.append(df)
    return data_centers, genes

# -----------------------
# Federated Server Logic
# -----------------------
def start_flower_server():
    def fit_round(server_round):
        return {}

    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server("0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=1), strategy=strategy)

# -----------------------
# Conformal + Clustering
# -----------------------
def run_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    return kmeans.cluster_centers_, kmeans.predict(data)

def compute_conformal_scores(X, centroids):
    closest, distances = pairwise_distances_argmin_min(X, centroids)
    return distances, closest

# -----------------------
# OpenRouter Free Model
# -----------------------
def annotate_cluster(gene_list):
    headers = {"Authorization": "Bearer YOUR_FREE_OPENROUTER_API_KEY"}
    prompt = f"Summarize this gene list into a biological theme: {gene_list}"
    data = {
        "model": "mistral/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return "Annotation unavailable."

# -----------------------
# Streamlit UI
# -----------------------
st.title("Federated Conformal Clustering for Biomarker Discovery")
n_centers = st.sidebar.slider("Number of Centers", 2, 5, 3)
n_clusters = st.sidebar.slider("Number of Clusters", 2, 5, 3)

# Start federated server in background thread (for demo simulation)
threading.Thread(target=start_flower_server, daemon=True).start()
time.sleep(2)  # give the server time to initialize

# Simulate federated data
data_centers, gene_names = simulate_federated_data(n_centers)
full_data = pd.concat(data_centers, ignore_index=True)
X = full_data.drop(columns=["Center"]).values

# Run federated-style clustering
centroids, labels = run_kmeans(X, n_clusters=n_clusters)
distances, assigned_clusters = compute_conformal_scores(X, centroids)

# Prepare output
full_data["Cluster"] = assigned_clusters
full_data["ConformalScore"] = distances

# PCA projection
pca = PCA(n_components=2)
proj = pca.fit_transform(X)
st.scatter_chart(pd.DataFrame({"PC1": proj[:, 0], "PC2": proj[:, 1], "Cluster": assigned_clusters}))

# Cluster annotations
st.subheader("Cluster Annotations (via OpenRouter)")
for i in range(n_clusters):
    top_genes = full_data[full_data["Cluster"] == i].mean().sort_values(ascending=False).head(5).index.tolist()
    if st.button(f"Annotate Cluster {i}"):
        summary = annotate_cluster(top_genes)
        st.markdown(f"**Cluster {i} Annotation:** {summary}")
