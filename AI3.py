# Streamlit + Flower + OpenRouter: Federated Conformal Clustering with Real Gene Names
import streamlit as st
import pandas as pd
import numpy as np
import flwr as fl
import threading
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
import requests
import altair as alt
from sklearn import metrics

# -----------------------
# Simulate Federated Data
# -----------------------
def load_real_gene_names(n_features=50):
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
# Flower Client Definition
# -----------------------
class ClusterClient(fl.client.NumPyClient):
    def __init__(self, data: np.ndarray, n_clusters: int):
        self.data = data
        self.n_clusters = n_clusters

    def get_parameters(self):
        return []

    def fit(self, parameters, config):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(self.data)
        return [kmeans.cluster_centers_], len(self.data), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.data), {}


def start_flower_server():
    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server("0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=1), strategy=strategy)


def start_flower_client(data: np.ndarray, n_clusters: int):
    client = ClusterClient(data, n_clusters)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

# -----------------------
# Conformal + Clustering
# -----------------------
def run_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    return kmeans, kmeans.cluster_centers_, kmeans.predict(data)


def compute_conformal_scores(X, centroids):
    _, distances = pairwise_distances_argmin_min(X, centroids)
    return distances

# -----------------------
# OpenRouter Free Model
# -----------------------
def annotate_cluster(gene_list):
    api_key = st.secrets["OPENROUTER_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = f"Summarize this gene list into a biological theme: {gene_list}"
    data = {"model": "mistralai/Mistral-7B-Instruct-v0.2", "messages": [{"role": "user", "content": prompt}], "max_tokens": 200}
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        st.error(f"OpenRouter API error {response.status_code}: {response.text}")
        return "Annotation unavailable."

# -----------------------
# Streamlit UI
# -----------------------
st.title("Federated Conformal Clustering for Biomarker Discovery")

# Sidebar options
n_centers = st.sidebar.slider("Number of Centers", 2, 5, 3)
n_clusters = st.sidebar.slider("Number of Clusters", 2, 5, 3)
plot_type = st.sidebar.radio("Plot Type", ["Connected Scatter", "Pure Scatter"])
metric_choice = st.sidebar.selectbox("Quality Metric", [
    "Inertia", "Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Index"
])

# -----------------------
# Streamlit UI: Settings
# -----------------------

# Federated Learning rounds selector
num_rounds = st.sidebar.slider("Federated Rounds", 1, 10, 1)

# Conformal confidence level
conf_level = st.sidebar.slider("Conformal Confidence", 0.50, 0.99, 0.90, step=0.01)

# Flower setup with dynamic rounds and strategy to capture centroids
class SaveCentroidsStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.aggregated_centroids = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        # aggregated[0] is combined parameters: list of ndarrays (centroids)
        self.aggregated_centroids = aggregated[0]
        return aggregated

strategy = SaveCentroidsStrategy()
# CHANGED: Use num_rounds from UI
threading.Thread(
    target=lambda: fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    ),
    daemon=True
).start()

data_centers, gene_names = simulate_federated_data(n_centers)
# CHANGED: spawn clients
for center_df in data_centers:
    threading.Thread(
        target=start_flower_client,
        args=(center_df.drop(columns=["Center"]).values, n_clusters),
        daemon=True
    ).start()

time.sleep(2 * num_rounds)  # wait for all FL rounds to complete

# -----------------------
# Retrieve global centroids
# -----------------------
if strategy.aggregated_centroids is not None:
    # CHANGED: fetch from server directly
    centroids = strategy.aggregated_centroids[0]
else:
    # fallback: recompute on full data
    full_data = pd.concat(data_centers, ignore_index=True)
    X = full_data.drop(columns=["Center"]).values
    _, centroids, _ = run_kmeans(X, n_clusters)

# Final clustering on full data using aggregated centroids
full_data = pd.concat(data_centers, ignore_index=True)
X = full_data.drop(columns=["Center"]).values
labels = np.argmin(((X[:, None, :] - centroids[None, :, :])**2).sum(axis=2), axis=1)
full_data["Cluster"] = labels

distances = compute_conformal_scores(X, centroids)
full_data["ConformalScore"] = distances

# -----------------------
# Conformal calibration
# -----------------------
# CHANGED: compute threshold for given confidence level
threshold = np.quantile(distances, conf_level)
st.subheader(f"Conformal threshold at {int(conf_level*100)}%: {threshold:.2f}")

# Mark high-confidence vs. ambiguous
full_data["HighConfidence"] = full_data["ConformalScore"] <= threshold

# Compute chosen metric

full_data = pd.concat(data_centers, ignore_index=True)
X = full_data.drop(columns=["Center"]).values
kmeans_model, centroids, labels = run_kmeans(X, n_clusters)
full_data["Cluster"] = labels
full_data["ConformalScore"] = compute_conformal_scores(X, centroids)

# Compute chosen metric
if metric_choice == "Inertia":
    metric_value = kmeans_model.inertia_
elif metric_choice == "Silhouette Score":
    metric_value = metrics.silhouette_score(X, labels)
elif metric_choice == "Calinski-Harabasz Index":
    metric_value = metrics.calinski_harabasz_score(X, labels)
else:
    metric_value = metrics.davies_bouldin_score(X, labels)

st.subheader(f"Clustering Quality: {metric_choice}")
st.write(f"**{metric_choice}:** {metric_value:.2f}")

# PCA projection
pca = PCA(n_components=2)
proj = pca.fit_transform(X)
plot_df = pd.DataFrame({"PC1": proj[:, 0], "PC2": proj[:, 1], "Cluster": labels.astype(str)})

# Render plot
if plot_type == "Connected Scatter":
    st.subheader("Connected Scatter: PC1 vs PC2")
    st.scatter_chart(pd.DataFrame({"PC1": proj[:,0], "PC2": proj[:,1], "Cluster": labels}))
else:
    st.subheader("Pure Scatter: PC1 vs PC2 (Altair)")
    chart = (
        alt.Chart(plot_df)
        .mark_circle(size=60)
        .encode(x=alt.X("PC1", title="PC1"), y=alt.Y("PC2", title="PC2"), color=alt.Color("Cluster", title="Cluster"), tooltip=["PC1", "PC2", "Cluster"])
        .properties(width=600, height=400)
    )
    st.altair_chart(chart, use_container_width=True)

# Cluster Annotations
st.subheader("Cluster Annotations (via OpenRouter)")
for i in range(n_clusters):
    cluster_data = full_data[full_data["Cluster"] == i]
    top_genes = cluster_data.select_dtypes(include=np.number).mean().sort_values(ascending=False).head(5).index.tolist()
    if st.button(f"Annotate Cluster {i}"):
        st.markdown(f"**Cluster {i} Annotation:** {annotate_cluster(top_genes)}")
