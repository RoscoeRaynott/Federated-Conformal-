import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # For heatmap
import flwr as fl
import threading
import time
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering # Added missing imports from original context
from sklearn.mixture import GaussianMixture # Added missing import
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
import requests
import altair as alt
from sklearn import metrics
import os # To check for example file existence

# -----------------------
# Cached Simulation
# -----------------------
@st.cache_data # Keep this for simulating the base data
def simulate_federated_data_cached(n_centers_sim, n_samples_sim, n_features_sim):
    np.random.seed(42)
    # Expanded gene list for more features if needed, but will be sliced by n_features_sim
    available_genes = ["TP53","EGFR","VEGFA","IL6","TNF","BRCA1","APOE","ESR1","CRP","IFNG",
                       "CDKN2A","MTHFR","AKT1","BRAF","CTNNB1","FTO","INS","JAK2","KIT","MAPK1",
                       "MTOR","NOS3","PIK3CA","PTEN","RAF1","TGFB1","TGFBR2","TSC1","TSC2","PDGFRA",
                       "NFKB1","SMAD4","CCND1","FGFR1","ABL1","EGF","CXCL8","IL10","RELA","STAT3",
                       "CXCL10","CDK4","CDK6","CASP3","CASP8","CD19","CD8A","FOXP3","GATA3","MYC"]
    
    if n_features_sim > len(available_genes):
        st.warning(f"Requested {n_features_sim} features, but only {len(available_genes)} unique gene names are available. Using all available.")
        n_features_sim = len(available_genes)
        
    selected_genes = available_genes[:n_features_sim]
    
    data_centers_list = []
    for i in range(n_centers_sim):
        mean_shift = np.random.normal(0, 1.5, n_features_sim)
        X = np.random.normal(loc=mean_shift, scale=1.0, size=(n_samples_sim, n_features_sim))
        df = pd.DataFrame(X, columns=selected_genes)
        df["Center"] = f"Center_{i+1}"
        df["SampleID"] = [f"S{i+1}_{j+1}" for j in range(n_samples_sim)]
        data_centers_list.append(df)
    return data_centers_list, selected_genes

# -----------------------
# Streamlit UI: Setup
# -----------------------
st.set_page_config(page_title="Federated Conformal Clustering", layout="wide")
st.title("🧬 Federated Conformal Clustering for Biomarker Discovery")

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Simulation & Model Parameters")

# Input parameters with explicit keys
n_centers_input = st.sidebar.slider("Number of Data Centers (N)", 2, 5, 3, key="n_centers_input")
n_samples_input = st.sidebar.slider("Samples per Center (M)", 10, 200, 50, key="n_samples_input") # Reduced default for speed
n_features_input = st.sidebar.slider("Number of Genes (G)", 5, 50, 10, key="n_features_input") # Reduced default

# Initialize session state for data if it doesn't exist
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'sim_params' not in st.session_state:
    st.session_state.sim_params = {}

# Button to generate/regenerate data
if st.sidebar.button("Generate/Update Data", key="generate_data_btn"):
    st.session_state.data_centers, st.session_state.gene_names = simulate_federated_data_cached(
        n_centers_input, n_samples_input, n_features_input
    )
    st.session_state.data_generated = True
    st.session_state.sim_params = {
        'n_centers': n_centers_input, 
        'n_samples': n_samples_input, 
        'n_features': n_features_input
    }
    st.sidebar.success("Data generated/updated!")

# Ensure data is generated on first run or if parameters changed and button not clicked yet
current_params = {'n_centers': n_centers_input, 'n_samples': n_samples_input, 'n_features': n_features_input}
if not st.session_state.data_generated or st.session_state.sim_params != current_params:
    st.session_state.data_centers, st.session_state.gene_names = simulate_federated_data_cached(
        n_centers_input, n_samples_input, n_features_input
    )
    st.session_state.data_generated = True
    st.session_state.sim_params = current_params
    if not st.session_state.data_generated: # If it's the very first run
         st.sidebar.info("Default data loaded. Click 'Generate/Update Data' to use new parameters.")


# Display data preview if data exists
if st.session_state.data_generated:
    st.subheader(f"Preview: Data from {st.session_state.data_centers[0]['Center'].iloc[0]}")
    st.dataframe(st.session_state.data_centers[0].head())
    
    # Heatmap Preview
    if st.checkbox("Show Heatmap Preview of Center 1 Data (first 10 samples)", key="show_heatmap_preview"):
        fig_preview, ax_preview = plt.subplots()
        num_genes_to_preview = len(st.session_state.gene_names)
        data_to_plot = st.session_state.data_centers[0][st.session_state.gene_names].iloc[:10]
        
        sns.heatmap(data_to_plot, ax=ax_preview, cmap="viridis", yticklabels=st.session_state.data_centers[0]["SampleID"][:10].tolist())
        ax_preview.set_title(f"Gene Expression Heatmap (Center 1, Top 10 Samples)")
        ax_preview.set_xlabel("Genes")
        ax_preview.set_ylabel("SampleID")
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig_preview)
else:
    st.info("Click 'Generate/Update Data' in the sidebar to begin.")
    st.stop()


# --- Clustering and FL Parameters ---
st.sidebar.header("🔬 Clustering & Federated Learning")
n_clusters_input = st.sidebar.slider("Number of Clusters (K)", 2, 5, 3, key="n_clusters_k")
num_rounds_input = st.sidebar.slider("Federated Rounds", 1, 5, 1, key="num_rounds_fl") # Reduced default
conf_level_input = st.sidebar.slider("Conformal Confidence Level (α)", 0.50, 0.99, 0.90, step=0.01, key="conf_level_alpha")

st.sidebar.header("📊 Visualization & Metrics")
plot_type_input = st.sidebar.radio("PCA Plot Type", ["Connected Scatter", "Pure Scatter (Altair)"], key="plot_type_viz")
metric_choice_input = st.sidebar.selectbox(
    "Clustering Quality Metric",
    ["Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Index", "Inertia"], # Inertia can be tricky with federated
    key="metric_choice_viz"
)

# -----------------------
# Flower Client Definition
# -----------------------
class ClusterClient(fl.client.NumPyClient):
    def __init__(self, client_data: np.ndarray, num_clusters: int):
        self.data = client_data
        self.n_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')

    def get_parameters(self, config): # Added config argument
        # Clients typically don't send parameters before first fit in FedAvg for K-Means centroids
        return [] 

    def fit(self, parameters, config):
        # parameters might be global centroids from server if strategy sends them
        # For simple FedAvg K-Means, each client fits its own data
        self.kmeans.fit(self.data)
        return [self.kmeans.cluster_centers_], len(self.data), {}

    def evaluate(self, parameters, config):
        # Evaluation could involve local inertia or other metrics if needed
        # For this example, returning a dummy loss
        return 0.0, len(self.data), {"local_samples": len(self.data)}

# -----------------------
# Flower Server Setup
# -----------------------
# Strategy to capture aggregated centroids
class SaveCentroidsStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs): # Pass kwargs to parent
        super().__init__(**kwargs)
        self.aggregated_centroids_list = [] # Store history if needed

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            # Assuming parameters are a list of ndarrays (centroids)
            self.aggregated_centroids_list.append(aggregated_parameters[0]) 
        return aggregated_parameters, aggregated_metrics

# Global variable to hold the strategy instance to access centroids later
# This is a bit of a hack for Streamlit's execution model.
# A more robust way would involve inter-process communication or a proper state management for the server.
if 'fl_strategy' not in st.session_state:
    st.session_state.fl_strategy = SaveCentroidsStrategy(
        min_fit_clients=n_centers_input, # Wait for all clients in this setup
        min_available_clients=n_centers_input,
    )

# --- Main Analysis Button ---
if st.button("🚀 Run Federated Conformal Clustering", key="run_analysis_btn"):
    if not st.session_state.data_generated:
        st.error("Please generate data first using the sidebar button.")
        st.stop()

    st.warning(
        "Flower's `fl.server.start_server()` is deprecated and may cause issues (like `ValueError: signal only works in main thread`) "
        "when run in a thread within Streamlit. For stable use, run the Flower server (SuperLink) "
        "as a separate command-line process: `$ flower-superlink --insecure`"
    )
    
    # Start Flower server in a thread
    # Using st.session_state.fl_strategy to access centroids later
    server_thread = threading.Thread(
        target=lambda: fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds_input),
            strategy=st.session_state.fl_strategy 
        ),
        daemon=True
    )
    server_thread.start()
    st.info("Flower server starting in a background thread...")
    time.sleep(2) # Give server a moment to start

    # Start Flower clients in threads
    client_threads = []
    for center_idx, center_df_loop in enumerate(st.session_state.data_centers):
        # IMPORTANT: Pass only numeric gene data to the client
        client_data_np = center_df_loop[st.session_state.gene_names].values 
        
        ct = threading.Thread(
            target=fl.client.start_numpy_client, # Use start_numpy_client
            args=( "0.0.0.0:8080", ClusterClient(client_data_np, n_clusters_input)),
            daemon=True
        )
        client_threads.append(ct)
        ct.start()
        st.info(f"Flower client for Center_{center_idx+1} starting...")

    # Wait for FL rounds to complete (approximate)
    # A more robust solution would involve checking server status or client completion.
    total_wait_time = 3 * num_rounds_input + n_centers_input # Heuristic
    with st.spinner(f"Running {num_rounds_input} federated rounds... (waiting approx {total_wait_time}s)"):
        time.sleep(total_wait_time) 
    st.success("Federated learning simulation finished.")

    # -----------------------
    # Retrieve global centroids and perform final clustering
    # -----------------------
    st.subheader("Global Clustering Results")
    global_centroids = None
    if st.session_state.fl_strategy.aggregated_centroids_list:
        global_centroids = st.session_state.fl_strategy.aggregated_centroids_list[-1] # Get the last aggregated centroids
        st.write("Retrieved aggregated centroids from federated learning.")
    else:
        st.warning("Could not retrieve federated centroids. Performing centralized K-Means as a fallback.")
        # Fallback: recompute on full data (if federated part failed or no centroids were aggregated)
        combined_data_df_fallback = pd.concat(st.session_state.data_centers, ignore_index=True)
        X_fallback = combined_data_df_fallback[st.session_state.gene_names].values
        kmeans_fallback_model, global_centroids, _ = KMeans(n_clusters=n_clusters_input, random_state=42, n_init='auto').fit(X_fallback)
        # Store the model if needed for inertia later
        st.session_state.kmeans_model_for_metrics = kmeans_fallback_model


    if global_centroids is not None:
        # Final clustering on full data using the (federated or fallback) global centroids
        combined_data_df = pd.concat(st.session_state.data_centers, ignore_index=True)
        X_full = combined_data_df[st.session_state.gene_names].values
        
        # Assign labels based on the global_centroids
        final_labels, final_distances_to_centroids = pairwise_distances_argmin_min(X_full, global_centroids)
        combined_data_df["Cluster"] = final_labels
        
        # Conformal scores are the distances to the assigned centroid
        conformal_scores = final_distances_to_centroids 
        combined_data_df["ConformalScore"] = conformal_scores

        # -----------------------
        # Conformal calibration
        # -----------------------
        conformal_threshold = np.quantile(conformal_scores, conf_level_input)
        st.write(f"**Conformal Threshold at {int(conf_level_input*100)}% confidence:** {conformal_threshold:.3f}")
        
        combined_data_df["HighConfidence"] = combined_data_df["ConformalScore"] <= conformal_threshold
        
        st.write(f"Number of high-confidence samples: {combined_data_df['HighConfidence'].sum()}")
        st.write(f"Number of ambiguous/low-confidence samples: {(~combined_data_df['HighConfidence']).sum()}")

        # -----------------------
        # Compute chosen metric
        # -----------------------
        st.subheader(f"Clustering Quality: {metric_choice_input}")
        metric_value_display = "N/A"
        try:
            if metric_choice_input == "Inertia":
                # Inertia is sum of squared distances of samples to their closest cluster center.
                # If we used fallback, we have the model. Otherwise, use conformal_scores.
                if 'kmeans_model_for_metrics' in st.session_state and not st.session_state.fl_strategy.aggregated_centroids_list:
                    metric_value_display = st.session_state.kmeans_model_for_metrics.inertia_
                else: # Calculate from distances if federated or if model not stored
                    metric_value_display = np.sum(conformal_scores**2)
            elif len(np.unique(final_labels)) > 1: # Metrics like Silhouette need at least 2 clusters
                if metric_choice_input == "Silhouette Score":
                    metric_value_display = metrics.silhouette_score(X_full, final_labels)
                elif metric_choice_input == "Calinski-Harabasz Index":
                    metric_value_display = metrics.calinski_harabasz_score(X_full, final_labels)
                elif metric_choice_input == "Davies-Bouldin Index":
                    metric_value_display = metrics.davies_bouldin_score(X_full, final_labels)
            else:
                metric_value_display = "N/A (requires >1 cluster for this metric)"
            st.write(f"**{metric_choice_input}:** {metric_value_display:.2f}" if isinstance(metric_value_display, (int, float)) else metric_value_display)
        except Exception as e:
            st.error(f"Error calculating metric '{metric_choice_input}': {e}")


        # -----------------------
        # PCA projection and Plotting
        # -----------------------
        st.subheader("PCA Projection of Clustered Data")
        try:
            pca_model = PCA(n_components=2)
            projected_data = pca_model.fit_transform(X_full)
            
            plot_df_pca = pd.DataFrame({
                "PC1": projected_data[:, 0], 
                "PC2": projected_data[:, 1], 
                "Cluster": final_labels.astype(str), # For Altair color encoding
                "HighConfidence": combined_data_df["HighConfidence"]
            })

            if plot_type_input == "Connected Scatter": # Using Streamlit's native scatter
                st.scatter_chart(plot_df_pca, x="PC1", y="PC2", color="Cluster")
            else: # Pure Scatter (Altair)
                chart = (
                    alt.Chart(plot_df_pca)
                    .mark_circle(size=60)
                    .encode(
                        x=alt.X("PC1", title="Principal Component 1"), 
                        y=alt.Y("PC2", title="Principal Component 2"), 
                        color=alt.Color("Cluster:N", title="Cluster"), # :N treats it as nominal
                        opacity=alt.condition(
                            alt.datum.HighConfidence, # altair condition for boolean
                            alt.value(0.9), # opacity for True
                            alt.value(0.3)  # opacity for False
                        ),
                        tooltip=["PC1", "PC2", "Cluster", "HighConfidence"]
                    )
                    .properties(title="PCA of Clustered Data (High/Low Confidence)", width=700, height=500)
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error during PCA and plotting: {e}")

        # -----------------------
        # Cluster Annotations (Optional - requires API key)
        # -----------------------
        st.subheader("Cluster Annotations (via OpenRouter LLM)")
        if not st.secrets.get("OPENROUTER_API_KEY"):
            st.warning("OpenRouter API key not found in secrets. Annotation feature disabled.")
        else:
            for i in range(n_clusters_input):
                cluster_subset_df = combined_data_df[combined_data_df["Cluster"] == i]
                # Get mean expression for genes in this cluster
                # Ensure we only try to get means of numeric columns (gene names)
                cluster_gene_means = cluster_subset_df[st.session_state.gene_names].mean()
                top_genes_list = cluster_gene_means.sort_values(ascending=False).head(5).index.tolist()
                
                if st.button(f"Annotate Cluster {i} (Top 5 mean expression genes: {', '.join(top_genes_list)})", key=f"annotate_btn_{i}"):
                    if top_genes_list:
                        with st.spinner(f"Annotating Cluster {i}..."):
                            annotation = "Annotation unavailable (LLM function not implemented or error)."
                            # Placeholder for your actual annotate_cluster function
                            # annotation = annotate_cluster(top_genes_list) # Assuming annotate_cluster is defined
                            st.markdown(f"**Cluster {i} Annotation:** {annotation}")
                    else:
                        st.write(f"Cluster {i} has no gene data to annotate or is empty.")
    else:
        st.error("Global centroids could not be determined. Cannot proceed with final clustering and visualization.")

# Helper function for OpenRouter (if you want to integrate it)
# This was in your original code, ensure st.secrets["OPENROUTER_API_KEY"] is set
def annotate_cluster_llm(gene_list_to_annotate): # Renamed to avoid conflict if you have another
    api_key_llm = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key_llm:
        return "OpenRouter API key not configured."
    
    headers = {"Authorization": f"Bearer {api_key_llm}"}
    prompt_content = f"Provide a concise biological theme or pathway related to this list of genes: {', '.join(gene_list_to_annotate)}. Focus on shared functions or roles."
    
    payload = {
        "model": "mistralai/mistral-7b-instruct:free", # Example free model
        "messages": [{"role": "user", "content": prompt_content}],
        "max_tokens": 150
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            st.error(f"OpenRouter API error {response.status_code}: {response.text}")
            return "Annotation unavailable due to API error."
    except Exception as e:
        st.error(f"Error connecting to OpenRouter: {e}")
        return "Annotation unavailable due to connection error."
