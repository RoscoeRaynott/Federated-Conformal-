import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import flwr as fl
import threading
import time
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
import requests
import altair as alt
from sklearn import metrics
import os
import plotly.express as px

# -----------------------
# Cached Simulation
# -----------------------
@st.cache_data
def simulate_federated_data_cached(n_centers_sim, n_samples_sim, n_features_sim):
    np.random.seed(42)
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
        X_data = np.random.normal(loc=mean_shift, scale=1.0, size=(n_samples_sim, n_features_sim))
        df = pd.DataFrame(X_data, columns=selected_genes)
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

n_centers_input = st.sidebar.slider("Number of Data Centers (N)", 2, 5, 3, key="n_centers_input")
n_samples_input = st.sidebar.slider("Samples per Center (M)", 10, 200, 50, key="n_samples_input")
n_features_input = st.sidebar.slider("Number of Genes (G)", 5, 50, 10, key="n_features_input")

if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'sim_params' not in st.session_state:
    st.session_state.sim_params = {}
if 'data_centers' not in st.session_state:
    st.session_state.data_centers = []
if 'gene_names' not in st.session_state:
    st.session_state.gene_names = []
if 'analysis_run_complete' not in st.session_state:
    st.session_state.analysis_run_complete = False

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
    st.session_state.analysis_run_complete = False
    st.sidebar.success("Data generated/updated!")

current_params = {'n_centers': n_centers_input, 'n_samples': n_samples_input, 'n_features': n_features_input}
if not st.session_state.data_generated or st.session_state.sim_params != current_params:
    st.session_state.data_centers, st.session_state.gene_names = simulate_federated_data_cached(
        n_centers_input, n_samples_input, n_features_input
    )
    st.session_state.data_generated = True
    st.session_state.sim_params = current_params
    st.session_state.analysis_run_complete = False
    if not st.session_state.get('fl_strategy'):
         st.sidebar.info("Default data loaded. Change parameters and click 'Generate/Update Data' if needed.")

if st.session_state.data_generated and st.session_state.data_centers:
    st.subheader(f"Preview: Data from {st.session_state.data_centers[0]['Center'].iloc[0]}")
    st.dataframe(st.session_state.data_centers[0].head())

    if st.checkbox("Show Heatmap Preview of Center 1 Data (first 10 samples)", key="show_heatmap_preview"):
        fig_preview, ax_preview = plt.subplots()
        if st.session_state.gene_names:
            data_to_plot = st.session_state.data_centers[0][st.session_state.gene_names].iloc[:10]
            sns.heatmap(data_to_plot, ax=ax_preview, cmap="viridis", yticklabels=st.session_state.data_centers[0]["SampleID"][:10].tolist())
            ax_preview.set_title(f"Gene Expression Heatmap (Center 1, Top 10 Samples)")
            ax_preview.set_xlabel("Genes")
            ax_preview.set_ylabel("SampleID")
            plt.xticks(rotation=90, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig_preview)
        else:
            st.warning("Gene names not available for heatmap preview.")
else:
    st.info("Click 'Generate/Update Data' in the sidebar to simulate data and begin.")
    st.stop()

st.sidebar.header("🔬 Clustering & Federated Learning")
n_clusters_input = st.sidebar.slider("Number of Clusters (K)", 2, 5, 3, key="n_clusters_k")
num_rounds_input = st.sidebar.slider("Federated Rounds", 1, 5, 1, key="num_rounds_fl")
conf_level_input = st.sidebar.slider("Conformal Confidence Level (α)", 0.50, 0.99, 0.90, step=0.01, key="conf_level_alpha")

st.sidebar.header("📊 Visualization & Metrics")
metric_choice_input = st.sidebar.selectbox(
    "Clustering Quality Metric",
    ["Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Index", "Inertia"],
    key="metric_choice_viz"
)
additional_viz_choice = st.sidebar.selectbox(
    "Additional Cluster Visualization",
    ["None", "Mean Cluster Curves", "Parallel Coordinates", "Heatmap of Mean Profiles", "3D PCA Scatter", "Gene BoxPlots", "PCA Scatter (Streamlit Native)"],
    key="additional_viz"
)

class ClusterClient(fl.client.NumPyClient):
    def __init__(self, client_data: np.ndarray, num_clusters: int):
        self.data = client_data
        self.n_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')

    def get_parameters(self, config):
        return [self.kmeans.cluster_centers_] if hasattr(self.kmeans, 'cluster_centers_') else []

    def fit(self, parameters, config):
        self.kmeans.fit(self.data)
        return [self.kmeans.cluster_centers_], len(self.data), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.data), {"local_samples": len(self.data)}

class SaveCentroidsStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aggregated_centroids_list = []

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None and isinstance(aggregated_parameters, list) and len(aggregated_parameters) > 0:
            self.aggregated_centroids_list.append(aggregated_parameters[0])
        return aggregated_parameters, aggregated_metrics

if 'fl_strategy' not in st.session_state:
    st.session_state.fl_strategy = SaveCentroidsStrategy(
        min_fit_clients=n_centers_input,
        min_available_clients=n_centers_input,
    )
else:
    st.session_state.fl_strategy.min_fit_clients = n_centers_input
    st.session_state.fl_strategy.min_available_clients = n_centers_input

def annotate_cluster_llm(gene_list_to_annotate):
    api_key_llm = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key_llm:
        return "OpenRouter API key not configured."
    headers = {"Authorization": f"Bearer {api_key_llm}"}
    prompt_content = f"Provide a concise biological theme or pathway related to this list of genes: {', '.join(gene_list_to_annotate)}. Focus on shared functions or roles."
    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [{"role": "user", "content": prompt_content}],
        "max_tokens": 150
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to OpenRouter: {e}")
        return "Annotation unavailable due to connection error."
    except KeyError:
        st.error(f"Unexpected response structure from OpenRouter: {response.text}")
        return "Annotation unavailable due to API response format."
    except Exception as e:
        st.error(f"An unexpected error occurred during LLM annotation: {e}")
        return "Annotation unavailable."

if st.button("🚀 Run Federated Conformal Clustering", key="run_analysis_btn"):
    if not st.session_state.data_generated or not st.session_state.data_centers:
        st.error("Please generate data first using the sidebar button.")
        st.stop()

    st.warning(
        "Flower's `fl.server.start_server()` is deprecated and may cause issues (like `ValueError: signal only works in main thread`) "
        "when run in a thread within Streamlit. For stable use, run the Flower server (SuperLink) "
        "as a separate command-line process: `$ flower-superlink --insecure`"
    )

    st.session_state.fl_strategy = SaveCentroidsStrategy(
        min_fit_clients=n_centers_input,
        min_available_clients=n_centers_input,
    )

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
    time.sleep(2)

    client_threads = []
    for center_idx, center_df_loop in enumerate(st.session_state.data_centers):
        if st.session_state.gene_names:
            client_data_np = center_df_loop[st.session_state.gene_names].values
            ct = threading.Thread(
                target=fl.client.start_numpy_client,
                args=("0.0.0.0:8080", ClusterClient(client_data_np, n_clusters_input)),
                daemon=True
            )
            client_threads.append(ct)
            ct.start()
            st.info(f"Flower client for Center_{center_idx+1} starting...")
        else:
            st.error("Gene names are missing. Cannot start clients.")
            st.stop()

    total_wait_time = 3 * num_rounds_input + n_centers_input + 2
    with st.spinner(f"Running {num_rounds_input} federated rounds... (waiting approx {total_wait_time}s)"):
        time.sleep(total_wait_time)
    st.success("Federated learning simulation finished.")

    global_centroids = None
    if st.session_state.fl_strategy.aggregated_centroids_list:
        global_centroids = st.session_state.fl_strategy.aggregated_centroids_list[-1]
    else:
        st.warning("Could not retrieve federated centroids. Performing centralized K-Means as a fallback.")
        combined_data_df_fallback = pd.concat(st.session_state.data_centers, ignore_index=True)
        X_fallback = combined_data_df_fallback[st.session_state.gene_names].values
        kmeans_fallback_model = KMeans(n_clusters=n_clusters_input, random_state=42, n_init='auto').fit(X_fallback)
        global_centroids = kmeans_fallback_model.cluster_centers_
        st.session_state.kmeans_model_for_metrics = kmeans_fallback_model

    if global_centroids is not None:
        combined_data_df = pd.concat(st.session_state.data_centers, ignore_index=True)
        X_full = combined_data_df[st.session_state.gene_names].values
        final_labels, final_distances_to_centroids = pairwise_distances_argmin_min(X_full, global_centroids)
        combined_data_df["Cluster"] = final_labels
        conformal_scores = final_distances_to_centroids
        combined_data_df["ConformalScore"] = conformal_scores
        conformal_threshold = np.quantile(conformal_scores, conf_level_input)
        combined_data_df["HighConfidence"] = combined_data_df["ConformalScore"] <= conformal_threshold

        st.session_state.combined_data_df_clustered = combined_data_df
        st.session_state.global_centroids_final = global_centroids
        st.session_state.final_labels_for_plot = final_labels
        st.session_state.X_full_for_plot = X_full
        st.session_state.analysis_run_complete = True
    else:
        st.error("Global centroids could not be determined. Cannot proceed.")
        st.session_state.analysis_run_complete = False

if st.session_state.get('analysis_run_complete', False):
    combined_data_df_to_use = st.session_state.combined_data_df_clustered
    final_labels_to_use = st.session_state.final_labels_for_plot
    X_full_to_use = st.session_state.X_full_for_plot
    gene_names_to_use = st.session_state.gene_names

    st.subheader("Global Clustering Results")
    st.write(f"**Conformal Threshold at {int(conf_level_input*100)}% confidence:** {combined_data_df_to_use['ConformalScore'].quantile(conf_level_input):.3f}")
    st.write(f"Number of high-confidence samples: {combined_data_df_to_use['HighConfidence'].sum()}")
    st.write(f"Number of ambiguous/low-confidence samples: {(~combined_data_df_to_use['HighConfidence']).sum()}")

    st.subheader(f"Clustering Quality: {metric_choice_input}")
    metric_value_display = "N/A"
    try:
        if metric_choice_input == "Inertia":
            if 'kmeans_model_for_metrics' in st.session_state and not st.session_state.fl_strategy.aggregated_centroids_list:
                metric_value_display = st.session_state.kmeans_model_for_metrics.inertia_
            else:
                metric_value_display = np.sum(combined_data_df_to_use["ConformalScore"]**2)
        elif len(np.unique(final_labels_to_use)) > 1:
            if metric_choice_input == "Silhouette Score":
                metric_value_display = metrics.silhouette_score(X_full_to_use, final_labels_to_use)
            elif metric_choice_input == "Calinski-Harabasz Index":
                metric_value_display = metrics.calinski_harabasz_score(X_full_to_use, final_labels_to_use)
            elif metric_choice_input == "Davies-Bouldin Index":
                metric_value_display = metrics.davies_bouldin_score(X_full_to_use, final_labels_to_use)
        else:
            metric_value_display = "N/A (requires >1 cluster for this metric)"
        st.write(f"**{metric_choice_input}:** {metric_value_display:.2f}" if isinstance(metric_value_display, (int, float)) else metric_value_display)
    except Exception as e:
        st.error(f"Error calculating metric '{metric_choice_input}': {e}")

    # Default PCA Plot (Altair)
    st.subheader("PCA Projection of Clustered Data (Altair)")
    try:
        pca_model = PCA(n_components=2)
        projected_data = pca_model.fit_transform(X_full_to_use)
        plot_df_pca = pd.DataFrame({
            "PC1": projected_data[:, 0],
            "PC2": projected_data[:, 1],
            "Cluster": final_labels_to_use.astype(str),
            "HighConfidence": combined_data_df_to_use["HighConfidence"]
        })
        chart = (
            alt.Chart(plot_df_pca)
            .mark_circle(size=60)
            .encode(
                x=alt.X("PC1", title="Principal Component 1"),
                y=alt.Y("PC2", title="Principal Component 2"),
                color=alt.Color("Cluster:N", title="Cluster"),
                opacity=alt.condition(alt.datum.HighConfidence, alt.value(0.9), alt.value(0.3)),
                tooltip=["PC1", "PC2", "Cluster", "HighConfidence"]
            )
            .properties(title="PCA of Clustered Data (High/Low Confidence)", width=700, height=500)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error during PCA and plotting: {e}")

    # --- Additional Visualizations ---
    if additional_viz_choice == "Mean Cluster Curves":
        st.subheader("Mean Curves per Cluster")
        unique_clusters = sorted(combined_data_df_to_use['Cluster'].unique())
        n_cols_mc = 2
        n_rows_mc = (len(unique_clusters) + n_cols_mc - 1) // n_cols_mc
        fig_mean_curves, axs_mc = plt.subplots(n_rows_mc, n_cols_mc, figsize=(n_cols_mc * 6, n_rows_mc * 4), squeeze=False)
        axs_mc_flat = axs_mc.flatten()
        for idx, cluster_id in enumerate(unique_clusters):
            if idx < len(axs_mc_flat):
                ax = axs_mc_flat[idx]
                cluster_data = combined_data_df_to_use[combined_data_df_to_use['Cluster'] == cluster_id][gene_names_to_use]
                if not cluster_data.empty:
                    mean_curve = cluster_data.mean(axis=0)
                    ax.plot(gene_names_to_use, mean_curve, color=f'C{cluster_id}', linewidth=2, label=f"Mean Cluster {cluster_id}")
                    if st.checkbox(f"Show individual curves for Cluster {cluster_id}", key=f"show_ind_c{cluster_id}"):
                        for _, row_val in cluster_data.iterrows():
                            ax.plot(gene_names_to_use, row_val, color=f'C{cluster_id}', alpha=0.2)
                    ax.set_title(f"Cluster {cluster_id} (N={len(cluster_data)})")
                    ax.set_xlabel("Genes / Features")
                    ax.set_ylabel("Expression Value")
                    ax.legend()
        for i in range(len(unique_clusters), len(axs_mc_flat)): fig_mean_curves.delaxes(axs_mc_flat[i])
        plt.tight_layout()
        st.pyplot(fig_mean_curves)

    elif additional_viz_choice == "Parallel Coordinates":
        st.subheader("Parallel Coordinates Plot of Clusters")
        num_genes_for_parallel = min(10, len(gene_names_to_use))
        genes_for_plot_parallel = gene_names_to_use[:num_genes_for_parallel]
        if not combined_data_df_to_use.empty and 'Cluster' in combined_data_df_to_use:
            df_for_parallel = combined_data_df_to_use.copy()
            df_for_parallel['Cluster'] = df_for_parallel['Cluster'].astype(str)
            fig_parallel = px.parallel_coordinates(
                df_for_parallel, color="Cluster", dimensions=genes_for_plot_parallel,
                title="Parallel Coordinates Plot by Cluster"
            )
            st.plotly_chart(fig_parallel, use_container_width=True)

    elif additional_viz_choice == "Heatmap of Mean Profiles":
        st.subheader("Heatmap of Mean Cluster Profiles")
        if not combined_data_df_to_use.empty and 'Cluster' in combined_data_df_to_use:
            mean_profiles = combined_data_df_to_use.groupby('Cluster')[gene_names_to_use].mean()
            if not mean_profiles.empty:
                fig_heatmap_mean, ax_heatmap_mean = plt.subplots(figsize=(10, max(4, len(mean_profiles) * 0.5)))
                sns.heatmap(mean_profiles, annot=True, fmt=".2f", cmap="viridis", ax=ax_heatmap_mean, cbar=True)
                ax_heatmap_mean.set_xlabel("Genes / Features")
                ax_heatmap_mean.set_ylabel("Cluster ID")
                ax_heatmap_mean.set_title("Mean Gene Expression Profiles per Cluster")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig_heatmap_mean)

    elif additional_viz_choice == "3D PCA Scatter":
        st.subheader("3D PCA Projection of Clustered Data")
        if X_full_to_use.shape[1] >= 3:
            try:
                pca_3d = PCA(n_components=3)
                projected_3d = pca_3d.fit_transform(X_full_to_use)
                df_3d_pca = pd.DataFrame({
                    "PC1": projected_3d[:, 0], "PC2": projected_3d[:, 1], "PC3": projected_3d[:, 2],
                    "Cluster": final_labels_to_use.astype(str),
                    "HighConfidence": combined_data_df_to_use["HighConfidence"]
                })
                df_3d_pca['MarkerOpacity'] = df_3d_pca['HighConfidence'].apply(lambda x: 0.9 if x else 0.3)
                fig_3d = px.scatter_3d(
                    df_3d_pca, x='PC1', y='PC2', z='PC3', color='Cluster',
                    opacity='MarkerOpacity',
                    title="3D PCA of Clustered Data", labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'},
                    hover_data=["Cluster", "HighConfidence"]
                )
                fig_3d.update_traces(marker=dict(size=5))
                fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig_3d, use_container_width=True)
            except Exception as e: st.error(f"Error during 3D PCA: {e}")
        else: st.warning("Need at least 3 features/genes for 3D PCA.")

    elif additional_viz_choice == "Gene BoxPlots":
        st.subheader("Gene Expression Distribution by Cluster (Box Plots)")
        num_genes_to_boxplot = min(5, len(gene_names_to_use))
        selected_genes_for_boxplot = st.multiselect(
            "Select genes for box plot:", options=gene_names_to_use,
            default=gene_names_to_use[:num_genes_to_boxplot], key="boxplot_genes_select"
        )
        if selected_genes_for_boxplot and not combined_data_df_to_use.empty and 'Cluster' in combined_data_df_to_use:
            n_cols_bp = 2
            n_rows_bp = (len(selected_genes_for_boxplot) + n_cols_bp - 1) // n_cols_bp
            fig_boxplots, axs_bp = plt.subplots(n_rows_bp, n_cols_bp, figsize=(n_cols_bp * 7, n_rows_bp * 5), squeeze=False)
            axs_bp_flat = axs_bp.flatten()
            for idx, gene_name in enumerate(selected_genes_for_boxplot):
                if idx < len(axs_bp_flat):
                    ax = axs_bp_flat[idx]
                    sns.boxplot(x='Cluster', y=gene_name, data=combined_data_df_to_use, ax=ax, palette="Set2")
                    sns.stripplot(x='Cluster', y=gene_name, data=combined_data_df_to_use, ax=ax, color=".25", alpha=0.5, dodge=True)
                    ax.set_title(f"Expression of {gene_name}")
            for i in range(len(selected_genes_for_boxplot), len(axs_bp_flat)): fig_boxplots.delaxes(axs_bp_flat[i])
            plt.tight_layout()
            st.pyplot(fig_boxplots)

    elif additional_viz_choice == "PCA Scatter (Streamlit Native)":
        st.subheader("PCA Projection of Clustered Data (Streamlit Native)")
        if 'X_full_for_plot' in st.session_state and 'final_labels_for_plot' in st.session_state:
            X_full_viz = st.session_state.X_full_for_plot
            final_labels_viz = st.session_state.final_labels_for_plot
            try:
                pca_model_st = PCA(n_components=2)
                projected_data_st = pca_model_st.fit_transform(X_full_viz)
                df_pca_st = pd.DataFrame({
                    "PC1": projected_data_st[:, 0],
                    "PC2": projected_data_st[:, 1],
                    "Cluster": final_labels_viz.astype(str)
                })
                st.scatter_chart(df_pca_st, x="PC1", y="PC2", color="Cluster")
            except Exception as e:
                st.error(f"Error generating Streamlit native PCA scatter plot: {e}")
        else:
            st.warning("Clustering results needed for Streamlit native PCA scatter plot are not available.")


    st.subheader("Cluster Annotations (via OpenRouter LLM)")
    if not st.secrets.get("OPENROUTER_API_KEY"):
        st.warning("OpenRouter API key not found in secrets. Annotation feature disabled.")
    else:
        for i in range(n_clusters_input): # Use the current slider value for n_clusters
            cluster_subset_df = combined_data_df_to_use[combined_data_df_to_use["Cluster"] == i]
            if gene_names_to_use and not cluster_subset_df.empty:
                cluster_gene_means = cluster_subset_df[gene_names_to_use].mean()
                top_genes_list = cluster_gene_means.sort_values(ascending=False).head(5).index.tolist()
                if st.button(f"Annotate Cluster {i} (Top 5 genes: {', '.join(top_genes_list)})", key=f"annotate_btn_{i}"):
                    if top_genes_list:
                        with st.spinner(f"Annotating Cluster {i}..."):
                            annotation = annotate_cluster_llm(top_genes_list)
                            st.markdown(f"**Cluster {i} Annotation:** {annotation}")
                            if 'cluster_annotations' not in st.session_state:
                                st.session_state.cluster_annotations = {}
                            st.session_state.cluster_annotations[i] = annotation
                    else:
                        st.write(f"Cluster {i} has no gene data to determine top genes for annotation.")
            elif cluster_subset_df.empty:
                 st.write(f"Cluster {i} is empty, skipping annotation.")
            else:
                st.warning("Gene names not available for cluster annotation.")
elif st.session_state.data_generated:
    st.info("Click '🚀 Run Federated Conformal Clustering' to start the analysis.")
