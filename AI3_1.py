import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import flwr as fl # REMOVED
# import threading # REMOVED for Flower, might be used for other UI things if added later
import time
from sklearn.cluster import KMeans # DBSCAN, AgglomerativeClustering not used in this specific path now
# from sklearn.mixture import GaussianMixture # Not used in this specific path
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
import requests
import altair as alt
from sklearn import metrics
import os
import plotly.express as px
# import pickle # REMOVED (was for Flower strategy)

# -----------------------
# Cached Simulation
# -----------------------
@st.cache_data
def simulate_federated_data_cached(n_centers_sim, n_samples_sim, n_features_sim, heterogeneity_factor=0.95): # Add heterogeneity factor
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

    K_true = 3
    global_centroids = np.random.normal(0, 2.0, size=(K_true, n_features_sim)) # Spread out global centroids

    data_centers_list = []
    for i in range(n_centers_sim):
        associated_cluster = i % K_true
        
        # Make preference even stronger for the associated cluster
        if K_true > 1:
            prob_others = (1.0 - heterogeneity_factor) / (K_true - 1)
            p = [heterogeneity_factor if k == associated_cluster else prob_others for k in range(K_true)]
        else: # Only one true cluster
            p = [1.0]

        X_data_center = [] # Renamed to avoid conflict
        for _ in range(n_samples_sim):
            k = np.random.choice(K_true, p=p)
            # Make local data points tighter around their chosen global centroid to emphasize center differences
            sample = global_centroids[k] + np.random.normal(0, 4.5, n_features_sim) # Reduced local noise
            X_data_center.append(sample)
        X_data_center_np = np.array(X_data_center) # Renamed

        df = pd.DataFrame(X_data_center_np, columns=selected_genes)
        df["Center"] = f"Center_{i+1}"
        df["SampleID"] = [f"S{i+1}_{j+1}" for j in range(n_samples_sim)]
        data_centers_list.append(df)

    return data_centers_list, selected_genes

# In your Streamlit UI for data generation:
# Add a slider for heterogeneity_factor if you want to experiment with it
# heterogeneity_slider = st.sidebar.slider("Center Data Heterogeneity", 0.6, 0.99, 0.95, step=0.01, key="hetero_factor")
# Then pass it to simulate_federated_data_cached

# When calling:
# st.session_state.data_centers, st.session_state.gene_names = simulate_federated_data_cached(
# n_centers_input, n_samples_input, n_features_input, heterogeneity_factor=0.95 # or heterogeneity_slider
# )

# -----------------------
# Streamlit UI: Setup
# -----------------------
st.set_page_config(page_title="Federated Conformal Clustering vs Centralized Clustering", layout="wide")
st.title("🧬 Simulated Distributed Conformal Clustering for Biomarker Discovery")

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Simulation & Model Parameters")

n_centers_input = st.sidebar.slider("Number of Simulated Data Centers (N)", 2, 5, 3, key="n_centers_input")
n_samples_input = st.sidebar.slider("Samples per Center (M)", 10, 200, 50, key="n_samples_input")
n_features_input = st.sidebar.slider("Number of Genes (G)", 5, 50, 10, key="n_features_input")

# Initialize session state
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

st.sidebar.header("🔬 Clustering Parameters")
n_clusters_input = st.sidebar.slider("Number of Clusters (K)", 2, 5, 3, key="n_clusters_k")
num_simulation_rounds_input = st.sidebar.slider("Simulated Distributed Iterative Rounds", 1, 10, 3, key="manual_rounds_sim")
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

# Enhanced LLM Annotation Function
def annotate_cluster_llm(gene_list_to_annotate, annotation_type="summary"):
    api_key_llm = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key_llm:
        return "OpenRouter API key not configured."
    
    headers = {"Authorization": f"Bearer {api_key_llm}"}
    
    # Configure prompts and parameters based on annotation type
    if annotation_type == "detailed":
        prompt_content = f"""Provide a detailed biological analysis of this gene cluster: {', '.join(gene_list_to_annotate)}.

Please include:
1. Primary biological pathways and processes
2. Functional relationships between the genes
3. Clinical or therapeutic relevance
4. Potential biomarker significance

Keep the response comprehensive but under 500 words."""
        max_tokens = 700
        temperature = 0.7
    else:  # summary
        prompt_content = f"""Provide a concise biological theme or pathway summary for these genes: {', '.join(gene_list_to_annotate)}.

Focus on:
- Main shared biological function
- Key pathway(s) involved
- Brief clinical relevance

Keep response under 150 words."""
        max_tokens = 250
        temperature = 0.8
    
    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [{"role": "user", "content": prompt_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": None
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", 
            headers=headers, 
            json=payload, 
            timeout=45  # Longer timeout for detailed responses
        )
        response.raise_for_status()
        result = response.json()
        
        # Better error handling for API response
        if "choices" not in result or len(result["choices"]) == 0:
            return f"No response generated. API returned: {result}"
        
        annotation = result["choices"][0]["message"]["content"].strip()
        
        # Check if response was truncated
        if result["choices"][0].get("finish_reason") == "length":
            annotation += f" [Note: {annotation_type.title()} response was truncated due to length limits]"
            
        return annotation
        
    except requests.exceptions.Timeout:
        return f"Annotation request timed out. Please try again with {'detailed' if annotation_type == 'summary' else 'summary'} mode."
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to OpenRouter: {e}")
        return "Annotation unavailable due to connection error."
    except KeyError as e:
        st.error(f"Unexpected response structure from OpenRouter. Missing key: {e}")
        return f"Annotation unavailable due to API response format."
    except Exception as e:
        st.error(f"An unexpected error occurred during LLM annotation: {e}")
        return "Annotation unavailable."

# Updated cluster annotation section for the main code
def render_cluster_annotations(combined_data_df_to_use, gene_names_to_use, n_clusters_input):
    st.subheader("Cluster Annotations (via OpenRouter LLM)")
    
    if not st.secrets.get("OPENROUTER_API_KEY"):
        st.warning("OpenRouter API key not found in secrets. Annotation feature disabled.")
        return
    
    # Add annotation type selection
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Generate biological annotations for each cluster based on top-expressed genes:")
    with col2:
        annotation_mode = st.selectbox(
            "Annotation Detail Level:",
            ["summary", "detailed"],
            format_func=lambda x: "📝 Summary" if x == "summary" else "📖 Detailed",
            key="annotation_mode_select"
        )
    
    # Batch annotation option
    if st.button("🚀 Annotate All Clusters", key="annotate_all_btn"):
        if 'cluster_annotations' not in st.session_state:
            st.session_state.cluster_annotations = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i_cluster in range(n_clusters_input):
            progress_bar.progress((i_cluster) / n_clusters_input)
            status_text.text(f"Annotating Cluster {i_cluster}...")
            
            cluster_subset_df = combined_data_df_to_use[combined_data_df_to_use["Cluster"] == i_cluster]
            if gene_names_to_use and not cluster_subset_df.empty:
                cluster_gene_means = cluster_subset_df[gene_names_to_use].mean()
                top_genes_list = cluster_gene_means.sort_values(ascending=False).head(5).index.tolist()
                
                if top_genes_list:
                    annotation = annotate_cluster_llm(top_genes_list, annotation_mode)
                    st.session_state.cluster_annotations[f"{i_cluster}_{annotation_mode}"] = annotation
        
        progress_bar.progress(1.0)
        status_text.text("All clusters annotated!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
    
    # Individual cluster annotation buttons and display
    for i_cluster in range(n_clusters_input):
        cluster_subset_df = combined_data_df_to_use[combined_data_df_to_use["Cluster"] == i_cluster]
        
        if gene_names_to_use and not cluster_subset_df.empty:
            cluster_gene_means = cluster_subset_df[gene_names_to_use].mean()
            top_genes_list = cluster_gene_means.sort_values(ascending=False).head(5).index.tolist()
            
            # Create expandable section for each cluster
            with st.expander(f"🧬 Cluster {i_cluster} (N={len(cluster_subset_df)}) - Top genes: {', '.join(top_genes_list)}", expanded=False):
                
                col_btn, col_switch = st.columns([2, 1])
                
                with col_btn:
                    if st.button(f"Generate {annotation_mode.title()} Annotation", key=f"annotate_btn_{i_cluster}_{annotation_mode}"):
                        if top_genes_list:
                            with st.spinner(f"Generating {annotation_mode} annotation for Cluster {i_cluster}..."):
                                annotation = annotate_cluster_llm(top_genes_list, annotation_mode)
                                st.session_state.cluster_annotations = st.session_state.get('cluster_annotations', {})
                                st.session_state.cluster_annotations[f"{i_cluster}_{annotation_mode}"] = annotation
                        else:
                            st.write(f"Cluster {i_cluster} has no gene data for annotation.")
                
                # Display existing annotations
                summary_key = f"{i_cluster}_summary"
                detailed_key = f"{i_cluster}_detailed"
                
                if summary_key in st.session_state.get('cluster_annotations', {}):
                    st.markdown("**📝 Summary Annotation:**")
                    st.markdown(st.session_state.cluster_annotations[summary_key])
                
                if detailed_key in st.session_state.get('cluster_annotations', {}):
                    st.markdown("**📖 Detailed Annotation:**")
                    st.markdown(st.session_state.cluster_annotations[detailed_key])
                
                # Show top genes with their mean expression values
                if top_genes_list:
                    st.markdown("**Top 5 Genes by Mean Expression:**")
                    top_gene_values = cluster_gene_means.sort_values(ascending=False).head(5)
                    for gene, value in top_gene_values.items():
                        st.write(f"• {gene}: {value:.3f}")
        
        elif cluster_subset_df.empty:
            st.write(f"Cluster {i_cluster} is empty, skipping annotation.")
        else:
            st.warning("Gene names not available for cluster annotation.")

# Manual (Simulated) Federated K-Means Function
def run_manual_simulated_distributed_kmeans(data_centers_list, gene_names_list, n_clusters_fed, num_rounds_fed):
    st.write("--- Starting Manual Simulated Distributed K-Means ---")
    all_local_data_np = [df[gene_names_list].values for df in data_centers_list]
    
    # Robust initialization: Use K-Means++ on a combined sample for initial centroids
    # This is better than just using the first center or random points.
    combined_sample_for_init = np.vstack([data[:min(50, len(data))] for data in all_local_data_np if len(data) > 0]) # Sample up to 50 from each
    if len(combined_sample_for_init) < n_clusters_fed:
        st.error("Not enough data across all centers for robust centroid initialization. Consider increasing samples per center.")
        # Fallback to simpler initialization if combined sample is too small
        if len(all_local_data_np[0]) >= n_clusters_fed:
             initial_kmeans = KMeans(n_clusters=n_clusters_fed, random_state=42, n_init='auto').fit(all_local_data_np[0])
             global_centroids_fed = initial_kmeans.cluster_centers_
        else: # Absolute fallback: random points from first center
            st.warning("Using random points from first center for initialization due to small sample size.")
            indices = np.random.choice(all_local_data_np[0].shape[0], n_clusters_fed, replace=False)
            global_centroids_fed = all_local_data_np[0][indices]
    else:
        initial_kmeans = KMeans(n_clusters=n_clusters_fed, init='k-means++', random_state=42, n_init='auto').fit(combined_sample_for_init)
        global_centroids_fed = initial_kmeans.cluster_centers_
    
    st.write(f"Initial global centroids shape: {global_centroids_fed.shape}")

    for r_idx in range(num_rounds_fed):
        st.write(f"Simulated Distributed Round {r_idx+1}/{num_rounds_fed}")
        
        new_global_centroids_sum = np.zeros_like(global_centroids_fed)
        total_points_in_each_global_cluster = np.zeros(n_clusters_fed, dtype=int)

        for i, local_data in enumerate(all_local_data_np):
            if local_data.shape[0] == 0:
                continue # Skip empty centers
            
            # Assign local data points to the current global centroids
            local_assignments = np.argmin(
                metrics.pairwise_distances(local_data, global_centroids_fed, metric='euclidean'),
                axis=1
            )

            # Calculate new local centroids based on these assignments
            current_center_local_centroids = np.zeros_like(global_centroids_fed)
            counts_for_local_centroids = np.zeros(n_clusters_fed, dtype=int)

            for k_cluster in range(n_clusters_fed):
                points_in_k = local_data[local_assignments == k_cluster]
                if len(points_in_k) > 0:
                    current_center_local_centroids[k_cluster] = points_in_k.mean(axis=0)
                    counts_for_local_centroids[k_cluster] = len(points_in_k)
                else:
                    # If a cluster is empty for this local center, it contributes nothing to its update
                    # Or, one might use the global centroid as its local estimate (less common for simple FedAvg)
                    pass
            
            # Accumulate for weighted averaging of centroids
            new_global_centroids_sum += current_center_local_centroids * counts_for_local_centroids[:, np.newaxis]
            total_points_in_each_global_cluster += counts_for_local_centroids
        
        # Aggregate: Calculate new global centroids (weighted average)
        for k_cluster in range(n_clusters_fed):
            if total_points_in_each_global_cluster[k_cluster] > 0:
                global_centroids_fed[k_cluster] = new_global_centroids_sum[k_cluster] / total_points_in_each_global_cluster[k_cluster]
            else:
                # Handle globally empty cluster: Re-initialize from furthest point in all data (more robust)
                # For simplicity, if a global cluster becomes empty, we might just leave its centroid as is from previous round,
                # or pick a random point from all_local_data_np. This needs careful handling in production.
                st.warning(f"Global cluster {k_cluster} became empty. Re-initializing strategies needed for robustness (not fully implemented here).")
                # As a simple fallback here, we don't update it, or re-initialize from a random data point.
                # For demo, if this happens with few points, it's okay.
                if len(all_local_data_np) > 0 and len(all_local_data_np[0]) > 0 :
                    random_center_idx = np.random.randint(len(all_local_data_np))
                    if len(all_local_data_np[random_center_idx]) > 0:
                         random_point_idx = np.random.randint(all_local_data_np[random_center_idx].shape[0])
                         global_centroids_fed[k_cluster] = all_local_data_np[random_center_idx][random_point_idx]


        st.write(f"Round {r_idx+1} global centroids updated.")
    
    st.write("--- Manual Simulated Distributed K-Means Finished ---")
    return global_centroids_fed


# --- Main Analysis Button ---
# --- Main Analysis Button ---
if st.button("🚀 Run Simulated Distributed Conformal Clustering", key="run_analysis_btn"):
    if not st.session_state.data_generated or not st.session_state.data_centers:
        st.error("Please generate data first using the sidebar button.")
        st.stop()

    st.session_state.analysis_run_complete = False
    st.session_state.centralized_analysis_complete = False

    if st.session_state.gene_names:
        with st.spinner(f"Running {num_simulation_rounds_input} rounds of simulated distributed K-Means..."):
            global_centroids = run_manual_simulated_distributed_kmeans(
                st.session_state.data_centers,
                st.session_state.gene_names,
                n_clusters_input,
                num_simulation_rounds_input
            )
        st.success("Simulated distributed K-Means finished.")
    else:
        st.error("Gene names are missing, cannot run K-Means.")
        st.stop()
    
    if global_centroids is not None:
        combined_data_df = pd.concat(st.session_state.data_centers, ignore_index=True)
        X_full = combined_data_df[st.session_state.gene_names].values
        
        # Federated Conformal Clustering
        final_labels_fed, final_distances_to_centroids = pairwise_distances_argmin_min(X_full, global_centroids)
        combined_data_df["Cluster_Federated"] = final_labels_fed
        conformal_scores = final_distances_to_centroids
        combined_data_df["ConformalScore_Federated"] = conformal_scores
        conformal_threshold = np.quantile(conformal_scores, conf_level_input)
        combined_data_df["HighConfidence_Federated"] = combined_data_df["ConformalScore_Federated"] <= conformal_threshold

        # Fully Centralized Clustering
        kmeans_centralized = KMeans(n_clusters=n_clusters_input, random_state=42, n_init='auto').fit(X_full)
        centralized_labels = kmeans_centralized.labels_
        combined_data_df["Cluster_Centralized"] = centralized_labels

        # Store results in session state
        st.session_state.combined_data_df_clustered = combined_data_df
        st.session_state.X_full = X_full
        st.session_state.analysis_run_complete = True
        st.session_state.centralized_analysis_complete = True
    else:
        st.error("Simulated distributed centroids could not be determined. Cannot proceed.")
        st.session_state.analysis_run_complete = False
        st.session_state.centralized_analysis_complete = False

# --- Display Results Side by Side ---
if st.session_state.get('analysis_run_complete', False) and st.session_state.get('centralized_analysis_complete', False):
    combined_data_df = st.session_state.combined_data_df_clustered
    X_full = st.session_state.X_full
    gene_names_to_use = st.session_state.gene_names

    st.subheader("Clustering Results Comparison")
    col1, col2 = st.columns(2)

    # Federated Conformal Clustering Results
    with col1:
        st.subheader("Federated Conformal Clustering")
        
        # Cluster Sizes
        cluster_counts_fed = combined_data_df["Cluster_Federated"].value_counts().sort_index()
        st.write("**Cluster Sizes:**")
        st.write(cluster_counts_fed)
        
        # Silhouette Score
        if len(np.unique(combined_data_df["Cluster_Federated"])) > 1:
            sil_score_fed = metrics.silhouette_score(X_full, combined_data_df["Cluster_Federated"])
            st.write(f"**Silhouette Score:** {sil_score_fed:.3f}")
        else:
            st.write("**Silhouette Score:** N/A (single cluster)")
        
        # Conformal Metrics
        conformal_threshold = np.quantile(combined_data_df["ConformalScore_Federated"], conf_level_input)
        st.write(f"**Conformal Threshold at {int(conf_level_input*100)}% confidence:** {conformal_threshold:.3f}")
        st.write(f"**High-Confidence Samples:** {combined_data_df['HighConfidence_Federated'].sum()}")
        st.write(f"**Ambiguous/Low-Confidence Samples:** {(~combined_data_df['HighConfidence_Federated']).sum()}")
        
        # PCA Plot
        try:
            pca_model = PCA(n_components=2)
            projected_data = pca_model.fit_transform(X_full)
            plot_df_pca_fed = pd.DataFrame({
                "PC1": projected_data[:, 0],
                "PC2": projected_data[:, 1],
                "Cluster": combined_data_df["Cluster_Federated"].astype(str),
                "HighConfidence": combined_data_df["HighConfidence_Federated"]
            })
            chart_fed = (
                alt.Chart(plot_df_pca_fed)
                .mark_circle(size=60)
                .encode(
                    x=alt.X("PC1", title="Principal Component 1"),
                    y=alt.Y("PC2", title="Principal Component 2"),
                    color=alt.Color("Cluster:N", title="Cluster"),
                    opacity=alt.condition(alt.datum.HighConfidence, alt.value(0.9), alt.value(0.3)),
                    tooltip=["PC1", "PC2", "Cluster", "HighConfidence"]
                )
                .properties(title="PCA of Federated Clustered Data", width=350, height=300)
                .interactive()
            )
            st.altair_chart(chart_fed, use_container_width=True)
        except Exception as e:
            st.error(f"Error during PCA and plotting for federated: {e}")

    # Fully Centralized Clustering Results
    with col2:
        st.subheader("Fully Centralized Clustering")
        
        # Cluster Sizes
        cluster_counts_cen = combined_data_df["Cluster_Centralized"].value_counts().sort_index()
        st.write("**Cluster Sizes:**")
        st.write(cluster_counts_cen)
        
        # Silhouette Score
        if len(np.unique(combined_data_df["Cluster_Centralized"])) > 1:
            sil_score_cen = metrics.silhouette_score(X_full, combined_data_df["Cluster_Centralized"])
            st.write(f"**Silhouette Score:** {sil_score_cen:.3f}")
        else:
            st.write("**Silhouette Score:** N/A (single cluster)")
        
        # PCA Plot
        try:
            pca_model = PCA(n_components=2)
            projected_data = pca_model.fit_transform(X_full)
            plot_df_pca_cen = pd.DataFrame({
                "PC1": projected_data[:, 0],
                "PC2": projected_data[:, 1],
                "Cluster": combined_data_df["Cluster_Centralized"].astype(str),
            })
            chart_cen = (
                alt.Chart(plot_df_pca_cen)
                .mark_circle(size=60)
                .encode(
                    x=alt.X("PC1", title="Principal Component 1"),
                    y=alt.Y("PC2", title="Principal Component 2"),
                    color=alt.Color("Cluster:N", title="Cluster"),
                    tooltip=["PC1", "PC2", "Cluster"]
                )
                .properties(title="PCA of Centralized Clustered Data", width=350, height=300)
                .interactive()
            )
            st.altair_chart(chart_cen, use_container_width=True)
        except Exception as e:
            st.error(f"Error during PCA and plotting for centralized: {e}")

    # --- Additional Visualizations and Annotations for Federated Clustering ---
    st.subheader(f"Clustering Quality: {metric_choice_input} (Federated)")
    metric_value_display = "N/A"
    try:
        if metric_choice_input == "Inertia":
            metric_value_display = np.sum(combined_data_df["ConformalScore_Federated"]**2)
        elif len(np.unique(combined_data_df["Cluster_Federated"])) > 1:
            if metric_choice_input == "Silhouette Score":
                metric_value_display = metrics.silhouette_score(X_full, combined_data_df["Cluster_Federated"])
            elif metric_choice_input == "Calinski-Harabasz Index":
                metric_value_display = metrics.calinski_harabasz_score(X_full, combined_data_df["Cluster_Federated"])
            elif metric_choice_input == "Davies-Bouldin Index":
                metric_value_display = metrics.davies_bouldin_score(X_full, combined_data_df["Cluster_Federated"])
        else:
            metric_value_display = "N/A (requires >1 cluster for this metric)"
        st.write(f"**{metric_choice_input}:** {metric_value_display:.2f}" if isinstance(metric_value_display, (int, float)) else metric_value_display)
    except Exception as e:
        st.error(f"Error calculating metric '{metric_choice_input}': {e}")

    # Default PCA Plot (already shown above, skipped here to avoid redundancy)

    # Additional Visualizations (Federated Only)
    if additional_viz_choice == "Mean Cluster Curves":
        st.subheader("Mean Curves per Cluster (Federated)")
        unique_clusters = sorted(combined_data_df['Cluster_Federated'].unique())
        n_cols_mc = 2
        n_rows_mc = (len(unique_clusters) + n_cols_mc - 1) // n_cols_mc
        fig_mean_curves, axs_mc = plt.subplots(n_rows_mc, n_cols_mc, figsize=(n_cols_mc * 6, n_rows_mc * 4), squeeze=False)
        axs_mc_flat = axs_mc.flatten()
        for idx, cluster_id in enumerate(unique_clusters):
            if idx < len(axs_mc_flat):
                ax = axs_mc_flat[idx]
                cluster_data_mc = combined_data_df[combined_data_df['Cluster_Federated'] == cluster_id][gene_names_to_use]
                if not cluster_data_mc.empty:
                    mean_curve = cluster_data_mc.mean(axis=0)
                    ax.plot(gene_names_to_use, mean_curve, color=f'C{int(cluster_id) % 10}', linewidth=2, label=f"Mean Cluster {cluster_id}")
                    if st.checkbox(f"Show individual curves for Cluster {cluster_id}", key=f"show_ind_c{cluster_id}"):
                        for _, row_val in cluster_data_mc.iterrows():
                            ax.plot(gene_names_to_use, row_val, color=f'C{int(cluster_id) % 10}', alpha=0.2)
                    ax.set_title(f"Cluster {cluster_id} (N={len(cluster_data_mc)})")
                    ax.set_xlabel("Genes / Features")
                    ax.set_ylabel("Expression Value")
                    ax.legend()
        for i_ax_mc in range(len(unique_clusters), len(axs_mc_flat)): fig_mean_curves.delaxes(axs_mc_flat[i_ax_mc])
        plt.tight_layout()
        st.pyplot(fig_mean_curves)

    elif additional_viz_choice == "Parallel Coordinates":
        st.subheader("Parallel Coordinates Plot of Clusters (Federated)")
        if not gene_names_to_use:
            st.warning("No gene names available for parallel coordinates plot.")
        else:
            num_genes_for_parallel = min(10, len(gene_names_to_use))
            genes_for_plot_parallel = gene_names_to_use[:num_genes_for_parallel]
            if not genes_for_plot_parallel:
                st.warning("No genes selected or available for parallel coordinates plot.")
            elif not combined_data_df.empty and 'Cluster_Federated' in combined_data_df:
                try:
                    df_for_parallel = combined_data_df.copy()
                    unique_cluster_labels_str = sorted(df_for_parallel['Cluster_Federated'].astype(str).unique())
                    label_to_numeric_map = {label_str_val: i for i, label_str_val in enumerate(unique_cluster_labels_str)}
                    df_for_parallel['ClusterNumeric'] = df_for_parallel['Cluster_Federated'].astype(str).map(label_to_numeric_map)
                    
                    temp_genes_for_plot_parallel = list(genes_for_plot_parallel)
                    for gene_iter in temp_genes_for_plot_parallel[:]:
                        if gene_iter in df_for_parallel.columns:
                            df_for_parallel[gene_iter] = pd.to_numeric(df_for_parallel[gene_iter], errors='coerce')
                        else:
                            st.warning(f"Gene column '{gene_iter}' not found in DataFrame for parallel plot. Removing from dimensions.")
                            if gene_iter in genes_for_plot_parallel: genes_for_plot_parallel.remove(gene_iter)
                    
                    final_genes_for_plot = [g for g in genes_for_plot_parallel if g in df_for_parallel.columns]
                    if final_genes_for_plot:
                         df_for_parallel = df_for_parallel.dropna(subset=final_genes_for_plot)
                    
                    if df_for_parallel.empty or len(df_for_parallel) < 2:
                        st.warning("Insufficient data after cleaning for parallel coordinates.")
                    else:
                        numeric_genes = [gene for gene in final_genes_for_plot if gene in df_for_parallel.columns and pd.api.types.is_numeric_dtype(df_for_parallel[gene])]
                        if len(numeric_genes) < 2:
                            st.error("Need at least 2 numeric dimensions for parallel coordinates plot.")
                        else:
                            fig_parallel = px.parallel_coordinates(
                                df_for_parallel, color="ClusterNumeric", dimensions=numeric_genes,
                                title="Parallel Coordinates Plot by Cluster (Federated)"
                            )
                            st.plotly_chart(fig_parallel, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating parallel coordinates plot: {str(e)}")
            else:
                st.warning("No data available for parallel coordinates plot.")

    elif additional_viz_choice == "Heatmap of Mean Profiles":
        st.subheader("Heatmap of Mean Cluster Profiles (Federated)")
        if not combined_data_df.empty and 'Cluster_Federated' in combined_data_df and gene_names_to_use:
            mean_profiles = combined_data_df.groupby('Cluster_Federated')[gene_names_to_use].mean()
            if not mean_profiles.empty:
                fig_heatmap_mean, ax_heatmap_mean = plt.subplots(figsize=(10, max(4, len(mean_profiles) * 0.5)))
                sns.heatmap(mean_profiles, annot=True, fmt=".2f", cmap="viridis", ax=ax_heatmap_mean, cbar=True)
                ax_heatmap_mean.set_xlabel("Genes / Features")
                ax_heatmap_mean.set_ylabel("Cluster ID")
                ax_heatmap_mean.set_title("Mean Gene Expression Profiles per Cluster (Federated)")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig_heatmap_mean)
            else: st.info("No mean profiles to display.")
        else: st.info("Clustered data or gene names not available.")

    elif additional_viz_choice == "3D PCA Scatter":
        st.subheader("3D PCA Projection of Clustered Data (Federated)")
        if X_full.shape[1] >= 3:
            try:
                pca_3d = PCA(n_components=3)
                projected_3d = pca_3d.fit_transform(X_full)
                df_3d_pca = pd.DataFrame({
                    "PC1": projected_3d[:, 0], 
                    "PC2": projected_3d[:, 1], 
                    "PC3": projected_3d[:, 2],
                    "Cluster": combined_data_df["Cluster_Federated"].astype(str),
                    "HighConfidence": combined_data_df["HighConfidence_Federated"]
                })
                
                df_high_conf = df_3d_pca[df_3d_pca['HighConfidence'] == True]
                df_low_conf = df_3d_pca[df_3d_pca['HighConfidence'] == False]
                
                fig_3d = px.scatter_3d(
                    df_high_conf, x='PC1', y='PC2', z='PC3', color='Cluster',
                    title="3D PCA of Federated Clustered Data", 
                    labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'},
                    hover_data=["Cluster", "HighConfidence"]
                )
                fig_3d.update_traces(marker=dict(size=5, opacity=0.9), name="High Confidence")
                
                if not df_low_conf.empty:
                    fig_low_conf = px.scatter_3d(
                        df_low_conf, x='PC1', y='PC2', z='PC3', color='Cluster',
                        hover_data=["Cluster", "HighConfidence"]
                    )
                    fig_low_conf.update_traces(marker=dict(size=5, opacity=0.3), name="Low Confidence")
                    for trace in fig_low_conf.data:
                        fig_3d.add_trace(trace)
                
                fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig_3d, use_container_width=True)
            except Exception as e: 
                st.error(f"Error during 3D PCA: {e}")
        else: 
            st.warning("Need at least 3 features/genes for 3D PCA.")
            
    elif additional_viz_choice == "Gene BoxPlots":
        st.subheader("Gene Expression Distribution by Cluster (Federated Box Plots)")
        if gene_names_to_use:
            num_genes_to_boxplot = min(5, len(gene_names_to_use))
            selected_genes_for_boxplot = st.multiselect(
                "Select genes for box plot:", options=gene_names_to_use,
                default=gene_names_to_use[:num_genes_to_boxplot], key="boxplot_genes_select"
            )
            if selected_genes_for_boxplot and not combined_data_df.empty and 'Cluster_Federated' in combined_data_df:
                n_cols_bp = 2
                n_rows_bp = (len(selected_genes_for_boxplot) + n_cols_bp - 1) // n_cols_bp
                fig_boxplots, axs_bp = plt.subplots(n_rows_bp, n_cols_bp, figsize=(n_cols_bp * 7, n_rows_bp * 5), squeeze=False)
                axs_bp_flat = axs_bp.flatten()
                for idx, gene_name_bp in enumerate(selected_genes_for_boxplot):
                    if idx < len(axs_bp_flat):
                        ax = axs_bp_flat[idx]
                        sns.boxplot(x='Cluster_Federated', y=gene_name_bp, data=combined_data_df, ax=ax, palette="Set2")
                        sns.stripplot(x='Cluster_Federated', y=gene_name_bp, data=combined_data_df, ax=ax, color=".25", alpha=0.5, dodge=True)
                        ax.set_title(f"Expression of {gene_name_bp}")
                for i_ax_bp in range(len(selected_genes_for_boxplot), len(axs_bp_flat)): fig_boxplots.delaxes(axs_bp_flat[i_ax_bp])
                plt.tight_layout()
                st.pyplot(fig_boxplots)
        else:
            st.warning("Gene names not available for box plots.")

    elif additional_viz_choice == "PCA Scatter (Streamlit Native)":
        st.subheader("PCA Projection of Federated Clustered Data (Streamlit Native)")
        try:
            pca_model_st = PCA(n_components=2)
            projected_data_st = pca_model_st.fit_transform(X_full)
            df_pca_st = pd.DataFrame({
                "PC1": projected_data_st[:, 0],
                "PC2": projected_data_st[:, 1],
                "Cluster": combined_data_df["Cluster_Federated"].astype(str)
            })
            st.scatter_chart(df_pca_st, x="PC1", y="PC2", color="Cluster")
        except Exception as e:
            st.error(f"Error generating Streamlit native PCA scatter plot: {e}")

    # Render cluster annotations for federated clustering
    render_cluster_annotations(combined_data_df.rename(columns={"Cluster_Federated": "Cluster"}), gene_names_to_use, n_clusters_input)
