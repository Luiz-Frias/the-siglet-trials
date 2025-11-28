# Siglet‑Qubit Kernel POC

**Date:** 2025‑04‑21
**Author:** Luiz Frias

---

## 📖 Summary / Mission

We’re building a **classical prototype** of the Siglet‑Qubit kernel:
1. **Define** `SigletQubit(theta, τ, ε, μ, c)`
2. **Sweep** resonance (θ) and coherence (τ) across time to map “truth‑score” decay
3. **Cluster** stable regions to identify a finite siglet alphabet
4. **Visualize** results with decay curves and heatmaps

This notebook is the **ground floor**—today’s deliverable is a self‑contained classical sim. Tomorrow we’ll port core gates into a 1‑qubit quantum toy.

---

## 🎯 Objectives

- ✅ Implement `SigletQubit` class
- ✅ Generate decay curves for sample θ values
- ✅ Produce a heatmap of T(s,t) over (θ,τ) grid
- ✅ Cluster stable siglet regions
- ✅ Export CSV for memo & outreach

---

## 🛠 Tech Stack & Constraints

- **Language:** Python 3.9+
- **Libraries:** `numpy`, `matplotlib`, (`scikit‑learn` for clustering)
- **Environment:** Local Jupyter on M2 MacBook Air (no cloud)
- **Style:** Clean, modular cells; extensive comments; version control via git

---

# PRD: Siglet‑Qubit Kernel Prototype

## Background
RealityOS needs a **foundational symbolic unit** (“siglet”) that carries resonance, temporality, ethics, modality, and compression. We model it classically first, then port to photonic qubits.

## Functional Requirements
1. **Data structure**: `SigletQubit(theta, tau, eps_norm, mu, c)`
2. **Truth‑score function**: `T(s, t) = cos(theta) * tau^t * eps_norm * mu * c`
3. **Parameter sweep** over
   - `theta ∈ [0,π]` (50 steps)
   - `tau ∈ [0.8,1.0]` (20 steps)
4. **Visualization**
   - Decay curves for selected θ
   - 2D heatmap of T(s,t) across (θ,τ)

## Non‑Functional Requirements
- Runs in < 30 s on local M2
- Jupyter cells self‑documented
- Results exportable (CSV + PNG)

## Future Extensions
- Quantum toy via Qiskit / PennyLane


```python
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mlflow
import os
from sklearn.preprocessing import StandardScaler
import time
import random
from sklearn.metrics import (
    confusion_matrix,
    silhouette_score,
    cdist_dtw,
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import itertools
import optuna
from fastdtw import fastdtw

# Set up MLflow experiment
mlflow.set_experiment("SigletQubit-Classical-Sim")
```

    /Users/luizfrias/miniconda3/envs/siglet_architecture/lib/python3.10/site-packages/tslearn/bases/bases.py:15: UserWarning: h5py not installed, hdf5 features will not be supported.
    Install h5py to use hdf5 features: http://docs.h5py.org/
      warn(h5py_msg)





    <Experiment: artifact_location='file:///Users/luizfrias/CursorAI/data-science/siglet_architecture/notebooks/mlruns/833143775662486063', creation_time=1745263369880, experiment_id='833143775662486063', last_update_time=1745263369880, lifecycle_stage='active', name='SigletQubit-Classical-Sim', tags={}>




```python
# set working directory to the project root
ROOT_DIR = "/Users/luizfrias/CursorAI/data-science/siglet_architecture"

# Define project paths
DATA_DIR = os.path.join(ROOT_DIR, "data/processed")
FIGURES_DIR = os.path.join(ROOT_DIR, "reports/figures")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
NOTES_DIR = os.path.join(ROOT_DIR, "reports/notes")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)
```


```python
# Define the dataclass for the qubit
class SigletQubit:
    def __init__(self, theta, tau, eps_norm, mu, c):
        self.theta = theta  # resonance angle
        self.tau = tau  # coherence factor
        self.eps_norm = eps_norm  # ||ε||
        self.mu = mu  # modality fidelity
        self.c = c  # compression ratio

    def resonance(self):
        return np.cos(self.theta)

    def truth_score(self, t):
        r = self.resonance()
        return r * (self.tau**t) * self.eps_norm * self.mu * self.c
```


```python
# Parameter grid
thetas = np.linspace(0, np.pi, 50)
taus = np.linspace(0.8, 1.0, 20)
t_max = 10
```


```python
# Start MLflow run
with mlflow.start_run(run_name="decay_curves") as run:
    # Log parameters
    mlflow.log_param("theta_range", [0, np.pi])
    mlflow.log_param("tau_range", [0.8, 1.0])
    mlflow.log_param("t_max", t_max)
    mlflow.log_param("eps_norm", 1.0)
    mlflow.log_param("mu", 0.8)
    mlflow.log_param("c", 0.7)

    # Sweep & plot a few curves
    plt.figure(figsize=(8, 5))

    sample_thetas = [0, np.pi / 4, np.pi / 2]
    for theta in sample_thetas:
        sq = SigletQubit(theta=theta, tau=0.9, eps_norm=1.0, mu=0.8, c=0.7)
        scores = [sq.truth_score(t) for t in range(t_max + 1)]
        plt.plot(range(t_max + 1), scores, label=f"θ={theta:.2f}")

        # Log metrics for each theta
        mlflow.log_metric(f"initial_score_theta_{theta:.2f}", scores[0])
        mlflow.log_metric(f"final_score_theta_{theta:.2f}", scores[-1])
        mlflow.log_metric(
            f"decay_rate_theta_{theta:.2f}", (scores[0] - scores[-1]) / t_max
        )

    plt.xlabel("Time (t)")
    plt.ylabel("Truth Score T(s, t)")
    plt.title("Siglet‑Qubit Decay Curves")
    plt.legend()

    # Save figure locally
    decay_fig_path = os.path.join(FIGURES_DIR, "decay_curves.png")
    plt.savefig(decay_fig_path, dpi=300)

    # Log artifact
    mlflow.log_artifact(decay_fig_path)

    plt.show()
```


```python
# Generate the heatmap of T(s,t) over (θ,τ) grid
with mlflow.start_run(run_name="heatmap_generation") as run:
    start_time = time.time()

    # Log parameters
    mlflow.log_param("theta_steps", len(thetas))
    mlflow.log_param("tau_steps", len(taus))
    mlflow.log_param("t_eval", 5)  # Time point to evaluate

    t_eval = 5  # Time point to evaluate (can be adjusted)
    truth_scores = np.zeros((len(thetas), len(taus)))

    for i, theta in enumerate(thetas):
        for j, tau in enumerate(taus):
            sq = SigletQubit(theta=theta, tau=tau, eps_norm=1.0, mu=0.8, c=0.7)
            truth_scores[i, j] = sq.truth_score(t_eval)

    # Log performance metric
    computation_time = time.time() - start_time
    mlflow.log_metric("computation_time_seconds", computation_time)
    mlflow.log_metric("mean_truth_score", np.mean(truth_scores))
    mlflow.log_metric("max_truth_score", np.max(truth_scores))
    mlflow.log_metric("min_truth_score", np.min(truth_scores))

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    # Store the heatmap object in a variable
    hm = sns.heatmap(
        truth_scores,
        xticklabels=np.round(taus, 2),
        yticklabels=np.round(thetas, 2),
        cmap="viridis",
        cbar_kws={"label": "Truth Score"},
    )  # Add colorbar label directly here

    plt.xlabel("Coherence (τ)")
    plt.ylabel("Resonance (θ)")
    plt.title(f"Truth Score T(s,t) at t={t_eval}")
    # Remove this line as the colorbar is already created by sns.heatmap
    # plt.colorbar(label='Truth Score')
    plt.tight_layout()

    # Save figure locally
    heatmap_fig_path = os.path.join(FIGURES_DIR, "siglet_heatmap.png")
    plt.savefig(heatmap_fig_path, dpi=300)

    # Log artifact
    mlflow.log_artifact(heatmap_fig_path)

    # Save data for later use
    truth_scores_path = os.path.join(DATA_DIR, "truth_scores.npy")
    np.save(truth_scores_path, truth_scores)
    mlflow.log_artifact(truth_scores_path)

    plt.show()
```


```python
# Cluster stable siglet regions
with mlflow.start_run(run_name="cluster_analysis") as run:
    # Prepare data for clustering - we'll focus on regions with high truth scores
    # Create a DataFrame from our results
    data = []
    for i, theta in enumerate(thetas):
        for j, tau in enumerate(taus):
            data.append([theta, tau, truth_scores[i, j]])

    df = pd.DataFrame(data, columns=["theta", "tau", "truth_score"])

    # Log parameters
    stable_threshold = df["truth_score"].mean()  # Adjust threshold as needed
    mlflow.log_param("stable_threshold", stable_threshold)

    # Filter for stable regions (high truth scores)
    stable_regions = df[df["truth_score"] > stable_threshold]
    mlflow.log_metric("stable_regions_count", len(stable_regions))
    mlflow.log_metric("stable_regions_percentage", 100 * len(stable_regions) / len(df))

    # Normalize the data for clustering
    X = stable_regions[["theta", "tau"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-means clustering
    n_clusters = 5  # Adjust number of clusters as appropriate
    mlflow.log_param("n_clusters", n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    stable_regions["cluster"] = kmeans.fit_predict(X_scaled)

    # Log metrics about the clusters
    for i in range(n_clusters):
        cluster_size = np.sum(stable_regions["cluster"] == i)
        mlflow.log_metric(f"cluster_{i}_size", cluster_size)
        mlflow.log_metric(
            f"cluster_{i}_mean_score",
            stable_regions[stable_regions["cluster"] == i]["truth_score"].mean(),
        )

    # Log inertia (sum of squared distances to centroids)
    mlflow.log_metric("kmeans_inertia", kmeans.inertia_)

    # Visualize the clusters
    plt.figure(figsize=(10, 8))
    for cluster in range(n_clusters):
        cluster_points = stable_regions[stable_regions["cluster"] == cluster]
        plt.scatter(
            cluster_points["tau"], cluster_points["theta"], label=f"Cluster {cluster}"
        )

    plt.xlabel("Coherence (τ)")
    plt.ylabel("Resonance (θ)")
    plt.title("Clusters of Stable Siglet Regions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure locally
    clusters_fig_path = os.path.join(FIGURES_DIR, "siglet_clusters.png")
    plt.savefig(clusters_fig_path, dpi=300)

    # Log artifact
    mlflow.log_artifact(clusters_fig_path)

    plt.show()
```


```python
# Export results to CSV and save plots
with mlflow.start_run(run_name="export_results") as run:
    # Save the DataFrame with all results
    results_path = os.path.join(DATA_DIR, "siglet_qubit_results.csv")
    df.to_csv(results_path, index=False)
    mlflow.log_artifact(results_path)
    print(f"Exported results to {results_path}")

    # Save the clusters data
    clusters_path = os.path.join(DATA_DIR, "siglet_stable_clusters.csv")
    stable_regions.to_csv(clusters_path, index=False)
    mlflow.log_artifact(clusters_path)
    print(f"Exported clusters to {clusters_path}")

    # Log summary metrics
    mlflow.log_metric("total_datapoints", len(df))
    mlflow.log_metric("clustered_datapoints", len(stable_regions))
    mlflow.log_metric("clustering_coverage_pct", 100 * len(stable_regions) / len(df))

    # Create a summary of cluster centers
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_, columns=["theta_scaled", "tau_scaled"]
    )

    # Inverse transform to get original theta and tau values
    centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers["theta"] = centers_orig[:, 0]
    cluster_centers["tau"] = centers_orig[:, 1]

    # Add average truth score for each cluster
    cluster_centers["avg_truth_score"] = [
        stable_regions[stable_regions["cluster"] == i]["truth_score"].mean()
        for i in range(n_clusters)
    ]

    # Save cluster centers
    centers_path = os.path.join(DATA_DIR, "siglet_cluster_centers.csv")
    cluster_centers.to_csv(centers_path, index=False)
    mlflow.log_artifact(centers_path)
    print(f"Exported cluster centers to {centers_path}")

    # Display cluster centers
    print("\nSiglet Alphabet Candidates (Cluster Centers):")
    display(
        cluster_centers[["theta", "tau", "avg_truth_score"]].sort_values(
            "avg_truth_score", ascending=False
        )
    )
```


```python
# Generate complete decay curves for all parameter combinations
with mlflow.start_run(run_name="generate_decay_curves") as run:
    start_time = time.time()

    # Parameters for tracking
    mlflow.log_param("thetas_count", len(thetas))
    mlflow.log_param("taus_count", len(taus))
    mlflow.log_param("t_points", t_max + 1)

    # Create a matrix to store all decay curves
    # Shape: (n_curves, t_points) where n_curves = len(thetas) * len(taus)
    decay_curves = np.zeros((len(thetas) * len(taus), t_max + 1))

    # Parameters for each curve
    curve_params = []

    # Generate all decay curves
    curve_idx = 0
    for i, theta in enumerate(thetas):
        for j, tau in enumerate(taus):
            sq = SigletQubit(theta=theta, tau=tau, eps_norm=1.0, mu=0.8, c=0.7)
            decay_curves[curve_idx] = [sq.truth_score(t) for t in range(t_max + 1)]
            curve_params.append((theta, tau))
            curve_idx += 1

    # Convert parameters to numpy array for easier indexing
    curve_params = np.array(curve_params)

    # Log performance
    computation_time = time.time() - start_time
    mlflow.log_metric("decay_curves_generation_time", computation_time)

    # Save data for later use
    decay_curves_path = os.path.join(DATA_DIR, "decay_curves.npy")
    curve_params_path = os.path.join(DATA_DIR, "curve_params.npy")
    np.save(decay_curves_path, decay_curves)
    np.save(curve_params_path, curve_params)
    mlflow.log_artifact(decay_curves_path)
    mlflow.log_artifact(curve_params_path)

    # Plot a sample of curves to verify
    plt.figure(figsize=(12, 8))
    sample_indices = np.random.choice(len(decay_curves), 10, replace=False)
    for idx in sample_indices:
        theta, tau = curve_params[idx]
        plt.plot(
            range(t_max + 1), decay_curves[idx], label=f"θ={theta:.2f}, τ={tau:.2f}"
        )

    plt.xlabel("Time (t)")
    plt.ylabel("Truth Score T(s, t)")
    plt.title("Sample of Decay Curves")
    plt.legend()
    plt.grid(True)

    # Save and log the sample plot
    sample_fig_path = os.path.join(FIGURES_DIR, "sample_decay_curves.png")
    plt.savefig(sample_fig_path, dpi=300)
    mlflow.log_artifact(sample_fig_path)

    plt.show()
```


```python
# Compute DTW distances and apply clustering
with mlflow.start_run(run_name="dtw_clustering") as run:
    start_time = time.time()

    # Compute DTW distance matrix (this can be time-consuming)
    print("Computing DTW distance matrix...")
    dtw_dist = cdist_dtw(decay_curves)

    # Log computation time
    dtw_time = time.time() - start_time
    mlflow.log_metric("dtw_computation_time", dtw_time)
    print(f"DTW computation completed in {dtw_time:.2f} seconds")

    # Save DTW distance matrix
    np.save(os.path.join(DATA_DIR, "dtw_distances.npy"), dtw_dist)
    mlflow.log_artifact(os.path.join(DATA_DIR, "dtw_distances.npy"))

    # Normalize the DTW distances - add this line
    dtw_dist_normalized = dtw_dist / np.max(dtw_dist)

    # Try a range of eps values to find optimal clustering
    eps_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    best_eps = None
    best_n_clusters = 0

    plt.figure(figsize=(15, 10))

    for i, eps in enumerate(eps_values):
        # Apply DBSCAN with different eps values on normalized distances
        dbscan = DBSCAN(metric="precomputed", eps=eps, min_samples=5)
        labels = dbscan.fit_predict(dtw_dist_normalized)  # Use normalized distances

        # Count number of clusters (excluding noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Log metrics
        mlflow.log_metric(f"dbscan_clusters_eps_{eps}", n_clusters)
        mlflow.log_metric(f"dbscan_noise_eps_{eps}", n_noise)

        # Plot in a subplot
        plt.subplot(1, len(eps_values), i + 1)

        # Plot cluster assignments in parameter space
        unique_labels = set(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise
                col = "black"

            class_member_mask = labels == k
            xy = curve_params[class_member_mask]
            plt.scatter(
                xy[:, 1],
                xy[:, 0],
                s=10,
                c=[col],
                label=f"Cluster {k}" if k != -1 else "Noise",
            )

        plt.title(f"eps={eps}: {n_clusters} clusters, {n_noise} noise")
        plt.xlabel("Coherence (τ)")
        plt.ylabel("Resonance (θ)" if i == 0 else "")
        plt.grid(True)

        # Keep track of best eps value (maximum number of clusters, but not too many noise points)
        noise_percentage = n_noise / len(labels) * 100
        if n_clusters > best_n_clusters and noise_percentage < 50:
            best_n_clusters = n_clusters
            best_eps = eps

    plt.tight_layout()

    # Save the parameter exploration figure
    eps_exploration_path = os.path.join(FIGURES_DIR, "dbscan_eps_exploration.png")
    plt.savefig(eps_exploration_path, dpi=300)
    mlflow.log_artifact(eps_exploration_path)

    plt.show()

    # Use the best eps value for final clustering
    print(f"Best eps value: {best_eps} with {best_n_clusters} clusters")

    # Rerun DBSCAN with the best eps on normalized distances
    dbscan = DBSCAN(metric="precomputed", eps=best_eps, min_samples=5)
    labels_dbscan = dbscan.fit_predict(dtw_dist_normalized)  # Use normalized distances

    # Log final results
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise_dbscan = list(labels_dbscan).count(-1)
    mlflow.log_param("best_eps", best_eps)
    mlflow.log_metric("dbscan_n_clusters", n_clusters_dbscan)
    mlflow.log_metric("dbscan_n_noise_points", n_noise_dbscan)

    # Save final labels
    labels_path = os.path.join(DATA_DIR, "dbscan_labels.npy")
    np.save(labels_path, labels_dbscan)
    mlflow.log_artifact(labels_path)
```


```python
# Let's verify this shape invariance hypothesis
with mlflow.start_run(run_name="shape_analysis") as run:
    # Normalize each curve to [0,1] range to compare pure shapes
    normalized_curves = np.zeros_like(decay_curves)
    for i in range(len(decay_curves)):
        curve = decay_curves[i]
        min_val = np.min(curve)
        max_val = np.max(curve)
        if max_val > min_val:  # Avoid division by zero
            normalized_curves[i] = (curve - min_val) / (max_val - min_val)
        else:
            normalized_curves[i] = curve

    # Compute DTW distances on normalized curves
    print("Computing DTW distances on normalized curves...")
    norm_dtw_dist = cdist_dtw(normalized_curves)

    # Visualize the distribution of distances
    plt.figure(figsize=(10, 8))
    plt.hist(norm_dtw_dist.flatten(), bins=50)
    plt.xlabel("DTW Distance (normalized curves)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Shape Differences Between Curves")
    plt.grid(True)

    # Save figure
    shape_dist_path = os.path.join(FIGURES_DIR, "shape_distance_distribution.png")
    plt.savefig(shape_dist_path, dpi=300)
    mlflow.log_artifact(shape_dist_path)
    plt.show()

    # Plot a sample of normalized curves
    plt.figure(figsize=(12, 8))
    sample_indices = np.random.choice(len(normalized_curves), 20, replace=False)
    for idx in sample_indices:
        theta, tau = curve_params[idx]
        plt.plot(
            range(t_max + 1),
            normalized_curves[idx],
            alpha=0.7,
            label=f"θ={theta:.2f}, τ={tau:.2f}" if idx == sample_indices[0] else "",
        )

    plt.xlabel("Time (t)")
    plt.ylabel("Normalized Truth Score")
    plt.title("Shape Comparison of Normalized Decay Curves")
    if len(sample_indices) > 0:
        plt.legend()
    plt.grid(True)

    # Save figure
    norm_curves_path = os.path.join(FIGURES_DIR, "normalized_curves_comparison.png")
    plt.savefig(norm_curves_path, dpi=300)
    mlflow.log_artifact(norm_curves_path)
    plt.show()

    # Calculate the mean curve and distance from mean
    mean_curve = np.mean(normalized_curves, axis=0)
    distances_from_mean = []

    for curve in normalized_curves:
        dist = np.sqrt(np.sum((curve - mean_curve) ** 2))
        distances_from_mean.append(dist)

    # Plot mean curve and distribution of distances
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(t_max + 1), mean_curve, "r-", linewidth=3)
    plt.xlabel("Time (t)")
    plt.ylabel("Normalized Truth Score")
    plt.title("Mean Curve Shape")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(distances_from_mean, bins=30)
    plt.xlabel("Euclidean Distance from Mean Curve")
    plt.ylabel("Frequency")
    plt.title("Curve Variation Around Mean Shape")
    plt.grid(True)

    plt.tight_layout()

    # Save figure
    mean_curve_path = os.path.join(FIGURES_DIR, "mean_curve_analysis.png")
    plt.savefig(mean_curve_path, dpi=300)
    mlflow.log_artifact(mean_curve_path)
    plt.show()

    # Log the scientific finding
    mlflow.log_metric("mean_shape_distance", np.mean(distances_from_mean))
    mlflow.log_metric("max_shape_distance", np.max(distances_from_mean))

    # Scientific conclusion
    conclusion = """
    # Shape Invariance in Siglet-Qubit System

    ## Finding

    The DTW-based clustering reveals that despite varying parameters (θ,τ), all decay curves
    belong to the same shape family. This suggests a fundamental shape invariant in the
    siglet-qubit temporal dynamics.

    ## Mathematical Explanation

    The truth score function T(s,t) = cos(theta) * tau^t * eps_norm * mu * c creates curves
    that share the same underlying exponential decay pattern, with differences only in:

    1. Initial amplitude (controlled by θ)
    2. Decay rate (controlled by τ)

    ## Implications

    This shape invariance suggests that siglet-qubits with different parameters maintain
    temporal coherence in their dynamic behavior, a potentially important property for
    quantum information processing.

    The siglet alphabet can be defined by (θ,τ) parameter clusters, with the understanding
    that temporal dynamics follow a universal pattern across all symbols.
    """

    with open(os.path.join(DATA_DIR, "shape_invariance_finding.md"), "w") as f:
        f.write(conclusion)

    mlflow.log_artifact(os.path.join(DATA_DIR, "shape_invariance_finding.md"))
```


```python
# Visualize DBSCAN clusters
with mlflow.start_run(run_name="dbscan_visualization") as run:
    # Identify unique clusters (excluding noise points)
    unique_clusters = sorted(set(labels_dbscan))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    # Create a colormap for clusters
    colors = cm.viridis(np.linspace(0, 1, len(unique_clusters)))

    # 1. Plot representative curves from each cluster
    plt.figure(figsize=(15, 10))

    # First plot noise points (if any)
    noise_indices = np.where(labels_dbscan == -1)[0]
    if len(noise_indices) > 0:
        # Sample at most 5 noise points
        sample_noise = np.random.choice(
            noise_indices, min(5, len(noise_indices)), replace=False
        )
        for idx in sample_noise:
            plt.plot(range(t_max + 1), decay_curves[idx], "k--", alpha=0.3)

    # Plot curves from each cluster
    for i, cluster in enumerate(unique_clusters):
        # Get indices for this cluster
        cluster_indices = np.where(labels_dbscan == cluster)[0]

        # Choose representative curves (cluster center + random samples)
        # Find curve closest to cluster center by minimizing sum of DTW distances
        if len(cluster_indices) > 1:
            within_cluster_dist = dtw_dist[np.ix_(cluster_indices, cluster_indices)]
            center_idx = cluster_indices[np.argmin(within_cluster_dist.sum(axis=1))]

            # Plot cluster center
            plt.plot(
                range(t_max + 1),
                decay_curves[center_idx],
                color=colors[i],
                linewidth=3,
                label=f"Cluster {cluster} (center)",
            )

            # Plot a few random members
            sample_size = min(3, len(cluster_indices) - 1)
            if sample_size > 0:
                other_indices = [idx for idx in cluster_indices if idx != center_idx]
                sample_indices = np.random.choice(
                    other_indices, sample_size, replace=False
                )
                for idx in sample_indices:
                    plt.plot(
                        range(t_max + 1), decay_curves[idx], color=colors[i], alpha=0.5
                    )
        else:
            # If only one curve in cluster
            plt.plot(
                range(t_max + 1),
                decay_curves[cluster_indices[0]],
                color=colors[i],
                linewidth=3,
                label=f"Cluster {cluster}",
            )

    plt.xlabel("Time (t)")
    plt.ylabel("Truth Score T(s, t)")
    plt.title("Representative Decay Curves by DBSCAN Cluster")
    plt.legend()
    plt.grid(True)

    # Save figure
    dbscan_curves_fig_path = os.path.join(
        FIGURES_DIR, "dbscan_representative_curves.png"
    )
    plt.savefig(dbscan_curves_fig_path, dpi=300)
    mlflow.log_artifact(dbscan_curves_fig_path)

    plt.show()

    # 2. Plot clusters in parameter space (θ,τ)
    plt.figure(figsize=(12, 10))

    # Plot noise points first
    if len(noise_indices) > 0:
        noise_params = curve_params[noise_indices]
        plt.scatter(
            noise_params[:, 1],
            noise_params[:, 0],
            c="gray",
            marker="x",
            alpha=0.5,
            label="Noise",
        )

    # Plot each cluster
    for i, cluster in enumerate(unique_clusters):
        cluster_indices = np.where(labels_dbscan == cluster)[0]
        cluster_params = curve_params[cluster_indices]
        plt.scatter(
            cluster_params[:, 1],
            cluster_params[:, 0],
            c=[colors[i]],
            label=f"Cluster {cluster}",
        )

    plt.xlabel("Coherence (τ)")
    plt.ylabel("Resonance (θ)")
    plt.title("DBSCAN Clusters in Parameter Space")
    plt.legend()
    plt.grid(True)

    # Save figure
    dbscan_params_fig_path = os.path.join(FIGURES_DIR, "dbscan_parameter_space.png")
    plt.savefig(dbscan_params_fig_path, dpi=300)
    mlflow.log_artifact(dbscan_params_fig_path)

    plt.show()
```


```python
# Visualize Spectral Clustering results
with mlflow.start_run(run_name="spectral_visualization") as run:
    # Create a colormap for clusters
    colors = cm.plasma(np.linspace(0, 1, n_clusters_spectral))

    # 1. Plot representative curves from each cluster
    plt.figure(figsize=(15, 10))

    # Plot curves from each cluster
    for cluster in range(n_clusters_spectral):
        # Get indices for this cluster
        cluster_indices = np.where(labels_spectral == cluster)[0]

        # Find curve closest to cluster center
        within_cluster_dist = dtw_dist[np.ix_(cluster_indices, cluster_indices)]
        center_idx = cluster_indices[np.argmin(within_cluster_dist.sum(axis=1))]

        # Plot cluster center
        plt.plot(
            range(t_max + 1),
            decay_curves[center_idx],
            color=colors[cluster],
            linewidth=3,
            label=f"Cluster {cluster} (center)",
        )

        # Plot a few random members
        sample_size = min(3, len(cluster_indices) - 1)
        if sample_size > 0:
            other_indices = [idx for idx in cluster_indices if idx != center_idx]
            sample_indices = np.random.choice(other_indices, sample_size, replace=False)
            for idx in sample_indices:
                plt.plot(
                    range(t_max + 1),
                    decay_curves[idx],
                    color=colors[cluster],
                    alpha=0.5,
                )

    plt.xlabel("Time (t)")
    plt.ylabel("Truth Score T(s, t)")
    plt.title("Representative Decay Curves by Spectral Cluster")
    plt.legend()
    plt.grid(True)

    # Save figure
    spectral_curves_fig_path = os.path.join(
        FIGURES_DIR, "spectral_representative_curves.png"
    )
    plt.savefig(spectral_curves_fig_path, dpi=300)
    mlflow.log_artifact(spectral_curves_fig_path)

    plt.show()

    # 2. Plot clusters in parameter space (θ,τ)
    plt.figure(figsize=(12, 10))

    # Plot each cluster
    for cluster in range(n_clusters_spectral):
        cluster_indices = np.where(labels_spectral == cluster)[0]
        cluster_params = curve_params[cluster_indices]
        plt.scatter(
            cluster_params[:, 1],
            cluster_params[:, 0],
            c=[colors[cluster]],
            label=f"Cluster {cluster}",
        )

    plt.xlabel("Coherence (τ)")
    plt.ylabel("Resonance (θ)")
    plt.title("Spectral Clusters in Parameter Space")
    plt.legend()
    plt.grid(True)

    # Save figure
    spectral_params_fig_path = os.path.join(FIGURES_DIR, "spectral_parameter_space.png")
    plt.savefig(spectral_params_fig_path, dpi=300)
    mlflow.log_artifact(spectral_params_fig_path)

    plt.show()
```


```python
# 3D visualization combining time series shape and parameter space
with mlflow.start_run(run_name="3d_visualization") as run:
    # Create 3D plot for DBSCAN
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Add cluster information to parameter data
    param_cluster_data = np.column_stack((curve_params, labels_dbscan))

    # Exclude noise points for clearer visualization
    valid_data = param_cluster_data[param_cluster_data[:, 2] != -1]

    # Extract parameters and cluster labels
    thetas_valid = valid_data[:, 0]
    taus_valid = valid_data[:, 1]
    labels_valid = valid_data[:, 2].astype(int)

    # Compute a representative value for each curve (e.g., decay rate)
    decay_rates = []
    for idx in range(len(decay_curves)):
        if labels_dbscan[idx] != -1:  # Skip noise points
            curve = decay_curves[idx]
            # Calculate decay rate (change over time)
            decay_rate = (curve[0] - curve[-1]) / t_max
            decay_rates.append(decay_rate)

    # Create scatter plot
    unique_clusters = sorted(set(labels_valid))
    colors = cm.viridis(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster in enumerate(unique_clusters):
        cluster_mask = labels_valid == cluster
        ax.scatter(
            taus_valid[cluster_mask],
            thetas_valid[cluster_mask],
            decay_rates[np.where(cluster_mask)[0]],
            c=[colors[i]],
            label=f"Cluster {int(cluster)}",
        )

    ax.set_xlabel("Coherence (τ)")
    ax.set_ylabel("Resonance (θ)")
    ax.set_zlabel("Decay Rate")
    ax.set_title("DBSCAN Clusters: Parameter Space + Decay Dynamics")
    plt.legend()

    # Save figure
    dbscan_3d_fig_path = os.path.join(FIGURES_DIR, "dbscan_3d_visualization.png")
    plt.savefig(dbscan_3d_fig_path, dpi=300)
    mlflow.log_artifact(dbscan_3d_fig_path)

    plt.show()

    # Create comparison table of clustering results
    cluster_stats = {
        "method": ["DBSCAN", "Spectral"],
        "n_clusters": [n_clusters_dbscan, n_clusters_spectral],
        "noise_points": [n_noise_dbscan, 0],  # Spectral assigns all points to clusters
    }

    # Create DataFrame
    stats_df = pd.DataFrame(cluster_stats)

    # Display and save
    print("Clustering Method Comparison:")
    display(stats_df)

    # Save to CSV
    stats_path = os.path.join(DATA_DIR, "clustering_comparison.csv")
    stats_df.to_csv(stats_path, index=False)
    mlflow.log_artifact(stats_path)

    # Log summary metrics
    mlflow.log_metric("dbscan_decay_shape_clusters", n_clusters_dbscan)
    mlflow.log_metric("spectral_decay_shape_clusters", n_clusters_spectral)
```


```python
# Compare parameter-based clustering with shape-based clustering
with mlflow.start_run(run_name="cluster_comparison") as run:
    # Create a comparison visualization
    plt.figure(figsize=(15, 10))

    # Setup subplots
    ax1 = plt.subplot(221)  # Parameter K-means
    ax2 = plt.subplot(222)  # DTW DBSCAN
    ax3 = plt.subplot(223)  # DTW Spectral
    ax4 = plt.subplot(224)  # Agreement heatmap

    # 1. Plot original parameter-based K-means clusters
    for cluster in range(n_clusters):  # From your original K-means
        cluster_points = stable_regions[stable_regions["cluster"] == cluster]
        ax1.scatter(
            cluster_points["tau"], cluster_points["theta"], label=f"Cluster {cluster}"
        )

    ax1.set_xlabel("Coherence (τ)")
    ax1.set_ylabel("Resonance (θ)")
    ax1.set_title("Parameter Space K-means")
    ax1.legend()
    ax1.grid(True)

    # 2. Plot DTW DBSCAN clusters
    # Get valid points (excluding noise)
    valid_indices = np.where(labels_dbscan != -1)[0]
    valid_params = curve_params[valid_indices]
    valid_labels = labels_dbscan[valid_indices]

    # Create colormap for DBSCAN
    dbscan_colors = cm.viridis(np.linspace(0, 1, len(set(valid_labels))))

    for i, cluster in enumerate(sorted(set(valid_labels))):
        cluster_mask = valid_labels == cluster
        ax2.scatter(
            valid_params[cluster_mask, 1],
            valid_params[cluster_mask, 0],
            c=[dbscan_colors[i]],
            label=f"Cluster {int(cluster)}",
        )

    ax2.set_xlabel("Coherence (τ)")
    ax2.set_ylabel("Resonance (θ)")
    ax2.set_title("DTW DBSCAN Clusters")
    ax2.legend()
    ax2.grid(True)

    # 3. Plot DTW Spectral clusters
    spectral_colors = cm.plasma(np.linspace(0, 1, n_clusters_spectral))

    for cluster in range(n_clusters_spectral):
        cluster_indices = np.where(labels_spectral == cluster)[0]
        cluster_params = curve_params[cluster_indices]
        ax3.scatter(
            cluster_params[:, 1],
            cluster_params[:, 0],
            c=[spectral_colors[cluster]],
            label=f"Cluster {cluster}",
        )

    ax3.set_xlabel("Coherence (τ)")
    ax3.set_ylabel("Resonance (θ)")
    ax3.set_title("DTW Spectral Clusters")
    ax3.legend()
    ax3.grid(True)

    # 4. Create confusion/agreement matrix between DBSCAN and Spectral
    # This shows how often they agree on cluster assignments

    # Filter for points that aren't noise in DBSCAN
    non_noise_idx = np.where(labels_dbscan != -1)[0]
    dbscan_filtered = labels_dbscan[non_noise_idx]
    spectral_filtered = labels_spectral[non_noise_idx]

    # Create contingency table
    contingency = confusion_matrix(dbscan_filtered, spectral_filtered)

    # Normalize by rows (DBSCAN clusters)
    contingency_norm = (
        contingency.astype("float") / contingency.sum(axis=1)[:, np.newaxis]
    )

    # Plot heatmap
    im = ax4.imshow(contingency_norm, cmap="Blues")

    # Add labels
    ax4.set_xlabel("Spectral Clusters")
    ax4.set_ylabel("DBSCAN Clusters")
    ax4.set_title("Cluster Assignment Agreement")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label("Proportion")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    comparison_fig_path = os.path.join(FIGURES_DIR, "clustering_method_comparison.png")
    plt.savefig(comparison_fig_path, dpi=300)
    mlflow.log_artifact(comparison_fig_path)

    plt.show()

    # Compute agreement metrics
    ari = adjusted_rand_score(dbscan_filtered, spectral_filtered)
    ami = adjusted_mutual_info_score(dbscan_filtered, spectral_filtered)

    mlflow.log_metric("adjusted_rand_index", ari)
    mlflow.log_metric("adjusted_mutual_info", ami)

    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Adjusted Mutual Information: {ami:.4f}")

    # Create summary of findings
    summary = f"""
    # Siglet Shape Clustering Summary

    ## Key Findings

    1. **DBSCAN identified {n_clusters_dbscan} natural clusters** based on decay curve shapes
       - {n_noise_dbscan} points classified as noise

    2. **Spectral Clustering produced {n_clusters_spectral} clusters** with smoother boundaries

    3. **Agreement between methods: ARI={ari:.4f}, AMI={ami:.4f}**
       - Higher values indicate more agreement (max=1.0)

    4. **Shape vs. Parameter Clustering**
       - Parameter clustering: Based on (θ,τ) coordinates
       - Shape clustering: Based on temporal dynamics

    ## Implications for Siglet Alphabet

    The stable regions in parameter space correspond to distinct temporal behaviors.
    These regions form the foundation for the siglet alphabet where each symbol has:

    1. A characteristic resonance angle (θ)
    2. A coherence factor (τ)
    3. A distinct temporal decay pattern

    ## Next Steps

    1. Fine-tune DTW parameters for optimal shape separation
    2. Export selected curves as canonical siglet prototypes
    3. Map to quantum circuit parameters
    """

    summary_path = os.path.join(NOTES_DIR, "siglet_shape_clustering_summary.md")
    with open(summary_path, "w") as f:
        f.write(summary)

    mlflow.log_artifact("siglet_shape_clustering_summary.md")
```

# Section Two: A Continued Exploration...

## Hybrid Enumeration + Optuna Search


```python
# Function to calculate commutation relations between operators
def commutation_relation(op1, op2, qubit):
    """
    Calculate the commutation relation between two primitive operators.

    Args:
        op1, op2: Indices of primitive operators
        qubit: SigletQubit instance to use for calculations

    Returns:
        Commutation value (lower is better for stable algebra)
    """
    # Get the matrices for each operator
    theta_values = np.linspace(0, np.pi, 10)
    tau_values = np.linspace(0.1, 0.9, 10)

    # Sample parameters for stability calculation
    commutation_values = []

    for theta in theta_values[:3]:  # Use subset for efficiency
        for tau in tau_values[:3]:
            # Set qubit parameters
            qubit.theta = theta
            qubit.tau = tau

            # Apply operators and calculate commutation
            op1_result = qubit.apply_primitive(op1)
            op2_result = qubit.apply_primitive(op2)

            # Calculate [op1, op2] = op1*op2 - op2*op1
            # For our purposes, we'll use the truth score difference
            comm = abs(op1_result * op2_result - op2_result * op1_result)
            commutation_values.append(comm)

    # Return average commutation value (lower means better commuting)
    return np.mean(commutation_values)
```


```python
def generate_decay_curves_for_triple(triple, qubit, t_max=100):
    """
    Generate decay curves using the triple of operators on various theta/tau configurations.

    Args:
        triple: Tuple of three operator indices
        qubit: SigletQubit instance
        t_max: Maximum time steps

    Returns:
        decay_curves: Array of decay curves
        curve_params: Array of parameter pairs (theta, tau)
    """
    # Create parameter grid
    theta_values = np.linspace(0, np.pi, 20)
    tau_values = np.linspace(0.1, 0.9, 20)

    # Prepare arrays for storing curves and parameters
    curves = []
    params = []

    # Generate curves for each parameter combination
    for theta in theta_values:
        for tau in tau_values:
            qubit.theta = theta
            qubit.tau = tau

            # Generate decay curve using this triple of operators
            curve = []
            for t in range(t_max + 1):
                # Apply the three operators in sequence and get truth score
                qubit.reset()
                qubit.apply_primitive(triple[0])
                qubit.apply_primitive(triple[1])
                qubit.apply_primitive(triple[2])
                truth_score = qubit.get_truth_score(t)
                curve.append(truth_score)

            curves.append(curve)
            params.append((theta, tau))

    return np.array(curves), np.array(params)
```


```python
def cdist_dtw(curves):
    """
    Compute distance matrix between curves using DTW.

    Args:
        curves: Array of time series

    Returns:
        Distance matrix
    """
    n_curves = len(curves)
    dist_matrix = np.zeros((n_curves, n_curves))

    for i in range(n_curves):
        for j in range(i, n_curves):
            # Use fastdtw for efficient computation
            distance, _ = fastdtw(curves[i], curves[j])
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance

    return dist_matrix
```


```python
def cluster_quality(curves, dist_matrix, eps=0.2, min_samples=5):
    """
    Evaluate clustering quality for a set of curves.

    Args:
        curves: Array of time series
        dist_matrix: Precomputed distance matrix
        eps, min_samples: DBSCAN parameters

    Returns:
        Dictionary with cluster quality metrics
    """
    # Normalize distance matrix for clustering
    dist_normalized = dist_matrix / np.max(dist_matrix)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(dist_normalized)

    # Count clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # If we have at least 2 clusters and not all points are noise, calculate metrics
    if n_clusters >= 2 and n_noise < len(curves):
        # Filter out noise points for metric calculation
        non_noise_indices = np.where(labels != -1)[0]
        if len(non_noise_indices) >= 2:  # Need at least 2 points for metrics
            # Calculate silhouette score (higher is better)
            try:
                silhouette = silhouette_score(
                    dist_matrix[np.ix_(non_noise_indices, non_noise_indices)],
                    labels[non_noise_indices],
                    metric="precomputed",
                )
            except:
                silhouette = -1  # Fallback if calculation fails

            # Calculate Davies-Bouldin index (lower is better)
            try:
                # For DB index, we need to use Euclidean in feature space instead of precomputed
                # Use PCA to project to 2D for simplified calculation
                pca = PCA(n_components=2)
                curves_2d = pca.fit_transform(curves[non_noise_indices])
                db_index = davies_bouldin_score(curves_2d, labels[non_noise_indices])
            except:
                db_index = float("inf")  # Fallback if calculation fails

            # Cluster separation metric (higher is better)
            cluster_sep = silhouette * (1 / (1 + db_index))
        else:
            cluster_sep = -1
    else:
        cluster_sep = -1

    return {
        "n_clusters": n_clusters,
        "noise_percentage": n_noise / len(curves) * 100,
        "cluster_sep": cluster_sep,
    }
```


```python
def evaluate_triple(triple, weights=None, qubit=None):
    """
    Evaluate a triple of primitive operators based on multiple metrics.

    Args:
        triple: Tuple of three operator indices
        weights: Optional dictionary of weights for the metrics
        qubit: SigletQubit instance (will create if None)

    Returns:
        Dictionary of evaluation metrics
    """

    qubit = SigletQubit()

    # 1. Calculate commutation sum (lower is better, so invert for consistent direction)
    comm_sum = 0
    for op1, op2 in itertools.combinations(triple, 2):
        comm_sum += commutation_relation(op1, op2, qubit)
    # Invert and normalize to 0-1 range (assuming typical values range 0-3)
    comm_sum = 1.0 - min(1.0, comm_sum / 3.0)

    # 2. Generate decay curves and calculate DTW distances
    decay_curves, curve_params = generate_decay_curves_for_triple(triple, qubit)
    dist_matrix = cdist_dtw(decay_curves)

    # 3. Evaluate cluster quality
    cluster_metrics = cluster_quality(decay_curves, dist_matrix)
    cluster_sep = (
        max(0, cluster_metrics["cluster_sep"])
        if cluster_metrics["cluster_sep"] != -1
        else 0
    )

    # 4. Evaluate noise resilience
    # Add small amounts of noise and see how stable the clustering remains
    noise_levels = [0.01, 0.05, 0.1]
    stability_scores = []

    base_labels = DBSCAN(metric="precomputed", eps=0.2, min_samples=5).fit_predict(
        dist_matrix / np.max(dist_matrix)
    )

    for noise_level in noise_levels:
        # Add noise to curves
        noisy_curves = decay_curves + np.random.normal(
            0, noise_level, decay_curves.shape
        )
        # Recalculate distances
        noisy_dist = cdist_dtw(noisy_curves)
        # Recluster
        noisy_labels = DBSCAN(metric="precomputed", eps=0.2, min_samples=5).fit_predict(
            noisy_dist / np.max(noisy_dist)
        )

        # Calculate adjusted Rand index to measure clustering stability
        from sklearn.metrics.cluster import adjusted_rand_score

        try:
            ari = adjusted_rand_score(base_labels, noisy_labels)
            stability_scores.append(ari)
        except:
            stability_scores.append(0)

    # Average stability score
    noise_resilience = np.mean(stability_scores) if stability_scores else 0

    return {
        "comm_sum": comm_sum,
        "cluster_sep": cluster_sep,
        "noise_resilience": noise_resilience,
        "n_clusters": cluster_metrics["n_clusters"],
        "noise_percentage": cluster_metrics["noise_percentage"],
    }
```


```python
# Now implement the main code block with MLflow tracking
with mlflow.start_run(run_name="operator_triple_optimization") as run:
    # Log start time
    start_time = time.time()

    # Define primitive operators
    primitives = list(range(12))  # ε₁…ε₁₂ operators
    mlflow.log_param("n_primitives", len(primitives))

    # Generate all possible triples
    triples = list(itertools.combinations(primitives, 3))
    mlflow.log_param("n_triples", len(triples))

    # Create SigletQubit instance
    qubit = SigletQubit()

    # Step A: Quick proxy evaluation on all triples
    mlflow.log_param("proxy_metric", "comm_sum")
    print(f"Step A: Evaluating proxy scores for {len(triples)} triples...")

    proxy_scores = {}
    for i, tri in enumerate(triples):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(triples)} triples evaluated")
        proxy_scores[tri] = evaluate_triple(tri, qubit=qubit)["comm_sum"]

    # Sort and select top K triples
    K = 50
    mlflow.log_param("top_K", K)
    topK = sorted(triples, key=lambda t: -proxy_scores[t])[:K]

    # Log top triples and their proxy scores
    for i, tri in enumerate(topK):
        mlflow.log_param(f"top_triple_{i}", tri)
        mlflow.log_metric(f"proxy_score_{i}", proxy_scores[tri])

    # Step B: Optuna optimization on topK triples
    print(f"Step B: Running Optuna optimization on top {K} triples...")

    def objective(trial):
        tri = trial.suggest_categorical("triple", topK)
        weights = {
            "α": trial.suggest_float("α", 0, 1),
            "β": trial.suggest_float("β", 0, 1),
            "γ": trial.suggest_float("γ", 0, 1),
        }

        # Normalize weights to sum to 1
        weight_sum = weights["α"] + weights["β"] + weights["γ"]
        weights = {k: v / weight_sum for k, v in weights.items()}

        # Log the trial params
        mlflow.log_params(
            {
                f"trial_{trial.number}_triple": tri,
                f"trial_{trial.number}_α": weights["α"],
                f"trial_{trial.number}_β": weights["β"],
                f"trial_{trial.number}_γ": weights["γ"],
            }
        )

        # Evaluate triple with the given weights
        sc = evaluate_triple(tri, weights, qubit=qubit)

        # Log the component scores
        mlflow.log_metrics(
            {
                f"trial_{trial.number}_comm_sum": sc["comm_sum"],
                f"trial_{trial.number}_cluster_sep": sc["cluster_sep"],
                f"trial_{trial.number}_noise_resilience": sc["noise_resilience"],
            }
        )

        # Calculate composite score
        composite_score = (
            weights["α"] * sc["comm_sum"]
            + weights["β"] * sc["cluster_sep"]
            + weights["γ"] * sc["noise_resilience"]
        )

        mlflow.log_metric(f"trial_{trial.number}_composite_score", composite_score)

        return composite_score

    # Create and run Optuna study
    mlflow.log_param("n_trials", 200)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    # Get and log best results
    best_triple = study.best_params["triple"]
    best_α = study.best_params["α"]
    best_β = study.best_params["β"]
    best_γ = study.best_params["γ"]

    mlflow.log_params(
        {
            "best_triple": best_triple,
            "best_α": best_α,
            "best_β": best_β,
            "best_γ": best_γ,
            "best_score": study.best_value,
        }
    )

    # Log total execution time
    execution_time = time.time() - start_time
    mlflow.log_metric("execution_time", execution_time)

    # Create a visualization of the optimization process
    plt.figure(figsize=(10, 6))
    trials = study.trials
    scores = [t.value for t in trials]
    plt.plot(range(len(scores)), scores, "o-")
    plt.xlabel("Trial Number")
    plt.ylabel("Composite Score")
    plt.title("Optimization Progress")
    plt.grid(True)

    # Save and log the figure
    optuna_figures_path = os.path.join(FIGURES_DIR, "optuna_optimization.png")
    plt.savefig(optuna_figures_path, dpi=300)
    mlflow.log_artifact(optuna_figures_path)

    # Print results
    print("Best triple:", best_triple)
    print(f"Best weights: α={best_α:.3f}, β={best_β:.3f}, γ={best_γ:.3f}")
    print(f"Best composite score: {study.best_value:.4f}")
    print(f"Total execution time: {execution_time:.2f} seconds")
```

## Cluster Drift with Noise


```python
def generate_random_siglets(n, seed=None):
    """
    Generate n random siglet parameter sets.

    Args:
        n: Number of siglets to generate
        seed: Random seed for reproducibility

    Returns:
        List of siglet parameter tuples (θ, τ, r, μ, c)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate parameters within reasonable ranges
    siglets = []
    for _ in range(n):
        # θ (theta): resonance parameter between 0 and π
        theta = np.random.uniform(0, np.pi)

        # τ (tau): coherence parameter between 0.1 and 0.9
        tau = np.random.uniform(0.1, 0.9)

        # r: amplitude scaling factor
        r = np.random.uniform(0.8, 1.2)

        # μ (mu): modulation factor
        mu = np.random.uniform(0.7, 1.3)

        # c: constant offset
        c = np.random.uniform(0.9, 1.1)

        siglets.append((theta, tau, r, mu, c))

    return siglets
```


```python
def compute_decay_matrix(eps_ops, siglets, noise_level=0.01):
    """
    Compute decay matrix for epsilon operators applied to siglets with noise.

    Args:
        eps_ops: List of epsilon operators (indices or operator objects)
        siglets: List of siglet parameter tuples (θ, τ, r, μ, c)
        noise_level: Standard deviation of Gaussian noise added to θ and τ

    Returns:
        Array of flattened trajectories for each operator
    """
    T = []
    for ε in eps_ops:
        trajs = []
        for s in siglets:
            θ, τ, r, μ, c = s
            noisy_θ = θ + np.random.normal(0, noise_level)
            noisy_τ = τ + np.random.normal(0, noise_level)
            traj = r * np.cos(noisy_θ) * (noisy_τ ** np.arange(0, 10)) * μ * c
            trajs.append(traj)
        T.append(np.array(trajs).flatten())
    return np.array(T)
```


```python
def evaluate_cluster_stability(
    eps_ops, siglets, n_clusters=3, n_trials=10, noise_levels=[0.01, 0.02, 0.05, 0.1]
):
    """
    Evaluate how noise affects cluster stability.

    Args:
        eps_ops: List of epsilon operators
        siglets: List of siglet parameters
        n_clusters: Number of clusters for spectral clustering
        n_trials: Number of trials at each noise level
        noise_levels: List of noise levels to test

    Returns:
        Dictionary of stability metrics
    """
    # First, generate baseline clustering with no noise
    baseline_matrix = compute_decay_matrix(eps_ops, siglets, noise_level=0)
    baseline_clustering = SpectralClustering(
        n_clusters=n_clusters, affinity="rbf", random_state=42
    ).fit(baseline_matrix)
    baseline_labels = baseline_clustering.labels_

    # Track stability metrics at each noise level
    stability_results = {
        "noise_levels": noise_levels,
        "mean_ari": [],
        "std_ari": [],
        "mean_silhouette": [],
        "std_silhouette": [],
    }

    for noise in noise_levels:
        ari_scores = []
        silhouette_scores = []

        for _ in range(n_trials):
            # Generate noisy decay matrix
            noisy_matrix = compute_decay_matrix(eps_ops, siglets, noise_level=noise)

            # Perform clustering
            noisy_clustering = SpectralClustering(
                n_clusters=n_clusters, affinity="rbf", random_state=42
            ).fit(noisy_matrix)
            noisy_labels = noisy_clustering.labels_

            # Calculate Adjusted Rand Index (similarity between clusterings)
            ari = adjusted_rand_score(baseline_labels, noisy_labels)
            ari_scores.append(ari)

            # Calculate Silhouette Score (cluster quality)
            try:
                sil = silhouette_score(noisy_matrix, noisy_labels)
                silhouette_scores.append(sil)
            except:
                # If only one cluster is found, silhouette score can't be computed
                silhouette_scores.append(0)

        # Record metrics
        stability_results["mean_ari"].append(np.mean(ari_scores))
        stability_results["std_ari"].append(np.std(ari_scores))
        stability_results["mean_silhouette"].append(np.mean(silhouette_scores))
        stability_results["std_silhouette"].append(np.std(silhouette_scores))

    return stability_results
```


```python
def visualize_stability_results(stability_results, title_prefix="Cluster"):
    """
    Visualize cluster stability results.

    Args:
        stability_results: Dictionary with stability metrics
        title_prefix: Prefix for plot titles

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot ARI (Adjusted Rand Index)
    axes[0].errorbar(
        stability_results["noise_levels"],
        stability_results["mean_ari"],
        yerr=stability_results["std_ari"],
        marker="o",
        linestyle="-",
        capsize=4,
    )
    axes[0].set_xlabel("Noise Level")
    axes[0].set_ylabel("Adjusted Rand Index")
    axes[0].set_title(f"{title_prefix} Stability vs Noise Level")
    axes[0].grid(True)

    # Plot Silhouette Score
    axes[1].errorbar(
        stability_results["noise_levels"],
        stability_results["mean_silhouette"],
        yerr=stability_results["std_silhouette"],
        marker="o",
        linestyle="-",
        capsize=4,
    )
    axes[1].set_xlabel("Noise Level")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title(f"{title_prefix} Quality vs Noise Level")
    axes[1].grid(True)

    plt.tight_layout()
    return fig
```


```python
# Main code with MLflow tracking
with mlflow.start_run(run_name="cluster_drift_analysis") as run:
    # 1. Log parameters
    n_siglets = 100
    n_clusters = 3
    noise_level = 0.02
    base_noise_levels = [0.01, 0.02, 0.05, 0.1]
    n_stability_trials = 5

    mlflow.log_params(
        {
            "n_siglets": n_siglets,
            "n_clusters": n_clusters,
            "noise_level": noise_level,
            "n_stability_trials": n_stability_trials,
        }
    )

    # Record start time for performance tracking
    start_time = time.time()

    # 2. Generate siglets
    np.random.seed(42)  # For reproducibility
    siglets = generate_random_siglets(n_siglets)

    # 3. Define epsilon operators (this would normally come from elsewhere in the notebook)
    # For illustration, we'll create synthetic operators with indices
    op1, op2, op3 = 0, 1, 2  # Replace with actual operator values from earlier cells
    eps_ops = [op1, op2, op3]  # your chosen operators

    mlflow.log_param("operators", eps_ops)

    # 4. Compute decay matrix and perform initial clustering
    print("Computing decay matrix with noise level:", noise_level)
    decay_mat = compute_decay_matrix(eps_ops, siglets, noise_level=noise_level)

    # 5. Perform clustering
    clust = SpectralClustering(
        n_clusters=n_clusters, affinity="rbf", random_state=42
    ).fit(decay_mat)

    # 6. Log basic metrics
    cluster_counts = np.bincount(clust.labels_)
    for i, count in enumerate(cluster_counts):
        mlflow.log_metric(f"cluster_{i}_size", count)

    # 7. Evaluate stability across noise levels
    print("Evaluating cluster stability across noise levels...")
    stability_results = evaluate_cluster_stability(
        eps_ops,
        siglets,
        n_clusters=n_clusters,
        n_trials=n_stability_trials,
        noise_levels=base_noise_levels,
    )

    # 8. Log stability metrics
    for i, noise in enumerate(stability_results["noise_levels"]):
        mlflow.log_metric(f"noise_{noise}_mean_ari", stability_results["mean_ari"][i])
        mlflow.log_metric(f"noise_{noise}_std_ari", stability_results["std_ari"][i])
        mlflow.log_metric(
            f"noise_{noise}_mean_silhouette", stability_results["mean_silhouette"][i]
        )
        mlflow.log_metric(
            f"noise_{noise}_std_silhouette", stability_results["std_silhouette"][i]
        )

    # Plot cluster distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(cluster_counts)), cluster_counts)
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Siglets")
    plt.title("Cluster Size Distribution")
    plt.grid(True)
    cluster_plot_path = os.path.join(FIGURES_DIR, "cluster_distribution.png")
    plt.savefig(cluster_plot_path, dpi=300)
    mlflow.log_artifact(cluster_plot_path)

    # Plot stability results
    stability_fig = visualize_stability_results(stability_results)
    stability_plot_path = os.path.join(FIGURES_DIR, "cluster_stability.png")
    stability_fig.savefig(stability_plot_path, dpi=300)
    mlflow.log_artifact(stability_plot_path)

    # 10. Log execution time
    execution_time = time.time() - start_time
    mlflow.log_metric("execution_time", execution_time)

    # 11. Print results
    print("Cluster labels:", clust.labels_)
    print(f"Execution time: {execution_time:.2f} seconds")

    # Print stability metrics
    print("\nStability Metrics:")
    print("------------------")
    print("Noise Level | ARI (mean ± std) | Silhouette (mean ± std)")
    for i, noise in enumerate(stability_results["noise_levels"]):
        print(
            f"{noise:.3f} | {stability_results['mean_ari'][i]:.3f} ± {stability_results['std_ari'][i]:.3f} | {stability_results['mean_silhouette'][i]:.3f} ± {stability_results['std_silhouette'][i]:.3f}"
        )
```

### 3D POVM Projection


```python
# TODO: Expand and fully implement this section
# Choose three Hermitian ε₁, ε₂, ε₃
vectors = []
for ε in [ε1, ε2, ε3]:
    # compute Bloch vector components:
    # b_i = Tr(ρ σ_i) for ρ = ε/||ε||  (normalized operator as state)
    normed = ε / np.linalg.norm(ε)
    b = np.array(
        [
            np.trace(normed.dot(σ_x)).real,
            np.trace(normed.dot(σ_y)).real,
            np.trace(normed.dot(σ_z)).real,
        ]
    )
    vectors.append(b)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
xs, ys, zs = zip(*vectors)
ax.scatter(xs, ys, zs, s=100)
for i, (x, y, z) in enumerate(vectors):
    ax.text(x, y, z, f"ε{i+1}")
ax.set_xlabel("σx proj")
ax.set_ylabel("σy proj")
ax.set_zlabel("σz proj")
plt.title("3D Hilbert Bloch-Vectors of ε-POVMs")
plt.show()
```


```python
# TODO: Expand and fully implement this section
# ── Cell: Truth & Loss Landscape ──

# Grid ranges
eps_range = np.linspace(0.5, 1.0, 50)
mu_range = np.linspace(0.5, 1.0, 50)
c_range = np.linspace(0.5, 1.0, 50)

# fix r, θ, τ for simplicity
r, θ, τ = 1.0, 0.5, 0.8
α, β, γ, δ = 1, 1, 1, 1
hat_ε, hat_μ = 0.8, 0.8

loss_grid = np.zeros((len(eps_range), len(mu_range)))
for i, ε in enumerate(eps_range):
    for j, μ in enumerate(mu_range):
        c = 0.75  # or loop c_range as 3D
        L = α * (1 - np.cos(θ)) + β * (1 - τ) + γ * abs(ε - hat_ε) + δ * abs(μ - hat_μ)
        loss_grid[i, j] = L

plt.imshow(
    loss_grid,
    origin="lower",
    extent=[mu_range[0], mu_range[-1], eps_range[0], eps_range[-1]],
    aspect="auto",
)
plt.colorbar(label="Loss L")
plt.xlabel("μ")
plt.ylabel("ε")
plt.title("Truth-Loss Landscape Slice (c fixed)")
plt.show()
```


```python
# TODO: Expand and fully implement this section
# ── Cell: Reinforcement Loop ──
siglets = initialize_siglets(N=500)
epochs = 20
η = 0.05
θ0 = 0.7

long_term = []
for ep in range(epochs):
    # Fast-Loop sampling & reinforce
    Ts = [(s, truth_score(s)) for s in siglets]
    Ts.sort(key=lambda x: -x[1])
    survivors = [s for s, _ in Ts[:50]]
    # Boost τ
    for s in survivors:
        s.τ = min(1.0, s.τ + η)
    # Slow-Loop admission after ep>10
    if ep >= 10:
        long_term += [
            s for s in survivors if np.mean([truth_score(s, t) for t in range(5)]) > θ0
        ]
    siglets = survivors

print("Long-term siglets count:", len(long_term))
```


```python
# TODO: Expand and fully implement this section
# ── Cell: Robustness Analysis ──


def cluster_stability(eps_ops, siglets, noise_level):
    labels = SpectralClustering(n_clusters=3).fit_predict(
        compute_decay_matrix(eps_ops, siglets, noise_level)
    )
    return silhouette_score(compute_decay_matrix(eps_ops, siglets, noise_level), labels)


for σ in [0.0, 0.01, 0.02, 0.05]:
    score = cluster_stability(eps_ops, siglets, σ)
    print(f"Noise {σ:.3f} → silhouette {score:.3f}")
```


```python
# TODO: Expand and fully implement this section
# ── Cell: Mini-RL Siglet Tuner ──
def reward(s):
    return truth_score(s, t=5) - 0.1 * (abs(s.θ - θ) + abs(s.τ - τ))


for episode in range(10):
    s = random.choice(siglets)
    for step in range(20):
        dθ, dτ = np.random.randn() * 0.01, np.random.randn() * 0.01
        s_new = Siglet(s.θ + dθ, s.τ + dτ, s.r, s.ε, s.μ, s.c)
        if reward(s_new) > reward(s):
            s = s_new
    print(f"Episode {episode}, final Θ={s.θ:.3f}, τ={s.τ:.3f}")
```


```python
# Next steps for quantum extension (preparation for tomorrow's work)
with mlflow.start_run(run_name="quantum_extension_prep") as run:
    # Import quantum libraries (commented out as they're for future use)
    # import qiskit
    # from qiskit import QuantumCircuit, Aer, execute
    # import pennylane as qml

    # Define a basic quantum circuit version of the SigletQubit
    def quantum_siglet_circuit(theta, tau):
        """
        Create a quantum circuit that implements the SigletQubit

        Args:
            theta (float): Resonance angle
            tau (float): Coherence factor

        Returns:
            QuantumCircuit: A Qiskit circuit implementing the siglet
        """
        # This is a placeholder for future quantum implementation
        # Example basic circuit structure:
        """
        qc = QuantumCircuit(1, 1)
        qc.rx(theta, 0)  # Apply resonance angle
        # Apply decoherence modeling based on tau
        # ...
        qc.measure(0, 0)
        return qc
        """
        pass

    # Log notes for tomorrow's work
    notes = """
    Tomorrow's quantum extension will:
    1. Port the SigletQubit model to a 1-qubit quantum circuit
    2. Map the classical parameters (theta, tau) to quantum gate parameters
    3. Test the quantum version against our classical simulation
    4. Evaluate decoherence effects in actual quantum systems vs our model
    """

    notes_path = os.path.join(NOTES_DIR, "quantum_extension_notes.md")
    with open(notes_path, "w") as f:
        f.write(notes)

    mlflow.log_artifact("quantum_extension_notes.md")

    # Log the cluster centers as potential quantum parameter targets
    mlflow.log_artifact(centers_path, "quantum_parameter_targets")

    print("Preparation for quantum extension complete.")
    print(
        "Identified stable siglet parameters as candidates for quantum implementation."
    )
```
