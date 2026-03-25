"""
Lab 2.5: K-Means Clustering and PCA Visualization

Apply K-means to find groups in data, choose K using the elbow method
and silhouette score, then visualize with PCA.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score


# ================================================================
# PART 1: Generate and Explore Data
# ================================================================

def generate_data():
    """Generate synthetic high-dimensional clustered data."""
    print("=" * 50)
    print("PART 1: Data Generation and Exploration")
    print("=" * 50)

    # Generate 5 clusters in 10 dimensions
    X, y_true = make_blobs(
        n_samples=500, n_features=10, centers=5,
        cluster_std=1.5, random_state=42
    )

    print(f"Shape: {X.shape}")
    print(f"True number of clusters: {len(np.unique(y_true))}")

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # TODO: Apply PCA to reduce to 2D for visualization
    # pca_vis = PCA(n_components=2)
    # X_2d = pca_vis.fit_transform(X_scaled)

    # TODO: Plot the raw data in 2D (color by true labels for reference)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap="tab10", s=20, alpha=0.7)
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.title("Raw Data Visualized with PCA (colored by true labels)")
    # plt.colorbar(label="True Cluster")
    # plt.grid(True, alpha=0.3)
    # plt.show()

    return X_scaled, y_true


# ================================================================
# PART 2: Elbow Method
# ================================================================

def elbow_method(X):
    """Find the optimal K using the elbow method."""
    print("\n" + "=" * 50)
    print("PART 2: Elbow Method")
    print("=" * 50)

    K_range = range(1, 11)
    inertias = []

    for k in K_range:
        # TODO: Fit KMeans and record inertia
        # kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        # kmeans.fit(X)
        # inertias.append(kmeans.inertia_)
        # print(f"K={k:2d}: Inertia = {kmeans.inertia_:.2f}")
        pass  # Replace with your code

    # TODO: Plot the elbow curve
    # plt.figure(figsize=(8, 5))
    # plt.plot(K_range, inertias, "o-", linewidth=2, markersize=8)
    # plt.xlabel("Number of Clusters (K)")
    # plt.ylabel("Inertia")
    # plt.title("Elbow Method")
    # plt.xticks(K_range)
    # plt.grid(True, alpha=0.3)
    # plt.show()

    pass  # Remove after implementing


# ================================================================
# PART 3: Silhouette Analysis
# ================================================================

def silhouette_analysis(X):
    """Find optimal K using silhouette scores."""
    print("\n" + "=" * 50)
    print("PART 3: Silhouette Analysis")
    print("=" * 50)

    K_range = range(2, 11)
    sil_scores = []

    for k in K_range:
        # TODO: Fit KMeans and compute silhouette score
        # kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        # labels = kmeans.fit_predict(X)
        # score = silhouette_score(X, labels)
        # sil_scores.append(score)
        # print(f"K={k:2d}: Silhouette Score = {score:.4f}")
        pass  # Replace with your code

    # TODO: Plot silhouette scores
    # plt.figure(figsize=(8, 5))
    # plt.plot(K_range, sil_scores, "o-", linewidth=2, markersize=8)
    # plt.xlabel("Number of Clusters (K)")
    # plt.ylabel("Silhouette Score")
    # plt.title("Silhouette Score vs. K")
    # plt.xticks(K_range)
    # plt.grid(True, alpha=0.3)
    # plt.show()

    # TODO: Report the best K
    # best_k = list(K_range)[np.argmax(sil_scores)]
    # print(f"\nBest K by silhouette: {best_k}")
    # return best_k

    return 5  # Placeholder


# ================================================================
# PART 4: Final Clustering
# ================================================================

def final_clustering(X, best_k):
    """Cluster with the chosen K and visualize."""
    print("\n" + "=" * 50)
    print(f"PART 4: Final Clustering (K={best_k})")
    print("=" * 50)

    # TODO: Fit KMeans with the chosen K
    # kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    # labels = kmeans.fit_predict(X)

    # TODO: Reduce to 2D for visualization
    # pca = PCA(n_components=2)
    # X_2d = pca.fit_transform(X)
    # centroids_2d = pca.transform(kmeans.cluster_centers_)

    # TODO: Plot clusters with centroids
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=20, alpha=0.7)
    # plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
    #             c="red", marker="X", s=200, edgecolors="black", linewidths=2,
    #             label="Centroids")
    # plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    # plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    # plt.title(f"K-Means Clustering (K={best_k})")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()

    # return labels
    return None  # Replace


# ================================================================
# PART 5: PCA Explained Variance
# ================================================================

def pca_analysis(X):
    """Analyze how many components are needed."""
    print("\n" + "=" * 50)
    print("PART 5: PCA Explained Variance")
    print("=" * 50)

    # TODO: Fit PCA with all components
    # pca_full = PCA().fit(X)
    # cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

    # TODO: Print variance explained by each component
    # for i, (var, cum) in enumerate(zip(pca_full.explained_variance_ratio_, cumulative_var)):
    #     print(f"  PC{i+1}: {var:.4f} (cumulative: {cum:.4f})")

    # TODO: Plot cumulative explained variance
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, len(cumulative_var)+1), cumulative_var, "o-", markersize=8)
    # plt.axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
    # plt.xlabel("Number of Components")
    # plt.ylabel("Cumulative Explained Variance")
    # plt.title("PCA: Cumulative Explained Variance")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()

    # TODO: Report components needed for 95% variance
    # n_95 = np.argmax(cumulative_var >= 0.95) + 1
    # print(f"\nComponents for 95% variance: {n_95} (out of {X.shape[1]})")

    pass  # Remove after implementing


# ================================================================
# PART 6: Evaluate Against True Labels
# ================================================================

def evaluate_clustering(labels, y_true):
    """Compare cluster assignments against true labels using ARI."""
    print("\n" + "=" * 50)
    print("PART 6: Evaluation Against True Labels")
    print("=" * 50)

    if labels is None:
        print("Complete Part 4 first to get cluster labels.")
        return

    # TODO: Compute Adjusted Rand Index
    # ari = adjusted_rand_score(y_true, labels)
    # print(f"Adjusted Rand Index: {ari:.4f}")
    # print(f"  (1.0 = perfect agreement, 0.0 = random)")

    # TODO: Visualize true labels vs cluster labels side by side
    # pca = PCA(n_components=2)
    # X_2d = pca.fit_transform(X)  # Note: X needs to be passed in
    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap="tab10", s=20)
    # axes[0].set_title("True Labels")
    # axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=20)
    # axes[1].set_title(f"K-Means Labels (ARI={ari:.2f})")
    # plt.tight_layout()
    # plt.show()

    pass  # Remove after implementing


# ================================================================
# MAIN
# ================================================================

def main():
    print("Lab 2.5: K-Means Clustering and PCA Visualization")
    print("=" * 50)

    # Part 1
    X, y_true = generate_data()

    # Part 2
    elbow_method(X)

    # Part 3
    best_k = silhouette_analysis(X)

    # Part 4
    labels = final_clustering(X, best_k)

    # Part 5
    pca_analysis(X)

    # Part 6
    evaluate_clustering(labels, y_true)

    print("\n" + "=" * 50)
    print("Lab complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
