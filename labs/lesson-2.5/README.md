# Lab 2.5: K-Means Clustering and PCA Visualization

## Objective

Apply K-means clustering to find groups in unlabeled data, use the elbow method and silhouette score to choose K, and visualize high-dimensional clusters using PCA.

## Tasks

### Part 1: Generate and Explore Data
1. Generate synthetic data with `make_blobs` (5 clusters, 10 features)
2. Pretend you do not know the true number of clusters
3. Apply PCA to visualize the raw data in 2D

### Part 2: Elbow Method
1. Run K-means for K = 1 through 10
2. Record inertia for each K
3. Plot the elbow curve and identify the best K

### Part 3: Silhouette Analysis
1. Compute silhouette scores for K = 2 through 10
2. Plot silhouette score vs. K
3. Compare with the elbow method result

### Part 4: Final Clustering
1. Run K-means with the chosen K
2. Visualize clusters in 2D using PCA
3. Plot centroids on the same figure

### Part 5: PCA Explained Variance
1. Fit PCA with all components
2. Plot cumulative explained variance
3. Determine how many components capture 95% of variance

### Part 6: Evaluate Against True Labels
1. Since we generated the data, we know the true labels
2. Compute Adjusted Rand Index (ARI) to measure agreement
3. Discuss: when would you NOT have true labels?

## Getting Started

```bash
pip install scikit-learn numpy matplotlib
python starter.py
```

## Expected Output

- 2D PCA scatter plot of raw data
- Elbow curve and silhouette score plots
- Final cluster visualization with centroids
- Cumulative explained variance plot

## Stretch Goals

- Try DBSCAN on `make_moons` data where K-means fails
- Implement K-means from scratch and compare with sklearn
- Apply clustering to a real dataset (e.g., load_digits) and visualize
