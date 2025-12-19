import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from ipywidgets import interact, IntSlider, Button, interactive_output

# Create some synthetic data
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.0, random_state=42)

# plt.scatter(X[:,0], X[:,1], s=30)
# plt.title("Sample Data for Clustering")
# plt.show()


def plot_kmeans(k):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    plt.figure(figsize=(6,6))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=40, alpha=0.7)
    plt.scatter(centers[:,0], centers[:,1], c='red', marker='X', s=200, label='Centroids')
    plt.title(f'K-Means Clustering (K={k})')
    plt.legend()
    plt.show()


def kmeans_iteration_demo(iterations=5):
    np.random.seed(42)
    k = 3
    centers = X[np.random.choice(len(X), k, replace=False)]

    for i in range(iterations):
        # Assign points
        labels = np.argmin(np.linalg.norm(X[:, None] - centers[None, :], axis=2), axis=1)
        # Plot clusters
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40, alpha=0.6)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
        plt.title(f"Iteration {i + 1}")
        plt.show()

        # Update centroids
        new_centers = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers


interact(kmeans_iteration_demo, iterations=IntSlider(min=1, max=20, step=1, value=10))

#interact(plot_kmeans, k=IntSlider(min=1, max=10, step=1, value=3))
