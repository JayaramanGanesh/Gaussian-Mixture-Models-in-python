import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Generate some random data for demonstration purposes
np.random.seed(42)
n_samples = 300
X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0)

# Plot the data before clustering
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='viridis')
plt.title("Data Before Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Fit Gaussian Mixture Model
n_components = 3  # Number of clusters to identify
gmm = GaussianMixture(n_components=n_components)
gmm.fit(X)

# Predict cluster assignments
labels = gmm.predict(X)

# Get the parameters of the learned Gaussians
means = gmm.means_
covariances = gmm.covariances_

# Plot the data after clustering
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.scatter(means[:, 0], means[:, 1], marker='X', c='red', s=100, label='Cluster Centers')

# Plot ellipses representing the covariance of each cluster
for i in range(n_components):
    v, w = np.linalg.eigh(covariances[i])
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # Convert to degrees
    v *= 2  # Multiply by some factor to make the ellipses more visible
    ell = plt.matplotlib.patches.Ellipse(means[i], v[0], v[1], 180 + angle, color='red', alpha=0.5)
    plt.gca().add_patch(ell)

plt.title("Data After Clustering with Gaussian Mixture Model")
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
