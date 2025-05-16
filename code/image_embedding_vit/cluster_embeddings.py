import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def cluster_embeddings(embeddings, max_clusters=100, plot=True):
    """
    Clusters the embeddings using KMeans and evaluates the clustering performance.

    Parameters:
    - embeddings: numpy array of shape (n_samples, n_features)
    - n_clusters: number of clusters for KMeans
    - plot: whether to plot the PCA-reduced embeddings and cluster centers

    Returns:
    - kmeans: fitted KMeans object
    - silhouette_avg: average silhouette score
    """
    # Determine the optimal number of clusters using the silhouette score method
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        print(f"Number of clusters: {n_clusters}, Silhouette Score: {silhouette_avg}")

    # Choose the number of clusters with the highest silhouette score
    n_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2
    print(f"Optimal number of clusters: {n_clusters}")

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
    print(f"Silhouette Score: {silhouette_avg}")

    # Optionally plot the results
    if plot:
        # Use standard scalar and t-SNE
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        # scaled_centers = scaler.transform(kmeans.cluster_centers_)

        tsne = TSNE(n_components=2, random_state=42)
        tsne_embeddings = tsne.fit_transform(scaled_embeddings)
        tsne_centers = tsne.fit_transform(kmeans.cluster_centers_)

        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
        # plt.scatter(tsne_centers[:, 0], tsne_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
        plt.title('t-SNE of Clustered Embeddings')
        plt.colorbar(label='Cluster Label')
        plt.xlabel('tSNE Component 1')
        plt.ylabel('tSNE Component 2')
        plt.legend()
        plt.savefig('clustered_embeddings.png')
        plt.show()

    return kmeans, silhouette_avg

data = np.load(r"D:\Vijay\NYU\Spring_25\BDMLS\Project\code\image_embedding_vit\vit_embeddings.npy", allow_pickle=True).item()

embeddings = np.array([v for k, v in data.items()])
print(f"Shape of embeddings: {embeddings.shape}")
# Perform clustering on the embeddings
kmeans, silhouette_avg = cluster_embeddings(embeddings.squeeze(1), plot=True)