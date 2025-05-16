#!/usr/bin/env python3
import os
import argparse
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
EMBED_DIR = Path("/scratch/vjh9526/bdml_2025/project/datasets/THINGS-EEG/eeg_dataset/embeds")
FIGS_DIR = Path("/scratch/vjh9526/bdml_2025/project/figs")

# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------
def load_embeddings(model_type, partition):
    """
    Load embeddings for given model (cbramod or labram) and partition (training/test).
    """
    pattern = f"*_{partition}_embeds.npy"
    files = list(EMBED_DIR.glob(pattern))
    all_embeds = []
    for fp in files:
        data = np.load(fp, allow_pickle=True).item()
        key = 'embeds_cbramod' if model_type == 'cbramod' else 'embeds_labram'
        embeds = data[key]
        all_embeds.append(embeds)
    # concatenate subjects
    return np.vstack(all_embeds)

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cluster and visualize EEG embeddings")
    parser.add_argument('--model', choices=['cbramod', 'labram'], required=True,
                        help="Which embeddings to load: cbramod or labram")
    parser.add_argument('--partition', choices=['training', 'test'], default='training',
                        help="Data partition to use")
    parser.add_argument('--method', choices=['tsne', 'umap'], default='tsne',
                        help="Dimensionality reduction method")
    parser.add_argument('--n_clusters', type=int, default=5,
                        help="Number of clusters for KMeans")
    args = parser.parse_args()

    # Load
    X = load_embeddings(args.model, args.partition)
    print(f"Loaded embeddings: shape={X.shape}")

    # Clustering
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    print(f"Performed KMeans with {args.n_clusters} clusters.")

    # Dimensionality reduction
    if args.method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
    X2 = reducer.fit_transform(X)
    print(f"Reduced to 2D using {args.method}.")

    # Plot
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X2[:,0], X2[:,1], c=labels, cmap='tab10', s=10)
    plt.title(f"{args.model.upper()} Embeddings ({args.partition}) - {args.method.upper()}")
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.colorbar(scatter, label='Cluster')
    out_fp = FIGS_DIR / f"{args.model}_{args.partition}_{args.method}_clusters_image.png"
    plt.savefig(out_fp, dpi=300)
    print(f"Saved plot to {out_fp}")

if __name__ == '__main__':
    main()
