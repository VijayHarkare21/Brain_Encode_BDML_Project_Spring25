#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# -----------------------------------------------------------------------------
# CONFIGURATION: directory with ZuCo2.0 embeddings
# -----------------------------------------------------------------------------
EMBED_DIR = Path("/scratch/vjh9526/bdml_2025/project/datasets/ZuCo2/2urht/osfstorage/task1 - NR/npy_file/embeds")
FIGS_DIR = Path("/scratch/vjh9526/bdml_2025/project/figs")

# -----------------------------------------------------------------------------
# LOAD EMBEDDINGS
# -----------------------------------------------------------------------------
def load_embeddings(model_type):
    """
    Loads all embeddings (CBraMod or LaBraM) from EMBED_DIR for ZuCo2.0 dataset.
    Returns an (N, D) numpy array of embeddings.
    """
    embeds = []
    for fp in EMBED_DIR.glob("*_embeds.npy"):
        data = np.load(fp, allow_pickle=True).item()
        subj_key = next(iter(data))
        items = data[subj_key]
        for item in items:
            if not item:
                continue
            key = 'embeds_cbramod' if model_type == 'cbramod' else 'embeds_labram'
            emb = item.get(key, None)
            if emb is None:
                continue
            arr = np.array(emb).flatten()
            embeds.append(arr)
    if not embeds:
        raise RuntimeError(f"No {model_type} embeddings found in {EMBED_DIR}")
    return np.vstack(embeds)

# -----------------------------------------------------------------------------
# MAIN: silhouette-based cluster selection + visualization
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Find optimal k via silhouette score and visualize ZuCo2.0 embeddings"
    )
    parser.add_argument('--model', choices=['cbramod', 'labram'], required=True,
                        help="Which embeddings to load: cbramod or labram")
    parser.add_argument('--method', choices=['tsne', 'umap'], default='tsne',
                        help="Dimensionality reduction method for final plot")
    parser.add_argument('--kmin', type=int, default=5,
                        help="Minimum number of clusters to try for silhouette")
    parser.add_argument('--kmax', type=int, default=20,
                        help="Maximum number of clusters to try for silhouette")
    parser.add_argument('--n_init', type=int, default=10,
                        help="Number of initializations for KMeans")
    args = parser.parse_args()

    # Load embeddings
    X = load_embeddings(args.model)
    print(f"Loaded {X.shape[0]} embeddings of dimension {X.shape[1]} for {args.model}")

    # Compute silhouette scores for range of k
    scores = []
    ks = range(args.kmin, args.kmax + 1)
    print("Computing silhouette scores...")
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=args.n_init)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
        print(f"k={k}: silhouette={score:.4f}")

    # Plot silhouette vs k
    plt.figure(figsize=(6,4))
    plt.plot(list(ks), scores, marker='o')
    plt.title(f"Silhouette Scores ({args.model})")
    plt.xlabel('Number of clusters k')
    plt.ylabel('Silhouette score')
    plt.xticks(list(ks))
    out_sil = FIGS_DIR / f"{args.model}_silhouette_scores.png"
    plt.savefig(out_sil, dpi=300)
    print(f"Saved silhouette plot to {out_sil}")

    # Determine optimal k
    best_k = ks[int(np.argmax(scores))]
    print(f"Optimal number of clusters by silhouette: k={best_k}")

    # Final clustering with best_k
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=args.n_init)
    labels_final = kmeans_final.fit_predict(X)

    # Dimensionality reduction
    if args.method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, metric='cosine')
    else:
        reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    X2 = reducer.fit_transform(X)

    # Plot final clusters
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X2[:,0], X2[:,1], c=labels_final, cmap='tab10', s=10)
    plt.title(f"{args.model.upper()} Embeddings - k={best_k} ({args.method.upper()})")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(scatter, label='Cluster')
    out_plot = FIGS_DIR / f"{args.model}_{args.method}_k{best_k}_clusters.png"
    plt.savefig(out_plot, dpi=300)
    print(f"Saved final cluster plot to {out_plot}")

if __name__ == '__main__':
    main()
