#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
try:
    # openTSNE is much faster and parallel
    from openTSNE import TSNE as oTSNE
    HAVE_OPENTSNE = True
except ImportError:
    from sklearn.manifold import TSNE
    HAVE_OPENTSNE = False
import umap
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
EMBED_DIR = Path("/scratch/vjh9526/bdml_2025/project/datasets/THINGS-EEG/eeg_dataset/embeds")
FIGS_DIR = Path("/scratch/vjh9526/bdml_2025/project/figs")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# LOADING (parallel)
# ----------------------------------------------------------------------------
def _load_file(fp, key):
    data = np.load(fp, allow_pickle=True).item()
    return data[key]

def load_embeddings(model_type, partition, n_jobs):
    key = 'embeds_cbramod' if model_type=='cbramod' else 'embeds_labram'
    pattern = f"*_{partition}_embeds.npy"
    files = list(EMBED_DIR.glob(pattern))
    # parallel map
    embeds_list = Parallel(n_jobs=n_jobs)(
        delayed(_load_file)(fp, key) for fp in files
    )
    return np.vstack(embeds_list)

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Faster cluster & visualize EEG embeddings"
    )
    parser.add_argument('--model',   choices=['cbramod','labram'], required=True)
    parser.add_argument('--partition', choices=['training','test'], default='training')
    parser.add_argument('--method',  choices=['tsne','umap'], default='tsne')
    parser.add_argument('--n_clusters', type=int, default=1654)
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help="Parallel workers for IO & clustering")
    args = parser.parse_args()

    # 1) LOAD
    X = load_embeddings(args.model, args.partition, args.n_jobs)
    print(f"Loaded: {X.shape[0]} samples × {X.shape[1]} dims")

    pca_dims = 200
    # 2) PCA to pca_dims dims
    # print(f"Running PCA → {pca_dims} dims…")
    # X50 = PCA(n_components=pca_dims, svd_solver='randomized', random_state=42)\
    #         .fit_transform(X)
    X50 = X
    
    # 3) CLUSTER with MiniBatchKMeans
    print(f"Clustering into {args.n_clusters} clusters…")
    mbk = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        batch_size= max(1000, args.n_clusters*10),
        random_state=42,
        n_init=1,
        max_no_improvement=10,
        verbose=0
    )
    labels = mbk.fit_predict(X50)

    # 4) REDUCE to 2D for plotting
    print(f"Reducing to 2D via {args.method}…")
    if args.method == 'tsne':
        if HAVE_OPENTSNE:
            reducer = oTSNE(
                n_components=2,
                n_jobs=args.n_jobs,
                metric='cosine',
                # perplexity=30,
                # n_iter=500,
                # exaggeration=4.0,
                random_state=42
            )
        else:
            reducer = TSNE(
                n_components=2,
                random_state=42,
                metric='cosine'
                # n_iter=500,
                # early_exaggeration=12.0
            )
        X2 = reducer.fit(X50)
    else:
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_jobs=args.n_jobs,
            metric='cosine'
        )
        X2 = reducer.fit_transform(X50)

    # 5) PLOT
    plt.figure(figsize=(8,6))
    sc = plt.scatter(X2[:,0], X2[:,1], c=labels, cmap='tab10', s=5)
    plt.title(f"{args.model.upper()} {args.partition.title()} ({args.method.upper()})")
    plt.xlabel('1'); plt.ylabel('2')
    plt.colorbar(sc, label='Cluster')
    out_fp = FIGS_DIR / f"{args.model}_{args.partition}_{args.method}_clusters_image.png"
    plt.savefig(out_fp, dpi=300)
    print(f"Saved → {out_fp}")

if __name__=='__main__':
    main()
