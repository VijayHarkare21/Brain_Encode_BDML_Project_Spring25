#!/usr/bin/env python3
"""
Group-level Representational Similarity Analysis (RSA) and Canonical
Correlation Analysis (CCA) for EEG embeddings versus image- or text-
embedding spaces, with caching of computed embeddings and filtering
only training EEG image files.

Usage
=====
python rsa_cca_group_analysis.py \
    --mode both \
    --eeg_image_dir path/to/eeg_images \
    --eeg_text_dir  path/to/eeg_texts \
    --output_csv   results.csv \
    --figures_dir  figs \
    --subject_aggregate average \          # or stack
    --batch_size 32 \                      # batch size for embeddings
    --embed_cache_dir embed_cache          # where to cache embeddings
"""
from __future__ import annotations

import argparse
import os
import hashlib
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA

# Compatibility patch for older torch versions (missing torch.compiler.disable)
if not hasattr(torch, "compiler"):
    class _CompilerStub:
        @staticmethod
        def disable(recursive: bool = False):
            def decorator(fn):
                return fn
            return decorator
    torch.compiler = _CompilerStub()  # type: ignore


def _zscore_rows(mat: np.ndarray) -> np.ndarray:
    """Feature-wise z-score (μ = 0, σ = 1) across rows."""
    mu = mat.mean(0, keepdims=True)
    std = mat.std(0, ddof=1, keepdims=True) + 1e-9
    return (mat - mu) / std


def load_eeg_image_data(directory: str) -> Dict[str, dict]:
    print(f"[DEBUG] Loading EEG image data from: {directory}")
    data = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".npy"):
            continue
        if "test" in fname.lower():
            print(f"[DEBUG]   Skipping test file: {fname}")
            continue
        key = os.path.splitext(fname)[0]
        print(f"[DEBUG]   Loading training file: {fname} -> key: {key}")
        data[key] = np.load(os.path.join(directory, fname), allow_pickle=True).item()
    print(f"[DEBUG] Loaded {len(data)} EEG image (training) subjects/files")
    return data


def load_eeg_text_data(directory: str) -> Dict[str, List[dict]]:
    print(f"[DEBUG] Loading EEG text data from: {directory}")
    data = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".npy"):
            continue
        print(f"[DEBUG]   Found file: {fname}")
        arr = np.load(os.path.join(directory, fname), allow_pickle=True).item()
        subject = next(iter(arr.keys()))
        print(f"[DEBUG]     Subject: {subject}, entries: {len(arr[subject])}")
        data[subject] = arr[subject]
    print(f"[DEBUG] Loaded text data for {len(data)} subjects")
    return data


def embed_texts(
    texts: List[str],
    model_name: str,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    embed_cache_dir: str = "embed_cache",
) -> np.ndarray:
    """
    Batch-embed a list of texts using a HF model, caching results to avoid
    recomputation on subsequent runs.
    """
    os.makedirs(embed_cache_dir, exist_ok=True)
    # create a hash of the texts list for cache lookup
    text_hash = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()
    cache_path = os.path.join(embed_cache_dir, f"text_{model_name}_{text_hash}.npz")
    if os.path.exists(cache_path):
        print(f"[DEBUG] Loading cached text embeddings from {cache_path}")
        npz = np.load(cache_path, allow_pickle=True)
        return npz["embs"]

    print(f"[DEBUG] Embedding {len(texts)} texts in batches of {batch_size} with model: {model_name}")
    from transformers import AutoTokenizer, AutoModel

    name = model_name.lower()
    pretrained = {"bert": "bert-base-uncased", "roberta": "roberta-base"}.get(name)
    if pretrained is None:
        raise ValueError(f"Unknown text model: {model_name}")

    tok = AutoTokenizer.from_pretrained(pretrained, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(pretrained, cache_dir=cache_dir).to(device).eval()

    embs = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            inputs = tok(batch_texts, return_tensors="pt", truncation=True, padding=True).to(device)
            cls = model(**inputs).last_hidden_state[:, 0]
            batch_embs = cls.cpu().numpy()
            embs.append(batch_embs)
            print(f"[DEBUG]   Processed text batch {b+1}/{num_batches} (size {len(batch_texts)})")
    result = np.vstack(embs)

    # cache to disk
    np.savez(cache_path, embs=result)
    print(f"[DEBUG] Saved text embeddings to cache: {cache_path}")
    return result


def embed_images(
    paths: List[str],
    model_name: str,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    embed_cache_dir: str = "embed_cache",
) -> np.ndarray:
    """
    Batch-embed a list of image paths using a HF model, caching results to avoid
    recomputation on subsequent runs.
    """
    os.makedirs(embed_cache_dir, exist_ok=True)
    # hash the sequence of paths for cache lookup
    path_hash = hashlib.md5("\n".join(paths).encode("utf-8")).hexdigest()
    cache_path = os.path.join(embed_cache_dir, f"image_{model_name}_{path_hash}.npz")
    if os.path.exists(cache_path):
        print(f"[DEBUG] Loading cached image embeddings from {cache_path}")
        npz = np.load(cache_path, allow_pickle=True)
        return npz["embs"]

    print(f"[DEBUG] Embedding {len(paths)} images in batches of {batch_size} with model: {model_name}")
    from PIL import Image

    name = model_name.lower()
    if name == "vit":
        from transformers import ViTModel, ViTImageProcessor
        model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k", cache_dir=cache_dir
        ).to(device).eval()
        processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k", cache_dir=cache_dir
        )
    elif name == "clip":
        from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
        model = AutoModelForZeroShotImageClassification.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=cache_dir
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=cache_dir
        )
    else:
        raise ValueError(f"Unknown image model: {model_name}")

    embs = []
    num_batches = (len(paths) + batch_size - 1) // batch_size
    with torch.no_grad():
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, len(paths))
            batch_paths = paths[start:end]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(device)
            if name == "vit":
                vecs = model(**inputs).last_hidden_state[:, 0, :]
            else:
                vecs = model.get_image_features(**inputs)
            batch_embs = vecs.cpu().numpy()
            embs.append(batch_embs)
            print(f"[DEBUG]   Processed image batch {b+1}/{num_batches} (size {len(batch_paths)})")
    result = np.vstack(embs)

    # cache to disk
    np.savez(cache_path, embs=result)
    print(f"[DEBUG] Saved image embeddings to cache: {cache_path}")
    return result


def _aggregate(
    by_key: defaultdict, how: str, zscore: bool
) -> Tuple[List[str], np.ndarray, Optional[List[int]]]:
    print(f"[DEBUG] Aggregating data: method={how}, zscore={zscore}")
    keys_sorted = sorted(by_key)
    if how == "average":
        mat = np.vstack([np.mean(by_key[k], 0) for k in keys_sorted])
        if zscore:
            mat = _zscore_rows(mat)
        print(f"[DEBUG] Aggregated (average) matrix shape: {mat.shape}")
        return keys_sorted, mat, None

    rows, subj_labels = [], []
    for k in keys_sorted:
        for subj_idx, e in enumerate(by_key[k]):
            rows.append(e)
            subj_labels.append(subj_idx)
    mat = np.vstack(rows)
    if zscore:
        subj_labels_arr = np.asarray(subj_labels)
        for s in np.unique(subj_labels_arr):
            idx = np.where(subj_labels_arr == s)[0]
            mat[idx] = _zscore_rows(mat[idx])
    print(f"[DEBUG] Aggregated (stack) matrix shape: {mat.shape}")
    return keys_sorted, mat, subj_labels


def aggregate_eeg_image(
    img_data: Dict[str, dict],
    eeg_type: str,
    how: str = "average",
    zscore: bool = True,
):
    print(f"[DEBUG] Starting EEG image aggregation for type: {eeg_type}")
    by_path = defaultdict(list)
    for arr in img_data.values():
        for p, e in zip(arr["img_paths"], arr[f"embeds_{eeg_type}"]):
            by_path[p].append(e)
    return _aggregate(by_path, how, zscore)


def aggregate_eeg_text(
    txt_data: Dict[str, List[dict]],
    eeg_type: str,
    how: str = "average",
    zscore: bool = True,
):
    print(f"[DEBUG] Starting EEG text aggregation for type: {eeg_type}")
    by_sent = defaultdict(list)
    for trials in txt_data.values():
        for t in trials:
            if f"embeds_{eeg_type}" in t:
                by_sent[t["content"]].append(t[f"embeds_{eeg_type}"])
    return _aggregate(by_sent, how, zscore)


def compute_rdm(mat: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(mat)
    rdm = 1 - corr
    rdm[np.isnan(rdm)] = 0.0
    return rdm


def visualize_rdm(name: str, rdm: np.ndarray, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.png")
    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(rdm, cmap="viridis", vmin=0, vmax=2)
    plt.title(name)
    plt.colorbar(label="1 − r")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[DEBUG] Saved RDM plot: {path}")


def plot_cca(corrs: List[float], name: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.png")
    plt.figure(figsize=(4, 3))
    plt.bar(np.arange(1, len(corrs) + 1), corrs)
    plt.xlabel("Canonical component")
    plt.ylabel("Correlation")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[DEBUG] Saved CCA plot: {path}")


def rsa_cca_pair(
    eeg_emb: np.ndarray,
    nat_emb: np.ndarray,
    labels: List[str],
    tag: str,
    figdir: str,
    cca_k: int,
):
    print(f"[DEBUG] Running RSA/CCA pair analysis for tag: {tag}")
    print(f"[DEBUG]   EEG emb shape: {eeg_emb.shape}, Nat emb shape: {nat_emb.shape}")

    # RSA
    rdm_eeg, rdm_nat = compute_rdm(eeg_emb), compute_rdm(nat_emb)
    visualize_rdm(f"RDM_EEG_{tag}", rdm_eeg, figdir)
    visualize_rdm(f"RDM_NAT_{tag}", rdm_nat, figdir)
    idx = np.triu_indices(len(labels), 1)
    rho, pval = spearmanr(rdm_eeg[idx], rdm_nat[idx])
    print(f"[DEBUG]   RSA spearman rho={rho}, p-value={pval}")

    # CCA
    cca = CCA(n_components=cca_k)
    cca.fit(eeg_emb, nat_emb)
    U, V = cca.transform(eeg_emb, nat_emb)
    corrs = [float(np.corrcoef(U[:, i], V[:, i])[0, 1]) for i in range(cca_k)]
    print(f"[DEBUG]   CCA correlations: {corrs}")
    plot_cca(corrs, f"CCA_{tag}", figdir)

    return float(rho), float(pval), corrs


def main():
    print("[DEBUG] Starting RSA/CCA group analysis script")
    p = argparse.ArgumentParser(description="Group-level RSA / CCA for EEG embeddings")
    p.add_argument("--mode", choices=["image", "text", "both"], required=True)
    p.add_argument("--eeg_image_dir", type=str, help="Directory with image EEG .npy files")
    p.add_argument("--eeg_text_dir", type=str, help="Directory with text EEG .npy files")
    p.add_argument("--text_models", nargs="+", default=["bert", "roberta"])
    p.add_argument("--image_models", nargs="+", default=["vit", "clip"])
    p.add_argument("--cca_components", type=int, default=5)
    p.add_argument("--output_csv", required=True, type=str)
    p.add_argument("--figures_dir", default="figures", type=str)
    p.add_argument("--cache_dir", default=None, type=str,
                   help="HuggingFace cache dir for model downloads")
    p.add_argument("--embed_cache_dir", default="embed_cache", type=str,
                   help="Directory to cache computed embeddings")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
    )
    p.add_argument(
        "--eeg_embedding_types", nargs="+", default=["cbramod", "labram"]
    )
    p.add_argument(
        "--subject_aggregate",
        choices=["average", "stack"],
        default="average",
        help="Across-subject aggregation method",
    )
    p.add_argument(
        "--no_zscore",
        action="store_true",
        help="Disable feature-wise z-scoring (not recommended)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for image/text embeddings",
    )

    args = p.parse_args()
    print(f"[DEBUG] Parsed args: {args}")

    dev = args.device
    zscore_flag = not args.no_zscore
    results = []

    # IMAGES
    if args.mode in ("image", "both"):
        if args.eeg_image_dir is None:
            raise ValueError("`--eeg_image_dir` must be provided for image mode")
        img_data = load_eeg_image_data(args.eeg_image_dir)
        n_subjects = len(img_data)
        print(f"[DEBUG] Number of image subjects (training only): {n_subjects}")

        for et in args.eeg_embedding_types:
            print(f"[DEBUG] Processing EEG image type: {et}")
            paths, eeg_mat, _ = aggregate_eeg_image(
                img_data, et, args.subject_aggregate, zscore_flag
            )
            for m in args.image_models:
                print(f"[DEBUG]   Using image model: {m}")
                nat_base = embed_images(
                    paths, m, dev,
                    cache_dir=args.cache_dir,
                    batch_size=args.batch_size,
                    embed_cache_dir=args.embed_cache_dir
                )
                if zscore_flag:
                    nat_base = _zscore_rows(nat_base)
                nat_emb = (
                    np.repeat(nat_base, repeats=n_subjects, axis=0)
                    if args.subject_aggregate == "stack"
                    else nat_base
                )
                tag = f"IMG_{et}_{m}_{args.subject_aggregate}"
                labels = (
                    paths
                    if args.subject_aggregate == "average"
                    else [f"S{s}_{p}" for s in range(n_subjects) for p in paths]
                )
                rho, pval, corrs = rsa_cca_pair(
                    eeg_mat,
                    nat_emb,
                    labels,
                    tag,
                    args.figures_dir,
                    args.cca_components,
                )
                print(f"[DEBUG]   Results: rho={rho}, pval={pval}, corrs={corrs}")
                results.append(
                    dict(
                        analysis="image",
                        eeg_type=et,
                        model=m,
                        aggregate=args.subject_aggregate,
                        rsa_rho=rho,
                        rsa_p=pval,
                        cca_corrs=corrs,
                    )
                )

    # TEXT
    if args.mode in ("text", "both"):
        if args.eeg_text_dir is None:
            raise ValueError("`--eeg_text_dir` must be provided for text mode")
        txt_data = load_eeg_text_data(args.eeg_text_dir)
        n_subjects = len(txt_data)
        print(f"[DEBUG] Number of text subjects: {n_subjects}")

        for et in args.eeg_embedding_types:
            print(f"[DEBUG] Processing EEG text type: {et}")
            sents, eeg_mat, _ = aggregate_eeg_text(
                txt_data, et, args.subject_aggregate, zscore_flag
            )
            for m in args.text_models:
                print(f"[DEBUG]   Using text model: {m}")
                nat_base = embed_texts(
                    sents, m, dev,
                    cache_dir=args.cache_dir,
                    batch_size=args.batch_size,
                    embed_cache_dir=args.embed_cache_dir
                )
                if zscore_flag:
                    nat_base = _zscore_rows(nat_base)
                nat_emb = (
                    np.repeat(nat_base, repeats=n_subjects, axis=0)
                    if args.subject_aggregate == "stack"
                    else nat_base
                )
                tag = f"TXT_{et}_{m}_{args.subject_aggregate}"
                labels = (
                    sents
                    if args.subject_aggregate == "average"
                    else [f"S{s}_{t}" for s in range(n_subjects) for t in sents]
                )
                rho, pval, corrs = rsa_cca_pair(
                    eeg_mat,
                    nat_emb,
                    labels,
                    tag,
                    args.figures_dir,
                    args.cca_components,
                )
                print(f"[DEBUG]   Results: rho={rho}, pval={pval}, corrs={corrs}")
                results.append(
                    dict(
                        analysis="text",
                        eeg_type=et,
                        model=m,
                        aggregate=args.subject_aggregate,
                        rsa_rho=rho,
                        rsa_p=pval,
                        cca_corrs=corrs,
                    )
                )

    # OUTPUT
    df = pd.DataFrame(results)
    print(f"[DEBUG] Final results DataFrame shape: {df.shape}")
    print(df.head())
    df.to_csv(args.output_csv, index=False)
    print(f"[DEBUG] Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
