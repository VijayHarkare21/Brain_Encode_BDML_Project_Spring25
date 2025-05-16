#!/usr/bin/env python3
"""
Group-level Representational Similarity Analysis (RSA) and Canonical
Correlation Analysis (CCA) for EEG embeddings versus image- or text-
embedding spaces, with caching of computed embeddings and filtering
only training EEG image files, plus optional cbramod+labram ensemble.

Usage
=====
python rsa_cca_group_analysis.py \
    --mode both \
    --eeg_image_dir path/to/eeg_images \
    --eeg_text_dir  path/to/eeg_texts \
    --output_csv   results.csv \
    --figures_dir  figs \
    --subject_aggregate average          # or stack \
    --batch_size 32                      # batch size for embeddings \
    --embed_cache_dir embed_cache        # where to cache embeddings \
    [--run_ensemble]                     # include ensemble analysis
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

# Compatibility patch for older torch versions
if not hasattr(torch, "compiler"):
    class _CompilerStub:
        @staticmethod
        def disable(recursive: bool = False):
            def decorator(fn): return fn
            return decorator
    torch.compiler = _CompilerStub()


def _zscore_rows(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(0, keepdims=True)
    std = mat.std(0, ddof=1, keepdims=True) + 1e-9
    return (mat - mu) / std


def load_eeg_image_data(directory: str) -> Dict[str, dict]:
    print(f"[DEBUG] Loading EEG image data from: {directory}")
    data = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.npy'): continue
        if 'test' in fname.lower():
            continue
        key = os.path.splitext(fname)[0]
        print(f"[DEBUG]   Loading train file: {fname}")
        data[key] = np.load(os.path.join(directory, fname), allow_pickle=True).item()
    return data


def load_eeg_text_data(directory: str) -> Dict[str, List[dict]]:
    print(f"[DEBUG] Loading EEG text data from: {directory}")
    data = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.npy'): continue
        print(f"[DEBUG]   Loading file: {fname}")
        arr = np.load(os.path.join(directory, fname), allow_pickle=True).item()
        subject = next(iter(arr.keys()))
        data[subject] = arr[subject]
    return data


def embed_texts(
    texts: List[str],
    model_name: str,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    embed_cache_dir: str = "embed_cache",
) -> np.ndarray:
    os.makedirs(embed_cache_dir, exist_ok=True)
    text_hash = hashlib.md5("\n".join(texts).encode()).hexdigest()
    cache_path = os.path.join(embed_cache_dir, f"text_{model_name}_{text_hash}.npz")
    if os.path.exists(cache_path):
        print(f"[DEBUG] Loading cached text embeddings from {cache_path}")
        return np.load(cache_path, allow_pickle=True)["embs"]

    print(f"[DEBUG] Embedding {len(texts)} texts in batches of {batch_size} with model {model_name}")
    from transformers import AutoTokenizer, AutoModel
    pretrained_map = {"bert": "bert-base-uncased", "roberta": "roberta-base"}
    pretrained = pretrained_map.get(model_name.lower())
    tok = AutoTokenizer.from_pretrained(pretrained, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(pretrained, cache_dir=cache_dir).to(device).eval()

    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tok(batch, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            cls = model(**inputs).last_hidden_state[:, 0]
        embs.append(cls.cpu().numpy())
        print(f"[DEBUG]   Text batch {i}-{i+len(batch)} embedded")
    result = np.vstack(embs)
    np.savez(cache_path, embs=result)
    print(f"[DEBUG] Saved text embeddings to {cache_path}")
    return result


def embed_images(
    paths: List[str],
    model_name: str,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    embed_cache_dir: str = "embed_cache",
) -> np.ndarray:
    os.makedirs(embed_cache_dir, exist_ok=True)
    path_hash = hashlib.md5("\n".join(paths).encode()).hexdigest()
    cache_path = os.path.join(embed_cache_dir, f"image_{model_name}_{path_hash}.npz")
    if os.path.exists(cache_path):
        print(f"[DEBUG] Loading cached image embeddings from {cache_path}")
        return np.load(cache_path, allow_pickle=True)["embs"]

    print(f"[DEBUG] Embedding {len(paths)} images in batches of {batch_size} with model {model_name}")
    from PIL import Image
    if model_name.lower() == "vit":
        from transformers import ViTModel, ViTImageProcessor
        model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k", cache_dir=cache_dir
        ).to(device).eval()
        proc = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k", cache_dir=cache_dir
        )
    elif model_name.lower() == "resnet":
        from torchvision.models import resnet50
        import torchvision.transforms as T

        # define the same preprocessing your ResNet expects:
        preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),                                      # PIL → FloatTensor, scales to [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406],            # Imagenet mean/std
                        std=[0.229, 0.224, 0.225]),
        ])
        model = resnet50(pretrained=True).eval().to(device)
        embs = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i : i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch]
            batch_tensor = torch.stack([preprocess(img) for img in imgs]).to(device)
            # inputs = proc(images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                vecs = model(batch_tensor)
            embs.append(vecs.cpu().numpy())
            print(f"[DEBUG]   Image batch {i}-{i+len(batch)} embedded")
        result = np.vstack(embs)
        np.savez(cache_path, embs=result)
        print(f"[DEBUG] Saved image embeddings to {cache_path}")
        return result
    else:
        from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
        model = AutoModelForZeroShotImageClassification.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=cache_dir
        ).to(device).eval()
        proc = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=cache_dir
        )

    embs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            if model_name.lower() == "vit":
                vecs = model(**inputs).last_hidden_state[:, 0, :]
            else:
                vecs = model.get_image_features(**inputs)
        embs.append(vecs.cpu().numpy())
        print(f"[DEBUG]   Image batch {i}-{i+len(batch)} embedded")
    result = np.vstack(embs)
    np.savez(cache_path, embs=result)
    print(f"[DEBUG] Saved image embeddings to {cache_path}")
    return result


def _aggregate(by_key: defaultdict, how: str, zscore: bool) -> Tuple[List[str], np.ndarray, Optional[List[int]]]:
    keys = sorted(by_key)
    if how == "average":
        mat = np.vstack([np.mean(by_key[k], 0) for k in keys])
        if zscore:
            mat = _zscore_rows(mat)
        return keys, mat, None
    rows, labels = [], []
    for k in keys:
        for idx, e in enumerate(by_key[k]):
            rows.append(e)
            labels.append(idx)
    mat = np.vstack(rows)
    if zscore:
        labels_arr = np.array(labels)
        for s in np.unique(labels_arr):
            mat[labels_arr == s] = _zscore_rows(mat[labels_arr == s])
    return keys, mat, labels


# def aggregate_eeg_image(
#     img_data: Dict[str, dict],
#     eeg_type: str,
#     how: str = "average",
#     zscore: bool = True
# ) -> Tuple[List[str], np.ndarray, Optional[List[int]]]:
#     by_path = defaultdict(list)
#     for arr in img_data.values():
#         for p, e in zip(arr["img_paths"], arr[f"embeds_{eeg_type}"]):
#             by_path[p].append(e)
#     return _aggregate(by_path, how, zscore)
def aggregate_eeg_image(
    img_data: Dict[str, dict],
    eeg_type: str,
    how: str = "average",
    zscore: bool = True,
) -> Tuple[List[str], np.ndarray, Optional[List[int]]]:
    by_path = defaultdict(list)
    key = f"embeds_{eeg_type}"
    for subj, arr in img_data.items():
        if "img_paths" not in arr or key not in arr:
            print(f"[WARNING] Subject {subj} missing '{key}', skipping.")
            continue
        paths = arr["img_paths"]
        embs  = arr[key]
        if len(paths) != len(embs):
            raise ValueError(f"Length mismatch for subject {subj}: "
                             f"{len(paths)} paths vs {len(embs)} embeds.")
        for p, e in zip(paths, embs):
            by_path[p].append(e)

    if not by_path:
        raise ValueError(f"No EEG‑image embeddings found for type '{eeg_type}'")

    return _aggregate(by_path, how, zscore)



def aggregate_eeg_text(
    txt_data: Dict[str, List[dict]],
    eeg_type: str,
    how: str = "average",
    zscore: bool = True,
) -> Tuple[List[str], np.ndarray, Optional[List[int]]]:
    # 1) identify sentences seen by all subjects
    subj_ids = sorted(txt_data.keys())
    per_subject_sets = []
    for sid in subj_ids:
        seen = {t["content"] for t in txt_data[sid] if t and f"embeds_{eeg_type}" in t}
        per_subject_sets.append(seen)
    common = sorted(set.intersection(*per_subject_sets))
    if not common:
        raise ValueError("No sentences are common across all subjects.")

    # 2) average repeated trials per subject
    subj_sent_emb: Dict[str, Dict[str, np.ndarray]] = {sid: {} for sid in subj_ids}
    for sid in subj_ids:
        by_sent = defaultdict(list)
        for t in txt_data[sid]:
            if t and f"embeds_{eeg_type}" in t and t["content"] in common:
                by_sent[t["content"]].append(t[f"embeds_{eeg_type}"])
        for sent in common:
            subj_sent_emb[sid][sent] = np.mean(by_sent[sent], axis=0)

    # 3) build final matrix
    if how == "average":
        mat = np.vstack([
            np.mean([subj_sent_emb[sid][sent] for sid in subj_ids], axis=0)
            for sent in common
        ])
        if zscore:
            mat = _zscore_rows(mat)
        return common, mat, None

    # stack mode: one row per subject×sentence
    rows, labels = [], []
    for subj_idx, sid in enumerate(subj_ids):
        for sent in common:
            rows.append(subj_sent_emb[sid][sent])
            labels.append(subj_idx)
    mat = np.vstack(rows)
    if zscore:
        labels_arr = np.array(labels)
        for s in np.unique(labels_arr):
            idxs = np.where(labels_arr == s)[0]
            mat[idxs] = _zscore_rows(mat[idxs])
    return common, mat, labels


def compute_rdm(mat: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(mat)
    rdm = 1 - corr
    rdm[np.isnan(rdm)] = 0.0
    return rdm


def visualize_rdm(name: str, rdm: np.ndarray, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(rdm, cmap="viridis", vmin=0, vmax=2)
    plt.title(name)
    plt.colorbar(label="1 − r")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close()


def plot_cca(corrs: List[float], name: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.bar(np.arange(1, len(corrs) + 1), corrs)
    plt.xlabel("Canonical component")
    plt.ylabel("Correlation")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close()


def rsa_cca_pair(
    eeg_emb: np.ndarray,
    nat_emb: np.ndarray,
    labels: List[str],
    tag: str,
    figdir: str,
    cca_k: int,
):
    print(f"[DEBUG] RSA/CCA for {tag}")
    rdm_eeg = compute_rdm(eeg_emb)
    rdm_nat = compute_rdm(nat_emb)
    visualize_rdm(f"RDM_EEG_{tag}", rdm_eeg, figdir)
    visualize_rdm(f"RDM_NAT_{tag}", rdm_nat, figdir)
    idx = np.triu_indices(len(labels), 1)
    rho, pval = spearmanr(rdm_eeg[idx], rdm_nat[idx])
    cca = CCA(n_components=cca_k).fit(eeg_emb, nat_emb)
    U, V = cca.transform(eeg_emb, nat_emb)
    corrs = [float(np.corrcoef(U[:, i], V[:, i])[0, 1]) for i in range(cca_k)]
    plot_cca(corrs, f"CCA_{tag}", figdir)
    return rho, pval, corrs


def main():
    p = argparse.ArgumentParser(description="Group-level RSA / CCA for EEG embeddings")
    p.add_argument("--mode", choices=["image", "text", "both"], required=True)
    p.add_argument("--eeg_image_dir", type=str)
    p.add_argument("--eeg_text_dir", type=str)
    p.add_argument("--text_models", nargs="+", default=["bert", "roberta"])
    p.add_argument("--image_models", nargs="+", default=["vit", "clip", "resnet"])
    p.add_argument("--cca_components", type=int, default=5)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--figures_dir", type=str, default="figures")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--embed_cache_dir", type=str, default="embed_cache")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eeg_embedding_types", nargs="+", default=["cbramod", "labram"])
    p.add_argument("--subject_aggregate", choices=["average", "stack"], default="average")
    p.add_argument("--no_zscore", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--run_ensemble", action="store_true", help="include ensemble analysis")
    p.add_argument(
        "--stack_fraction",
        type=float,
        default=0.05,
        help="if using `--subject_aggregate stack`, only sample this fraction of the items (0–1)",
    )
    args = p.parse_args()

    dev = args.device
    zscore = not args.no_zscore
    results = []

    # IMAGES
    if args.mode in ("image", "both"):
        img_data = load_eeg_image_data(args.eeg_image_dir)
        n_sub = len(img_data)
    
        for et in args.eeg_embedding_types:
            # --- decide whether to sample a fraction of keys for stack mode ---
            if args.subject_aggregate == "stack" and args.stack_fraction < 1.0:
                # first get the full list of keys
                all_paths, _, _ = aggregate_eeg_image(img_data, et, "average", zscore)
                k = max(1, int(len(all_paths) * args.stack_fraction))
                sampled = list(
                    np.random.choice(all_paths, size=k, replace=False)
                )
                # now build a filtered version of img_data containing only those paths
                filt_data = {}
                for subj, arr in img_data.items():
                    mask = [p in sampled for p in arr["img_paths"]]
                    filt_data[subj] = {
                        "img_paths": [p for p, m in zip(arr["img_paths"], mask) if m],
                        f"embeds_{et}": arr[f"embeds_{et}"][mask],
                    }
                paths, eeg_mat, _ = aggregate_eeg_image(filt_data, et, "stack", zscore)
            else:
                paths, eeg_mat, _ = aggregate_eeg_image(
                    img_data, et, args.subject_aggregate, zscore
                )
    
            for m in args.image_models:
                # natural embeddings on exactly the same `paths`
                nat_base = embed_images(
                    paths, m, dev, args.cache_dir, args.batch_size, args.embed_cache_dir
                )
                if zscore:
                    nat_base = _zscore_rows(nat_base)
    
                if args.subject_aggregate == "stack":
                    # repeat each row once per subject
                    nat_emb = np.repeat(nat_base, n_sub, axis=0)
                    labels = [f"S{s}_{p}" for s in range(n_sub) for p in paths]
                else:
                    nat_emb = nat_base
                    labels = paths
    
                rho, pval, corrs = rsa_cca_pair(
                    eeg_mat,
                    nat_emb,
                    labels,
                    f"IMG_{et}_{m}_{args.subject_aggregate}",
                    args.figures_dir,
                    args.cca_components,
                )
                results.append(
                    dict(
                        analysis="image",
                        eeg_type=et,
                        model=m,
                        rsa_rho=rho,
                        rsa_p=pval,
                        cca_corrs=corrs,
                    )
                )
    
        if args.run_ensemble:
            # same sampling logic for ensemble:
            # if args.subject_aggregate == "stack" and args.stack_fraction < 1.0:
            #     # reuse `sampled` from above
            #     filt_data = {
            #         subj: {
            #             "img_paths": data["img_paths"],
            #             f"embeds_{et}": data[f"embeds_{et}"],
            #         }
            #         for subj, data in filt_data.items()
            #     }
            #     mat1 = aggregate_eeg_image(
            #         filt_data, args.eeg_embedding_types[0], "stack", zscore
            #     )[1]
            #     mat2 = aggregate_eeg_image(
            #         filt_data, args.eeg_embedding_types[1], "stack", zscore
            #     )[1]
            # else:
            #     mat1 = aggregate_eeg_image(
            #         img_data, args.eeg_embedding_types[0], args.subject_aggregate, zscore
            #     )[1]
            #     mat2 = aggregate_eeg_image(
            #         img_data, args.eeg_embedding_types[1], args.subject_aggregate, zscore
            #     )[1]
            if args.subject_aggregate == "stack" and args.stack_fraction < 1.0:
                # sample the same `sampled` keys for both embedding types
                all_keys, _, _ = aggregate_eeg_image(
                    img_data, args.eeg_embedding_types[0], "average", zscore
                )
                k = max(1, int(len(all_keys) * args.stack_fraction))
                sampled = list(np.random.choice(all_keys, size=k, replace=False))
    
                # filter each subject *for both* embedding types
                filt_data = {}
                for subj, arr in img_data.items():
                    mask = [p in sampled for p in arr["img_paths"]]
                    filt_data[subj] = {
                        "img_paths":      [p for p,m in zip(arr["img_paths"],mask) if m],
                        "embeds_cbramod": arr["embeds_cbramod"][mask],
                        "embeds_labram":  arr["embeds_labram"][ mask ],
                    }
                mat1 = aggregate_eeg_image(filt_data, "cbramod", "stack", zscore)[1]
                mat2 = aggregate_eeg_image(filt_data, "labram",  "stack", zscore)[1]
            else:
                mat1 = aggregate_eeg_image(
                    img_data, args.eeg_embedding_types[0], args.subject_aggregate, zscore
                )[1]
                mat2 = aggregate_eeg_image(
                    img_data, args.eeg_embedding_types[1], args.subject_aggregate, zscore
                )[1]
    
            eeg_ens = np.concatenate([mat1, mat2], axis=1)
            for m in args.image_models:
                nat_base = embed_images(
                    paths, m, dev, args.cache_dir, args.batch_size, args.embed_cache_dir
                )
                if zscore:
                    nat_base = _zscore_rows(nat_base)
    
                if args.subject_aggregate == "stack":
                    nat_emb = np.repeat(nat_base, n_sub, axis=0)
                    labels = [f"S{s}_{p}" for s in range(n_sub) for p in paths]
                else:
                    nat_emb = nat_base
                    labels = paths
    
                rho, pval, corrs = rsa_cca_pair(
                    eeg_ens,
                    nat_emb,
                    labels,
                    f"IMG_ensemble_{m}_{args.subject_aggregate}",
                    args.figures_dir,
                    args.cca_components,
                )
                results.append(
                    dict(
                        analysis="image",
                        eeg_type="ensemble",
                        model=m,
                        rsa_rho=rho,
                        rsa_p=pval,
                        cca_corrs=corrs,
                    )
                )


    # TEXT
    if args.mode in ("text", "both"):
        txt_data = load_eeg_text_data(args.eeg_text_dir)
        n_sub = len(txt_data)
        for et in args.eeg_embedding_types:
            keys, eeg_mat, labels_idx = aggregate_eeg_text(txt_data, et, args.subject_aggregate, zscore)
            for m in args.text_models:
                nat_base = embed_texts(keys, m, dev, args.cache_dir, args.batch_size, args.embed_cache_dir)
                if zscore:
                    nat_base = _zscore_rows(nat_base)
                if args.subject_aggregate == "stack":
                    nat_emb = np.vstack([nat_base for _ in range(n_sub)])
                    labels = [f"S{s}_{keys[i]}" for s in range(n_sub) for i in range(len(keys))]
                else:
                    nat_emb = nat_base
                    labels = keys
                rho, pval, corrs = rsa_cca_pair(
                    eeg_mat, nat_emb, labels,
                    f"TXT_{et}_{m}_{args.subject_aggregate}", args.figures_dir, args.cca_components
                )
                results.append(dict(analysis="text", eeg_type=et, model=m, rsa_rho=rho, rsa_p=pval, cca_corrs=corrs))
        if args.run_ensemble:
            mat1 = aggregate_eeg_text(txt_data, args.eeg_embedding_types[0], args.subject_aggregate, zscore)[1]
            mat2 = aggregate_eeg_text(txt_data, args.eeg_embedding_types[1], args.subject_aggregate, zscore)[1]
            eeg_ens = np.concatenate([mat1, mat2], axis=1)
            for m in args.text_models:
                nat_base = embed_texts(keys, m, dev, args.cache_dir, args.batch_size, args.embed_cache_dir)
                if zscore:
                    nat_base = _zscore_rows(nat_base)
                if args.subject_aggregate == "stack":
                    nat_emb = np.vstack([nat_base for _ in range(n_sub)])
                    labels = [f"S{s}_{keys[i]}" for s in range(n_sub) for i in range(len(keys))]
                else:
                    nat_emb = nat_base
                    labels = keys
                rho, pval, corrs = rsa_cca_pair(
                    eeg_ens, nat_emb, labels,
                    f"TXT_ensemble_{m}_{args.subject_aggregate}", args.figures_dir, args.cca_components
                )
                results.append(dict(analysis="text", eeg_type="ensemble", model=m, rsa_rho=rho, rsa_p=pval, cca_corrs=corrs))

    # OUTPUT
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"[DEBUG] Saved results to {args.output_csv}")


if __name__ == '__main__':
    main()
