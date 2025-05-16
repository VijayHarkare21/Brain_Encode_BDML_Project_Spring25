#!/usr/bin/env python3
"""
Run four alignment tests between EEG embeddings and natural embeddings
(image or text): canonical‑spectrum mass, cross‑modal retrieval,
orthogonal Procrustes error, and centered CKA.  Handles missing‑trial
filtering in text by intersecting only sentences seen by *all* subjects,
and correctly aligns “stack” mode for both EEG and natural embeddings.
Optionally subsamples in stack mode and optionally runs an ensemble
(cbramod+labram).  Saves results to CSV.

Updated to support multiple text embedding models and ensemble text analysis.
"""

import argparse
import os
import sys
import hashlib
import logging
import time
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForZeroShotImageClassification,
    ViTModel,
    ViTImageProcessor,
)

# ─── logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s %(asctime)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─── utilities ───────────────────────────────────────────────────────────

def _zscore_rows(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(0, keepdims=True)
    std = mat.std(0, ddof=1, keepdims=True) + 1e-9
    return (mat - mu) / std

def load_eeg_data(directory: str, mode: str):
    data = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith('.npy'):
            continue
        if mode == 'image' and 'test' in fn.lower():
            continue
        arr = np.load(os.path.join(directory, fn), allow_pickle=True).item()
        if mode == 'text':
            # arr is { subject_id: [ trial_dict, … ] }
            subject_id, trials = next(iter(arr.items()))
            data[subject_id] = trials
        else:
            # image mode: arr already is a flat dict of numpy arrays
            key = os.path.splitext(fn)[0]
            data[key] = arr
    return data

# Updated embed_texts to support multiple models via text_models loop

def embed_texts(
    texts: List[str],
    model_name: str,
    device: str,
    model_cache_dir: str,
    embeds_cache_dir: str,
    batch_size: int
) -> np.ndarray:
    os.makedirs(embeds_cache_dir, exist_ok=True)
    keystr = "\n".join(texts)
    h = hashlib.md5(keystr.encode()).hexdigest()
    outpath = os.path.join(
        embeds_cache_dir,
        f"text_{model_name.replace('/', '_')}_{h}.npz"
    )
    if os.path.exists(outpath):
        logger.debug(f"Loading cached text embs from {outpath}")
        return np.load(outpath)['embs']

    logger.debug(f"Computing text embs: model={model_name}, n={len(texts)}")
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=model_cache_dir)
    model = model.to(device).eval()

    embs = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = [t for t in texts[i:i+batch_size] if t is not None]
        inp = tok(batch, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            last = model(**inp).last_hidden_state[:,0,:].cpu().numpy()
        embs.append(last)
    embs = np.vstack(embs).astype(np.float32)
    np.savez(outpath, embs=embs)
    logger.info(f" Text embs shape={embs.shape} computed in {time.time()-t0:.1f}s")
    return embs

def embed_images(
    paths: List[str],
    model_name: str,
    device: str,
    model_cache_dir: str,
    embeds_cache_dir: str,
    batch_size: int
) -> np.ndarray:
    """
    paths → [n_images] → ViT or CLIP
    caches embeddings in embeds_cache_dir, models in model_cache_dir
    """
    os.makedirs(embeds_cache_dir, exist_ok=True)
    keystr = "\n".join(paths)
    h = hashlib.md5(keystr.encode()).hexdigest()
    outpath = os.path.join(embeds_cache_dir, f"image_{model_name}_{h}.npz")
    if os.path.exists(outpath):
        logger.debug(f"Loading cached image embs from {outpath}")
        return np.load(outpath)['embs']

    logger.debug(f"Computing image embs: model={model_name}, n={len(paths)}")
    if model_name.lower() == 'vit':
        model = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k', cache_dir=model_cache_dir
        ).to(device).eval()
        proc  = ViTImageProcessor.from_pretrained(
            'google/vit-base-patch16-224-in21k', cache_dir=model_cache_dir
        )
    else:
        model = AutoModelForZeroShotImageClassification.from_pretrained(
            'openai/clip-vit-base-patch32', cache_dir=model_cache_dir
        ).to(device).eval()
        proc  = AutoProcessor.from_pretrained(
            'openai/clip-vit-base-patch32', cache_dir=model_cache_dir
        )

    embs = []
    t0 = time.time()
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert('RGB') for p in batch]
        inp  = proc(images=imgs, return_tensors='pt').to(device)
        with torch.no_grad():
            feats = (
                model(**inp).last_hidden_state[:,0,:]
                if model_name.lower()=='vit'
                else model.get_image_features(**inp)
            )
        embs.append(feats.cpu().numpy())
    embs = np.vstack(embs).astype(np.float32)
    np.savez(outpath, embs=embs)
    logger.info(f" Image embs shape={embs.shape} computed in {time.time()-t0:.1f}s")
    return embs

def aggregate_eeg_image(
    data: Dict[str, dict],
    eeg_type: str,
    how: str,
    zscore: bool
) -> Tuple[List[str], np.ndarray]:
    logger.debug(f"EEG‑IMG agg: type={eeg_type} how={how} zscore={zscore}")
    by_path = defaultdict(list)
    key = f"embeds_{eeg_type}"
    for subj, arr in data.items():
        if "img_paths" not in arr or key not in arr:
            continue
        for p, e in zip(arr["img_paths"], arr[key]):
            by_path[p].append(e)
    if not by_path:
        raise ValueError(f"No EEG image embeddings for '{eeg_type}'")
    keys = sorted(by_path)
    if how=='average':
        mat = np.vstack([ np.mean(by_path[k], axis=0) for k in keys ])
        if zscore: mat = _zscore_rows(mat)
    else:  # "stack"
        rows = [ by_path[k][i] for i in range(len(data)) for k in keys ]
        mat  = np.vstack(rows)
        if zscore:
            per = len(keys)
            for s in range(len(data)):
                mat[s*per:(s+1)*per] = _zscore_rows(mat[s*per:(s+1)*per])
    logger.info(f" → EEG‑IMG matrix shape {mat.shape}")
    return keys, mat.astype(np.float32)

def aggregate_eeg_text(
    data: Dict[str, list],
    eeg_type: str,
    how: str,
    zscore: bool
) -> Tuple[List[str], np.ndarray]:
    logger.debug(f"EEG‑TXT agg: type={eeg_type} how={how} zscore={zscore}")
    subs = sorted(data.keys())
    seen = [ {t['content'] for t in data[s] if t and f"embeds_{eeg_type}" in t} for s in subs ]
    common = sorted(set.intersection(*seen))
    if not common:
        raise ValueError("No common sentences across all subjects")
    # gather per‑subject
    subj_emb = { s:{} for s in subs }
    for s in subs:
        by_sent = defaultdict(list)
        for t in data[s]:
            if t and f"embeds_{eeg_type}" in t and t['content'] in common:
                by_sent[t['content']].append(t[f"embeds_{eeg_type}"])
        for sent in common:
            subj_emb[s][sent] = np.mean(by_sent[sent], axis=0)
    # assemble
    if how=='average':
        mat = np.vstack([
            np.mean([subj_emb[s][sent] for s in subs], axis=0)
            for sent in common
        ])
        if zscore: mat = _zscore_rows(mat)
    else:
        rows = []
        for s in subs:
            for sent in common:
                rows.append(subj_emb[s][sent])
        mat = np.vstack(rows)
        if zscore:
            per = len(common)
            for s in range(len(subs)):
                mat[s*per:(s+1)*per] = _zscore_rows(mat[s*per:(s+1)*per])
    logger.info(f" → EEG‑TXT matrix shape {mat.shape}")
    return common, mat.astype(np.float32)

def knn_accuracy(
    A: np.ndarray,
    B: np.ndarray,
    batch_size: int = 1024
) -> float:
    """
    Compute 1‑NN accuracy of A→B in batches (so A,B both shape [n,k]).
    """
    logger.debug(f"1-NN: A={A.shape}, B={B.shape}")
    n = A.shape[0]
    correct = 0
    for i in range(0, n, batch_size):
        j = min(i+batch_size, n)
        sims = cosine_similarity(A[i:j], B)
        correct += (sims.argmax(axis=1) == np.arange(i, j)).sum()
    acc = correct / n
    logger.info(f" → 1-NN acc={acc:.4f}")
    return acc

def rsa_cca_pair(
    X: np.ndarray,
    Y: np.ndarray,
    labels: List[str],
    tag: str,
    figdir: str,
    cca_k: int,
    max_rdm: int
) -> Tuple[float, float, List[float], CCA]:
    """
    RSA on up to max_rdm items, then CCA on full X,Y.
    Returns spearman rho/p, list of cca corrs, and the fitted CCA.
    Also writes RDM and CCA bar plots to figdir.
    """
    import matplotlib.pyplot as plt

    logger.debug(f"RSA+CCA {tag}: X={X.shape}, Y={Y.shape}")
    os.makedirs(figdir, exist_ok=True)

    n = len(labels)
    idx = np.random.choice(n, size=min(n, max_rdm), replace=False)
    def _rdm(M):
        R = np.corrcoef(M)
        D = 1 - R
        D[np.isnan(D)] = 0
        return D

    re = _rdm(X[idx])
    rn = _rdm(Y[idx])

    plt.imshow(re); plt.title(f"EEG {tag}") ; plt.colorbar()
    plt.savefig(f"{figdir}/RDM_EEG_{tag}.png"); plt.clf()
    plt.imshow(rn); plt.title(f"NAT {tag}"); plt.colorbar()
    plt.savefig(f"{figdir}/RDM_NAT_{tag}.png"); plt.clf()

    tri = np.triu_indices(len(idx), 1)
    rho, p = spearmanr(re[tri], rn[tri])
    logger.info(f"RSA Spearman rho={rho:.4f}, p={p:.2e}")

    cca = CCA(n_components=cca_k).fit(X, Y)
    U, V = cca.transform(X, Y)  # both shape [n, cca_k]
    corrs = [ float(np.corrcoef(U[:,i], V[:,i])[0,1]) for i in range(cca_k) ]

    plt.bar(range(1, cca_k+1), corrs)
    plt.title(f"CCA {tag}")
    plt.savefig(f"{figdir}/CCA_{tag}.png")
    plt.clf()
    logger.info(f"CCA corr per component = {corrs}")

    return rho, p, corrs, cca

def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    logger.debug(f"CKA: A={A.shape}, B={B.shape}")
    A_c = A - A.mean(0, keepdims=True)
    B_c = B - B.mean(0, keepdims=True)
    HSIC = np.linalg.norm(A_c.T @ B_c, 'fro')**2
    denom = np.linalg.norm(A_c.T @ A_c, 'fro') * np.linalg.norm(B_c.T @ B_c, 'fro')
    val = float(HSIC/denom)
    logger.info(f" → CKA={val:.4f}")
    return val

# ─── main suite ──────────────────────────────────────────────────────────

def test_suite(args):
    logger.info(f"TEST SUITE START (seed={args.seed})")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = []

    # ─ IMAGE ────────────────────────────────────────────────────────────────
    if args.mode in ('image','both'):
        img_data = load_eeg_data(args.eeg_image_dir, 'image')
        n_sub    = len(img_data)

        # pre‑sample once if stacking + subsample
        sampled = None
        if args.subject_aggregate=='stack' and args.stack_fraction<1.0:
            keys_all, _ = aggregate_eeg_image(
                img_data, args.eeg_embedding_types[0],
                'average', not args.no_zscore
            )
            k = max(1, int(len(keys_all)*args.stack_fraction))
            sampled = list(np.random.choice(keys_all, size=k, replace=False))
            logger.debug(f"Pre-sampled {k}/{len(keys_all)} for stack")

        for et in args.eeg_embedding_types:
            logger.info(f"IMAGE mode: EEG-type={et}")
            # build EEG matrix
            if sampled is not None:
                filt = {}
                for s, arr in img_data.items():
                    mask = [p in sampled for p in arr['img_paths']]
                    entry = {'img_paths': [p for p,m in zip(arr['img_paths'],mask) if m]}
                    for e2 in args.eeg_embedding_types:
                        entry[f"embeds_{e2}"] = arr[f"embeds_{e2}"][mask]
                    filt[s] = entry
                keys, eeg_mat = aggregate_eeg_image(
                    filt, et, 'stack', not args.no_zscore
                )
            else:
                keys, eeg_mat = aggregate_eeg_image(
                    img_data, et, args.subject_aggregate, not args.no_zscore
                )

            for imodel in args.image_models:
                logger.info(f" IMAGE model={imodel}")
                nat_base = embed_images(
                    keys, imodel, args.device,
                    args.cache_dir, args.embeds_cache_dir,
                    args.batch_size
                )
                if not args.no_zscore:
                    nat_base = _zscore_rows(nat_base).astype(np.float32)

                # expand for stack vs avg
                nat_mat = (
                    np.repeat(nat_base, n_sub, axis=0)
                    if args.subject_aggregate=='stack'
                    else nat_base
                )

                labels = (
                    [f"S{s}_{k}" for s in range(n_sub) for k in keys]
                    if args.subject_aggregate=='stack'
                    else keys
                )

                # RSA + CCA
                rho, p, corrs, cca = rsa_cca_pair(
                    eeg_mat, nat_mat, labels,
                    f"IMG_{et}_{imodel}_{args.subject_aggregate}",
                    args.figures_dir, args.cca_components,
                    args.max_items
                )

                # use CCA subspace for retrieval + Procrustes
                U, V = cca.transform(eeg_mat, nat_mat)  # both [n, cca_k]

                acc = knn_accuracy(U, V)
                idxs = np.arange(U.shape[0])
                tr, te = train_test_split(idxs, test_size=args.test_size, random_state=args.seed)
                R, _ = orthogonal_procrustes(V[tr], U[tr])
                pred = V[te] @ R
                mse  = mean_squared_error(U[te], pred)
                logger.info(f" Prsc MSE={mse:.4f}")

                cka = linear_cka(eeg_mat, nat_mat)

                results.append({
                    'analysis':'image',
                    'eeg_type':et,
                    'model':imodel,
                    'spectrum_mass':",".join(f"{x:.4f}" for x in np.cumsum(np.array(corrs)**2)),
                    'nn_accuracy':acc,
                    'procrustes_mse':mse,
                    'cka':cka
                })

        # ensemble
        if args.run_ensemble:
            logger.info("IMAGE ensemble: combining EEG types")
            if sampled is not None:
                m1 = aggregate_eeg_image(filt, args.eeg_embedding_types[0], 'stack', not args.no_zscore)[1]
                m2 = aggregate_eeg_image(filt, args.eeg_embedding_types[1], 'stack', not args.no_zscore)[1]
                ens_keys = sampled
            else:
                m1 = aggregate_eeg_image(img_data, args.eeg_embedding_types[0], args.subject_aggregate, not args.no_zscore)[1]
                m2 = aggregate_eeg_image(img_data, args.eeg_embedding_types[1], args.subject_aggregate, not args.no_zscore)[1]
                ens_keys = aggregate_eeg_image(img_data, args.eeg_embedding_types[0], 'average', not args.no_zscore)[0]
            eeg_ens = np.concatenate([m1, m2], axis=1)
            logger.info(f" → EEG ensemble shape {eeg_ens.shape}")

            for imodel in args.image_models:
                logger.info(f" ENSEMBLE IMAGE model={imodel}")
                nat_base = embed_images(
                    ens_keys, imodel, args.device,
                    args.cache_dir, args.embeds_cache_dir,
                    args.batch_size
                )
                if not args.no_zscore:
                    nat_base = _zscore_rows(nat_base).astype(np.float32)
                nat_mat = (
                    np.repeat(nat_base, n_sub, axis=0)
                    if args.subject_aggregate=='stack'
                    else nat_base
                )

                rho, p, corrs, cca = rsa_cca_pair(
                    eeg_ens, nat_mat, ens_keys,
                    f"IMG_ensemble_{imodel}_{args.subject_aggregate}",
                    args.figures_dir, args.cca_components,
                    args.max_items
                )

                U, V = cca.transform(eeg_ens, nat_mat)
                acc = knn_accuracy(U, V)
                tr, te = train_test_split(np.arange(U.shape[0]), test_size=args.test_size, random_state=args.seed)
                R, _   = orthogonal_procrustes(V[tr], U[tr])
                pred   = V[te] @ R
                mse    = mean_squared_error(U[te], pred)
                logger.info(f" Ensm Prsc MSE={mse:.4f}")
                cka    = linear_cka(eeg_ens, nat_mat)

                results.append({
                    'analysis':'image','eeg_type':'ensemble','model':imodel,
                    'spectrum_mass':",".join(f"{x:.4f}" for x in np.cumsum(np.array(corrs)**2)),
                    'nn_accuracy':acc,'procrustes_mse':mse,'cka':cka
                })

    # ─ TEXT ─────────────────────────────────────────────────────────────────
    if args.mode in ('text','both'):
        txt_data = load_eeg_data(args.eeg_text_dir, 'text')
        n_sub    = len(txt_data)

        # loop over each EEG embed type
        for et in args.eeg_embedding_types:
            logger.info(f"TEXT mode: EEG-type={et}")
            keys, eeg_mat = aggregate_eeg_text(
                txt_data, et, args.subject_aggregate, not args.no_zscore
            )

            # loop over multiple text models
            for tmodel in args.text_models:
                logger.info(f" TEXT model={tmodel}")
                nat_base = embed_texts(
                    keys, tmodel, args.device,
                    args.cache_dir, args.embeds_cache_dir,
                    args.batch_size
                )
                if not args.no_zscore:
                    nat_base = _zscore_rows(nat_base).astype(np.float32)

                # expand for stack vs avg
                if args.subject_aggregate == 'stack':
                    nat_mat = np.repeat(nat_base, n_sub, axis=0)
                    labels = [f"S{s}_{k}" for s in range(n_sub) for k in keys]
                else:
                    nat_mat = nat_base
                    labels = keys

                # RSA + CCA
                rho, p, corrs, cca = rsa_cca_pair(
                    eeg_mat, nat_mat, labels,
                    f"TXT_{et}_{tmodel}_{args.subject_aggregate}",
                    args.figures_dir, args.cca_components,
                    args.max_items
                )

                U, V = cca.transform(eeg_mat, nat_mat)
                acc = knn_accuracy(U, V)
                tr, te = train_test_split(
                    np.arange(U.shape[0]),
                    test_size=args.test_size,
                    random_state=args.seed
                )
                R, _   = orthogonal_procrustes(V[tr], U[tr])
                pred   = V[te] @ R
                mse    = mean_squared_error(U[te], pred)
                cka    = linear_cka(eeg_mat, nat_mat)

                results.append({
                    'analysis':'text',
                    'eeg_type':et,
                    'model':tmodel,
                    'spectrum_mass':','.join(f"{x:.4f}" for x in np.cumsum(np.array(corrs)**2)),
                    'nn_accuracy':acc,
                    'procrustes_mse':mse,
                    'cka':cka
                })

        # ensemble text: combine cbramod+labram
        if args.run_ensemble:
            logger.info("TEXT ensemble: combining EEG types")
            m1 = aggregate_eeg_text(
                txt_data, args.eeg_embedding_types[0], args.subject_aggregate, not args.no_zscore
            )[1]
            m2 = aggregate_eeg_text(
                txt_data, args.eeg_embedding_types[1], args.subject_aggregate, not args.no_zscore
            )[1]
            eeg_ens = np.concatenate([m1, m2], axis=1)

            for tmodel in args.text_models:
                logger.info(f" TXT ensemble model={tmodel}")
                nat_base = embed_texts(
                    keys, tmodel, args.device,
                    args.cache_dir, args.embeds_cache_dir,
                    args.batch_size
                )
                if not args.no_zscore:
                    nat_base = _zscore_rows(nat_base).astype(np.float32)
                nat_mat = np.repeat(nat_base, n_sub, axis=0) if args.subject_aggregate=='stack' else nat_base
                labels = [f"S{s}_{k}" for s in range(n_sub) for k in keys] if args.subject_aggregate=='stack' else keys

                rho, p, corrs, cca = rsa_cca_pair(
                    eeg_ens, nat_mat, labels,
                    f"TXT_ensemble_{tmodel}_{args.subject_aggregate}",
                    args.figures_dir, args.cca_components,
                    args.max_items
                )

                U, V = cca.transform(eeg_ens, nat_mat)
                acc = knn_accuracy(U, V)
                tr, te = train_test_split(np.arange(U.shape[0]), test_size=args.test_size, random_state=args.seed)
                R, _ = orthogonal_procrustes(V[tr], U[tr])
                pred = V[te] @ R
                mse = mean_squared_error(U[te], pred)
                cka = linear_cka(eeg_ens, nat_mat)

                results.append({
                    'analysis':'text','eeg_type':'ensemble','model':tmodel,
                    'spectrum_mass':','.join(f"{x:.4f}" for x in np.cumsum(np.array(corrs)**2)),
                    'nn_accuracy':acc,'procrustes_mse':mse,'cka':cka
                })

    # ─── wrap up ────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Done: wrote {len(results)} rows to {args.output_csv}")


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",              choices=["image","text","both"], required=True)
    p.add_argument("--eeg_image_dir",     required=True)
    p.add_argument("--eeg_text_dir",      required=True)
    p.add_argument("--image_models",      nargs="+", default=["vit","clip"])
    p.add_argument("--text_models",       nargs="+", default=["bert-base-uncased","roberta-base"])
    p.add_argument("--eeg_embedding_types", nargs="+", default=["cbramod","labram"])
    p.add_argument("--subject_aggregate", choices=["average","stack"],
                   default="average")
    p.add_argument("--no_zscore",         action="store_true")
    p.add_argument("--batch_size",        type=int, default=32)
    p.add_argument("--cache_dir",         default="hf_cache",
                   help="where to cache HuggingFace model weights")
    p.add_argument("--embeds_cache_dir",  default="embed_cache",
                   help="where to cache your computed embeddings")
    p.add_argument("--cca_components",    type=int, default=5)
    p.add_argument("--test_size",         type=float, default=0.2)
    p.add_argument("--stack_fraction",    type=float, default=0.05,
                   help="if stacking, sample this fraction of items")
    p.add_argument("--run_ensemble",      action="store_true",
                   help="combine cbramod+labram embeddings")
    p.add_argument("--figures_dir",       default="figs")
    p.add_argument("--output_csv",        required=True)
    p.add_argument("--device",            default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--max_items",         type=int, default=5000)
    args = p.parse_args()
    test_suite(args)
