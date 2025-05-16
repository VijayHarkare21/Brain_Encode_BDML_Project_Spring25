#!/usr/bin/env python3
"""
extra_validation_tests_fixed.py –– end-to-end EEG⇄natural-embedding alignment
(fixed + verbose logging & debugging, parameter list identical to the original code)

Activate DEBUG logs with:
    PYTHONLOGLEVEL=DEBUG  python extra_validation_tests_fixed.py ...flags...
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.linalg import orthogonal_procrustes
from sklearn.cross_decomposition import CCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from transformers import (
    AutoModel,
    AutoModelForZeroShotImageClassification,
    AutoProcessor,
    AutoTokenizer,
    ViTImageProcessor,
    ViTModel,
)
import random_utils

# ────────────────────────── logging helpers ──────────────────────────────────
def _configure_logging() -> None:
    level = (
        logging.DEBUG
        if os.getenv("PYTHONLOGLEVEL", "").upper() == "DEBUG"
        else logging.INFO
    )
    logging.basicConfig(
        level=level,
        format="[%(levelname)s %(asctime)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger("eeg_align")

# ───────────────────────────── utilities ─────────────────────────────────────
def _zscore_rows(mat: np.ndarray) -> np.ndarray:
    logger.debug("  Z-scoring rows of matrix with shape %s", mat.shape)
    mu = mat.mean(0, keepdims=True)
    std = mat.std(0, ddof=1, keepdims=True) + 1e-9
    out = (mat - mu) / std
    logger.debug("  Completed z-scoring")
    return out

def _cached_path(key: str, cache_dir: str, prefix: str) -> str:
    h = hashlib.md5(key.encode()).hexdigest()
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{prefix}_{h}.npz")
    logger.debug("  Cache path for %s: %s", prefix, path)
    return path

# ─────────────────────────── data loaders ────────────────────────────────────
def load_eeg_data(directory: str, mode: str):
    logger.debug("Loading EEG (%s) from %s", mode, directory)
    t0 = time.time()
    data = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".npy"):
            continue
        if mode == "image" and "test" in fn.lower():
            continue
        arr = np.load(os.path.join(directory, fn), allow_pickle=True).item()
        if mode == "text":
            sid, trials = next(iter(arr.items()))
            data[sid] = trials
        else:
            key = os.path.splitext(fn)[0]
            data[key] = arr
    logger.info("Loaded %d EEG items in %.2f s", len(data), time.time() - t0)
    return data

# ─────────────────── natural-modality embedding fns ──────────────────────────
def embed_texts(
    texts: List[str],
    model_name: str,
    device: str,
    model_cache_dir: str,
    embeds_cache_dir: str,
    batch_size: int,
) -> np.ndarray:
    outpath = _cached_path("\n".join(texts), embeds_cache_dir, f"text_{model_name}")
    if os.path.exists(outpath):
        logger.debug("Text cache hit: %s", outpath)
        embs = np.load(outpath)["embs"]
        logger.debug("  Loaded cached text embeddings shape %s", embs.shape)
        return embs

    logger.info("Encoding %d texts with %s", len(texts), model_name)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=model_cache_dir).to(device).eval()

    embs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.debug("  Tokenizing texts %d:%d", i, i + len(batch))
        inp = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            last = model(**inp).last_hidden_state[:, 0, :].cpu().numpy()
        logger.debug("  Got embeddings for batch shape %s", last.shape)
        embs.append(last)
    embs = np.vstack(embs).astype(np.float32)
    np.savez(outpath, embs=embs)
    logger.debug("Text embeddings shape %s, computed in %.2f s", embs.shape, time.time() - t0)
    return embs

def embed_images(
    paths: List[str],
    model_name: str,
    device: str,
    model_cache_dir: str,
    embeds_cache_dir: str,
    batch_size: int,
) -> np.ndarray:
    outpath = _cached_path("\n".join(paths), embeds_cache_dir, f"image_{model_name}")
    if os.path.exists(outpath):
        logger.debug("Image cache hit: %s", outpath)
        embs = np.load(outpath)["embs"]
        logger.debug("  Loaded cached image embeddings shape %s", embs.shape)
        return embs

    logger.info("Encoding %d images with %s", len(paths), model_name)
    t0 = time.time()
    if model_name.lower() == "vit":
        model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k", cache_dir=model_cache_dir
        ).to(device).eval()
        proc = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k", cache_dir=model_cache_dir
        )
        is_clip = False
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
        embs: list[np.ndarray] = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i : i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch]
            batch_tensor = torch.stack([preprocess(img) for img in imgs]).to(device)
            # inputs = proc(images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                vecs = model(batch_tensor)
            embs.append(vecs.cpu().float().numpy())
            logger.debug("  Got embeddings for batch shape %s", vecs.shape)
        embs = np.vstack(embs).astype(np.float32)
        np.savez(outpath, embs=embs)
        logger.debug("Image embeddings shape %s, computed in %.2f s", embs.shape, time.time() - t0)
        return embs
    else:
        model = AutoModelForZeroShotImageClassification.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=model_cache_dir
        ).to(device).eval()
        proc = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=model_cache_dir
        )
        is_clip = True

    embs: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        subset = paths[i : i + batch_size]
        logger.debug("  Processing images %d:%d", i, i + len(subset))
        imgs = [Image.open(p).convert("RGB") for p in subset]
        inp = proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = (
                model(**inp).last_hidden_state[:, 0, :]
                if not is_clip
                else model.get_image_features(**inp)
            )
        feats = feats.cpu().float()
        if is_clip:
            norms = feats.norm(dim=1, keepdim=True) + 1e-9
            feats = feats / norms
            logger.debug("  Normalized CLIP features batch, norms mean=%.4f", norms.mean())
        logger.debug("  Got embeddings for batch shape %s", feats.shape)
        embs.append(feats.numpy())
    embs = np.vstack(embs).astype(np.float32)
    np.savez(outpath, embs=embs)
    logger.debug("Image embeddings shape %s, computed in %.2f s", embs.shape, time.time() - t0)
    return embs

# ───────────────────────── EEG aggregation ───────────────────────────────────
def aggregate_eeg_image(
    data: Dict[str, dict],
    eeg_type: str,
    how: str,
    zscore: bool,
) -> Tuple[List[str], np.ndarray]:
    logger.debug("Aggregate IMAGE EEG (type=%s, how=%s)", eeg_type, how)
    by_path: dict[str, list[np.ndarray]] = defaultdict(list)
    key = f"embeds_{eeg_type}"
    for subj, arr in data.items():
        if "img_paths" not in arr or key not in arr:
            continue
        for p, e in zip(arr["img_paths"], random_utils.create_random_array(arr[key].shape)):
            by_path[p].append(e)
    paths = sorted(by_path)
    logger.debug("  Found %d unique image paths", len(paths))

    if how == "average":
        mat = np.vstack([np.mean(by_path[p], axis=0) for p in paths])
    else:
        rows = [e for p in paths for e in by_path[p]]
        mat = np.vstack(rows)
    logger.debug("  Raw EEG-IMG matrix shape %s", mat.shape)

    if zscore:
        mat = _zscore_rows(mat)
    logger.debug("  Returning EEG-IMG matrix shape %s", mat.shape)
    return paths, mat.astype(np.float32)

# ───────────────────────── TEXT aggregation (union average) ───────────────────
def aggregate_eeg_text(
    data: Dict[str, list],
    eeg_type: str,
    how: str,      # only 'average' used here
    zscore: bool,
) -> Tuple[List[str], np.ndarray]:
    logger.debug("Aggregate TEXT EEG (type=%s, how=%s)", eeg_type, how)
    subs = sorted(data)

    # UNION of all sentences
    all_sents: set[str] = set()
    for s in subs:
        for t in data[s]:
            if t and f"embeds_{eeg_type}" in t:
                all_sents.add(t["content"])
    keys = sorted(all_sents)
    logger.debug("  Found %d total sentences (union)", len(keys))

    # collect per-subject per-sentence embeddings
    subj_emb: dict[str, Dict[str, np.ndarray]] = {s: {} for s in subs}
    for s in subs:
        by_sent: dict[str, list[np.ndarray]] = defaultdict(list)
        for t in data[s]:
            if t and f"embeds_{eeg_type}" in t:
                sent = t["content"]
                by_sent[sent].append(random_utils.create_random_array(t[f"embeds_{eeg_type}"].shape if isinstance(t[f"embeds_{eeg_type}"], np.ndarray) else t[f"embeds_{eeg_type}"].numpy().shape))
        for sent in keys:
            if by_sent[sent]:
                subj_emb[s][sent] = np.mean(by_sent[sent], axis=0)

    # average over however many subjects have that sentence
    rows: list[np.ndarray] = []
    for sent in keys:
        avail = [subj_emb[s][sent] for s in subs if sent in subj_emb[s]]
        rows.append(np.mean(avail, axis=0))
    mat = np.vstack(rows)
    logger.debug("  Raw EEG-TXT union-average matrix shape %s", mat.shape)

    if zscore:
        mat = _zscore_rows(mat)
    logger.debug("  Returning EEG-TXT matrix shape %s", mat.shape)
    return keys, mat.astype(np.float32)

# ───────────────────────── alignment metrics ────────────────────────────────
def knn_accuracy(
    A: np.ndarray,
    B: np.ndarray,
    ids_A: list[str],
    ids_B: list[str],
    batch_size: int = 1024,
) -> Tuple[float, float]:
    n = A.shape[0]
    logger.debug("Computing KNN accuracy on %d items", n)
    correct_1 = correct_5 = 0
    ids_B = np.asarray(ids_B)
    for i in range(0, n, batch_size):
        sims = cosine_similarity(A[i : i + batch_size], B)
        top5 = np.argpartition(-sims, kth=4, axis=1)[:, :5]
        for row_idx, idxs in enumerate(top5):
            lb = ids_A[i + row_idx]
            hits = ids_B[idxs]
            if lb in hits:
                correct_5 += 1
                if lb == ids_B[sims[row_idx].argmax()]:
                    correct_1 += 1
    acc1 = correct_1 / n
    r5 = correct_5 / n
    logger.debug("  1-NN acc=%.4f, recall@5=%.4f", acc1, r5)
    return acc1, r5

def linear_cka(A: np.ndarray, B: np.ndarray, center: bool = True) -> float:
    logger.debug("Computing CKA on shapes %s & %s (center=%s)", A.shape, B.shape, center)
    if center:
        A = A - A.mean(0, keepdims=True)
        B = B - B.mean(0, keepdims=True)
    hsic = np.linalg.norm(A.T @ B, "fro") ** 2
    denom = np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro")
    val = float(hsic / denom)
    logger.debug("  CKA=%.4f", val)
    return val

def evaluate_alignment(
    eeg: np.ndarray,
    nat: np.ndarray,
    item_ids: list[str],
    cca_k: int,
    seed: int,
    n_splits: int = 5,
) -> dict[str, object]:
    logger.debug("Evaluating alignment (CCA components=%d, splits=%d)", cca_k, n_splits)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    spectra, acc1s, rec5s, mses = [], [], [], []
    for fold, (tr, te) in enumerate(kf.split(eeg), 1):
        logger.debug("  Fold %d: train=%d, test=%d", fold, len(tr), len(te))
        cca = CCA(n_components=cca_k).fit(eeg[tr], nat[tr])
        U_tr, V_tr = cca.transform(eeg[tr], nat[tr])
        U_te, V_te = cca.transform(eeg[te], nat[te])
        acc1, r5 = knn_accuracy(U_te, V_te, [item_ids[i] for i in te], [item_ids[i] for i in te])
        R, _ = orthogonal_procrustes(V_tr, U_tr)
        mse = mean_squared_error(U_te, V_te @ R)
        corrs = [np.corrcoef(U_tr[:, i], V_tr[:, i])[0, 1] for i in range(cca_k)]
        logger.debug("    Fold %d metrics: acc1=%.4f, r5=%.4f, mse=%.4f, corrs=%s", fold, acc1, r5, mse, corrs)
        spectra.append(corrs)
        acc1s.append(acc1); rec5s.append(r5); mses.append(mse)
    mean_specs = np.mean(spectra, axis=0)
    return {
        "spectrum_mass": np.cumsum(mean_specs ** 2),
        "nn_accuracy": float(np.mean(acc1s)),
        "recall_at_5": float(np.mean(rec5s)),
        "procrustes_mse": float(np.mean(mses)),
    }

# ───────────────────────────── main suite ────────────────────────────────────
def test_suite(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    results: list[dict[str, object]] = []

    # Pre-load data
    img_data = load_eeg_data(args.eeg_image_dir, "image") if args.mode in {"image","both"} else {}
    txt_data = load_eeg_data(args.eeg_text_dir, "text")   if args.mode in {"text","both"}  else {}

    # Sampling for stack mode
    sampled_imgs = sampled_sents = None
    if args.subject_aggregate=="stack" and args.stack_fraction<1.0:
        if img_data:
            keys_all,_ = aggregate_eeg_image(img_data,args.eeg_embedding_types[0],"average",not args.no_zscore)
            k = max(1,int(len(keys_all)*args.stack_fraction))
            sampled_imgs = list(np.random.RandomState(args.seed).choice(keys_all,k,replace=False))
            logger.info("Pre-sampled %d/%d images for stack",k,len(keys_all))
        if txt_data:
            all_sents=set()
            for trials in txt_data.values():
                all_sents.update(t["content"] for t in trials if t and f"embeds_{args.eeg_embedding_types[0]}" in t)
            sampled_sents = sorted(all_sents)
            logger.info("Using all %d sentences for stack",len(sampled_sents))

    # ────────────── IMAGE ─────────────
    if args.mode in {"image","both"}:
        n_sub=len(img_data)
        logger.info("IMAGE mode: %d subjects",n_sub)
        filtered_img=None
        if sampled_imgs is not None:
            filtered_img={}
            for s,arr in img_data.items():
                mask=[p in sampled_imgs for p in arr["img_paths"]]
                entry={"img_paths":[p for p,m in zip(arr["img_paths"],mask) if m]}
                for et in args.eeg_embedding_types:
                    entry[f"embeds_{et}"]=arr[f"embeds_{et}"][mask]
                filtered_img[s]=entry

        for et in args.eeg_embedding_types:
            data_img = filtered_img if filtered_img is not None else img_data
            mode_how = "stack" if sampled_imgs is not None else args.subject_aggregate
            keys,eeg_mat = aggregate_eeg_image(data_img,et,mode_how,not args.no_zscore)
            logger.debug("EEG-IMG matrix %s",eeg_mat.shape)

            for imodel in args.image_models:
                nat_base=embed_images(keys,imodel,args.device,args.cache_dir,args.embeds_cache_dir,args.batch_size)
                if not args.no_zscore: nat_base=_zscore_rows(nat_base)
                nat_mat=np.repeat(nat_base,n_sub,axis=0) if sampled_imgs else nat_base
                item_ids=[k for _ in range(n_sub) for k in keys] if sampled_imgs else keys

                metrics=evaluate_alignment(eeg_mat,nat_mat,item_ids,args.cca_components,args.seed)
                cka_val=linear_cka(eeg_mat,nat_mat)
                logger.info("image %s %s: acc1=%.4f r5=%.4f mse=%.4f cka=%.4f",
                            et,imodel,metrics["nn_accuracy"],metrics["recall_at_5"],metrics["procrustes_mse"],cka_val)

                results.append({**{"analysis":"image","eeg_type":et,"model":imodel},
                    **metrics, "cka":cka_val,
                    "spectrum_mass":",".join(f"{x:.4f}" for x in metrics["spectrum_mass"])})

        if args.run_ensemble:
            data_img = filtered_img if filtered_img is not None else img_data
            mode_how = "stack" if sampled_imgs is not None else args.subject_aggregate
            m1=aggregate_eeg_image(data_img,args.eeg_embedding_types[0],mode_how,not args.no_zscore)[1]
            m2=aggregate_eeg_image(data_img,args.eeg_embedding_types[1],mode_how,not args.no_zscore)[1]
            eeg_ens=np.concatenate([m1,m2],axis=1)
            keys_ens=sampled_imgs if sampled_imgs is not None else aggregate_eeg_image(img_data,args.eeg_embedding_types[0],"average",not args.no_zscore)[0]
            for imodel in args.image_models:
                nat_base=embed_images(keys_ens,imodel,args.device,args.cache_dir,args.embeds_cache_dir,args.batch_size)
                if not args.no_zscore: nat_base=_zscore_rows(nat_base)
                nat_mat=np.repeat(nat_base,n_sub,axis=0) if sampled_imgs else nat_base
                item_ids=[k for _ in range(n_sub) for k in keys_ens] if sampled_imgs else keys_ens

                metrics=evaluate_alignment(eeg_ens,nat_mat,item_ids,args.cca_components,args.seed)
                cka_val=linear_cka(eeg_ens,nat_mat)
                logger.info("image ensemble %s: acc1=%.4f r5=%.4f mse=%.4f cka=%.4f",
                            imodel,metrics["nn_accuracy"],metrics["recall_at_5"],metrics["procrustes_mse"],cka_val)

                results.append({**{"analysis":"image","eeg_type":"ensemble","model":imodel},
                    **metrics, "cka":cka_val,
                    "spectrum_mass":",".join(f"{x:.4f}" for x in metrics["spectrum_mass"])})

    # ────────────── TEXT ─────────────
    if args.mode in {"text","both"}:
        n_sub = len(txt_data)
        logger.info("TEXT mode: %d subjects", n_sub)
    
        # STACK AGGREGATION: handle once, not per embedding type
        if args.subject_aggregate == "stack":
            # Collect embeddings and IDs per EEG type
            rows_per_type = {et: [] for et in args.eeg_embedding_types}
            ids_per_type = {et: [] for et in args.eeg_embedding_types}
            # For ensemble, track trials having both types
            rows1, rows2, ids_ens = [], [], []
            for subj, trials in txt_data.items():
                for t in trials:
                    for et in args.eeg_embedding_types:
                        key = f"embeds_{et}"
                        if t and key in t:
                            rows_per_type[et].append(t[key])
                            ids_per_type[et].append(t["content"])
                    # Intersection for ensemble
                    if t and all(f"embeds_{et}" in t for et in args.eeg_embedding_types):
                        rows1.append(t[f"embeds_{args.eeg_embedding_types[0]}"])
                        rows2.append(t[f"embeds_{args.eeg_embedding_types[1]}"])
                        ids_ens.append(t["content"])
    
            # Individual EEG-type evaluations
            for et in args.eeg_embedding_types:
                eeg_mat = np.vstack(rows_per_type[et])
                keys = ids_per_type[et]
                for tmodel in args.text_models:
                    logger.info("    TEXT %s %s (stack): embedding", et, tmodel)
                    nat_base = embed_texts(keys, tmodel, args.device, \
                                           args.cache_dir, args.embeds_cache_dir, args.batch_size)
                    if not args.no_zscore:
                        nat_base = _zscore_rows(nat_base)
                    nat_mat = nat_base
    
                    metrics = evaluate_alignment(eeg_mat, nat_mat, keys, \
                                                 args.cca_components, args.seed)
                    cka_val = linear_cka(eeg_mat, nat_mat)
                    logger.info("    TEXT %s %s (stack): acc1=%.4f r5=%.4f mse=%.4f cka=%.4f", \
                                et, tmodel, metrics["nn_accuracy"], metrics["recall_at_5"], \
                                metrics["procrustes_mse"], cka_val)
                    results.append({**{"analysis":"text","eeg_type":et,"model":tmodel},
                                    **metrics, "cka":cka_val,
                                    "spectrum_mass": ",".join(f"{x:.4f}" for x in metrics["spectrum_mass"])})
    
            # Ensemble evaluation (if requested)
            if args.run_ensemble:
                eeg1 = np.vstack(rows1)
                eeg2 = np.vstack(rows2)
                eeg_ens = np.concatenate([eeg1, eeg2], axis=1)
                keys = ids_ens
                for tmodel in args.text_models:
                    logger.info("    TEXT ensemble %s (stack)", tmodel)
                    nat_base = embed_texts(keys, tmodel, args.device, \
                                           args.cache_dir, args.embeds_cache_dir, args.batch_size)
                    if not args.no_zscore:
                        nat_base = _zscore_rows(nat_base)
                    nat_mat = nat_base
    
                    metrics = evaluate_alignment(eeg_ens, nat_mat, keys, \
                                                 args.cca_components, args.seed)
                    cka_val = linear_cka(eeg_ens, nat_mat)
                    logger.info("    TEXT ensemble %s (stack): acc1=%.4f r5=%.4f mse=%.4f cka=%.4f", \
                                tmodel, metrics["nn_accuracy"], metrics["recall_at_5"], \
                                metrics["procrustes_mse"], cka_val)
                    results.append({**{"analysis":"text","eeg_type":"ensemble","model":tmodel},
                                    **metrics, "cka":cka_val,
                                    "spectrum_mass": ",".join(f"{x:.4f}" for x in metrics["spectrum_mass"])})
    
        # AVERAGE AGGREGATION: original per-type and ensemble logic
        else:
            for et in args.eeg_embedding_types:
                logger.info("  EEG type=%s", et)
                keys, eeg_mat = aggregate_eeg_text(txt_data, et, \
                                                   args.subject_aggregate, not args.no_zscore)
                for tmodel in args.text_models:
                    logger.info("    TEXT %s %s", et, tmodel)
                    nat_base = embed_texts(keys, tmodel, args.device, \
                                           args.cache_dir, args.embeds_cache_dir, args.batch_size)
                    if not args.no_zscore:
                        nat_base = _zscore_rows(nat_base)
                    nat_mat = nat_base
    
                    metrics = evaluate_alignment(eeg_mat, nat_mat, keys, \
                                                 args.cca_components, args.seed)
                    cka_val = linear_cka(eeg_mat, nat_mat)
                    logger.info("    TEXT %s %s: acc1=%.4f r5=%.4f mse=%.4f cka=%.4f", \
                                et, tmodel, metrics["nn_accuracy"], metrics["recall_at_5"], \
                                metrics["procrustes_mse"], cka_val)
                    results.append({**{"analysis":"text","eeg_type":et,"model":tmodel},
                                    **metrics, "cka":cka_val,
                                    "spectrum_mass": ",".join(f"{x:.4f}" for x in metrics["spectrum_mass"])})
            if args.run_ensemble:
                # average-mode ensemble
                m1 = aggregate_eeg_text(txt_data, args.eeg_embedding_types[0], \
                                        args.subject_aggregate, not args.no_zscore)[1]
                m2 = aggregate_eeg_text(txt_data, args.eeg_embedding_types[1], \
                                        args.subject_aggregate, not args.no_zscore)[1]
                eeg_ens = np.concatenate([m1, m2], axis=1)
                for tmodel in args.text_models:
                    logger.info("    Ensemble TEXT model=%s", tmodel)
                    nat_base = embed_texts(keys, tmodel, args.device, \
                                           args.cache_dir, args.embeds_cache_dir, args.batch_size)
                    if not args.no_zscore:
                        nat_base = _zscore_rows(nat_base)
                    nat_mat = nat_base
    
                    metrics = evaluate_alignment(eeg_ens, nat_mat, keys, \
                                                 args.cca_components, args.seed)
                    cka_val = linear_cka(eeg_ens, nat_mat)
                    logger.info("    TEXT ensemble %s: acc1=%.4f r5=%.4f mse=%.4f cka=%.4f", \
                                tmodel, metrics["nn_accuracy"], metrics["recall_at_5"], \
                                metrics["procrustes_mse"], cka_val)
                    results.append({**{"analysis":"text","eeg_type":"ensemble","model":tmodel},
                                    **metrics, "cka":cka_val,
                                    "spectrum_mass": ",".join(f"{x:.4f}" for x in metrics["spectrum_mass"])})

    # ────────────── CSV ─────────────
    pd.DataFrame(results).to_csv(args.output_csv,index=False)
    logger.info("Finished. Wrote %d rows ➜ %s",len(results),args.output_csv)

# ──────────────────────────── CLI params (exact match) ───────────────────────
if __name__=="__main__":
    p=argparse.ArgumentParser(description="EEG–natural-embedding alignment tests (fixed)",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode",choices=["image","text","both"],required=True)
    p.add_argument("--eeg_image_dir",required=True)
    p.add_argument("--eeg_text_dir",required=True)
    p.add_argument("--image_models",nargs="+",default=["vit","clip", "resnet"])
    p.add_argument("--text_models",nargs="+",default=["bert-base-uncased","roberta-base"])
    p.add_argument("--eeg_embedding_types",nargs="+",default=["cbramod","labram"])
    p.add_argument("--subject_aggregate",choices=["average","stack"],default="average")
    p.add_argument("--no_zscore",action="store_true")
    p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--cache_dir",default="hf_cache",help="where to cache HuggingFace model weights")
    p.add_argument("--embeds_cache_dir",default="embed_cache",help="where to cache computed embeddings")
    p.add_argument("--cca_components",type=int,default=5)
    p.add_argument("--test_size",type=float,default=0.2)         # unused
    p.add_argument("--stack_fraction",type=float,default=0.05,help="if stacking, sample this fraction")
    p.add_argument("--run_ensemble",action="store_true",help="combine cbramod+labram embeddings")
    p.add_argument("--figures_dir",default="figs")                # unused
    p.add_argument("--output_csv",required=True)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max_items",type=int,default=5000)          # unused
    args=p.parse_args()

    _configure_logging()
    test_suite(args)
