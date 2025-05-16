#!/usr/bin/env python3
"""
Group-level Representational Similarity Analysis (RSA) and Canonical
Correlation Analysis (CCA) for EEG embeddings versus image- or text-
embedding spaces, with caching, batching, and a train/test CCA regression
pipeline evaluation.

Usage
=====
python rsa_cca_group_analysis.py \
    --mode both \
    --eeg_image_dir path/to/eeg_images \
    --eeg_text_dir  path/to/eeg_texts \
    --output_csv   results.csv \
    --figures_dir  figs \
    --subject_aggregate average          # or stack \
    --batch_size 32                       # batch size for embeddings \
    --embed_cache_dir embed_cache         # caching directory \
    --test_size 0.2                       # fraction for test split
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
            print(f"[DEBUG]   Skipping test file: {fname}")
            continue
        key = os.path.splitext(fname)[0]
        print(f"[DEBUG]   Loading train file: {fname}")
        data[key] = np.load(os.path.join(directory, fname), allow_pickle=True).item()
    print(f"[DEBUG] Loaded {len(data)} EEG image subjects (train only)")
    return data


def load_eeg_text_data(directory: str) -> Dict[str, List[dict]]:
    print(f"[DEBUG] Loading EEG text data from: {directory}")
    data = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.npy'): continue
        print(f"[DEBUG]   Found file: {fname}")
        arr = np.load(os.path.join(directory, fname), allow_pickle=True).item()
        subject = next(iter(arr.keys()))
        data[subject] = arr[subject]
    print(f"[DEBUG] Loaded {len(data)} EEG text subjects")
    return data


def embed_texts(
    texts: List[str], model_name: str, device: str='cpu',
    cache_dir: Optional[str]=None, batch_size: int=32,
    embed_cache_dir: str='embed_cache'
) -> np.ndarray:
    os.makedirs(embed_cache_dir, exist_ok=True)
    hash_ = hashlib.md5('\n'.join(texts).encode()).hexdigest()
    cache_path = os.path.join(embed_cache_dir, f"text_{model_name}_{hash_}.npz")
    if os.path.exists(cache_path):
        print(f"[DEBUG] Loading cached text embeddings: {cache_path}")
        return np.load(cache_path)['embs']
    from transformers import AutoTokenizer, AutoModel
    pretrained_map = {'bert':'bert-base-uncased','roberta':'roberta-base'}
    pretrained = pretrained_map.get(model_name.lower())
    tok = AutoTokenizer.from_pretrained(pretrained, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(pretrained, cache_dir=cache_dir).to(device).eval()
    all_embs = []
    num_batches = (len(texts)+batch_size-1)//batch_size
    with torch.no_grad():
        for b in range(num_batches):
            bs, be = b*batch_size, min((b+1)*batch_size, len(texts))
            batch = texts[bs:be]
            inputs = tok(batch, return_tensors='pt', truncation=True, padding=True).to(device)
            cls = model(**inputs).last_hidden_state[:,0]
            all_embs.append(cls.cpu().numpy())
            print(f"[DEBUG]   Text batch {b+1}/{num_batches} -> {bs}-{be}")
    embs = np.vstack(all_embs)
    np.savez(cache_path, embs=embs)
    print(f"[DEBUG] Saved text embeddings: {cache_path}")
    return embs


def embed_images(
    paths: List[str], model_name: str, device: str='cpu',
    cache_dir: Optional[str]=None, batch_size: int=32,
    embed_cache_dir: str='embed_cache'
) -> np.ndarray:
    os.makedirs(embed_cache_dir, exist_ok=True)
    hash_ = hashlib.md5('\n'.join(paths).encode()).hexdigest()
    cache_path = os.path.join(embed_cache_dir, f"image_{model_name}_{hash_}.npz")
    if os.path.exists(cache_path):
        print(f"[DEBUG] Loading cached image embeddings: {cache_path}")
        return np.load(cache_path)['embs']
    from PIL import Image
    if model_name.lower()=='vit':
        from transformers import ViTModel, ViTImageProcessor
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', cache_dir=cache_dir).to(device).eval()
        proc = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', cache_dir=cache_dir)
    else:
        from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
        model = AutoModelForZeroShotImageClassification.from_pretrained('openai/clip-vit-base-patch32', cache_dir=cache_dir).to(device).eval()
        proc = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir=cache_dir)
    all_embs=[]
    num_batches=(len(paths)+batch_size-1)//batch_size
    with torch.no_grad():
        for b in range(num_batches):
            bs,be = b*batch_size, min((b+1)*batch_size, len(paths))
            imgs = [Image.open(p).convert('RGB') for p in paths[bs:be]]
            inputs=proc(images=imgs,return_tensors='pt').to(device)
            if model_name.lower()=='vit': vec=model(**inputs).last_hidden_state[:,0,:]
            else: vec=model.get_image_features(**inputs)
            all_embs.append(vec.cpu().numpy())
            print(f"[DEBUG]   Image batch {b+1}/{num_batches} -> {bs}-{be}")
    embs=np.vstack(all_embs)
    np.savez(cache_path, embs=embs)
    print(f"[DEBUG] Saved image embeddings: {cache_path}")
    return embs


def _aggregate(by_key: defaultdict, how: str, zscore: bool):
    keys=sorted(by_key)
    if how=='average':
        mat=np.vstack([np.mean(by_key[k],0) for k in keys])
        if zscore: mat=_zscore_rows(mat)
        return keys,mat,None
    rows,subs=[],[]
    for k in keys:
        for i,e in enumerate(by_key[k]): rows.append(e); subs.append(i)
    mat=np.vstack(rows)
    if zscore:
        subs=np.array(subs)
        for s in np.unique(subs): mat[subs==s]=_zscore_rows(mat[subs==s])
    return keys,mat,subs

aggregate_eeg_image= lambda d,t,how,z: _aggregate(
    defaultdict(list, {p:vals for arr in d.values() for p,vals in [(pp,ee) for pp,ee in zip(arr['img_paths'], arr[f'embeds_{t}'])]}), how, z)
aggregate_eeg_text= lambda d,t,how,z: _aggregate(
    defaultdict(list, {s:[tr[f'embeds_{t}'] for tr in trials if f'embeds_{t}' in tr] for s,trials in d.items()}), how, z)

compute_rdm=lambda m: np.nan_to_num(1-np.corrcoef(m))


def visualize_rdm(name,rdm,outdir):
    os.makedirs(outdir,exist_ok=True)
    plt.figure(figsize=(4.2,4.2)); plt.imshow(rdm,cmap='viridis',vmin=0,vmax=2);
    plt.title(name); plt.colorbar(label='1-r'); plt.tight_layout(); plt.savefig(f"{outdir}/{name}.png"); plt.close()


def plot_cca(corrs,name,outdir):
    os.makedirs(outdir,exist_ok=True)
    plt.figure(figsize=(4,3)); plt.bar(np.arange(1,len(corrs)+1),corrs);
    plt.xlabel('Component'); plt.ylabel('Correlation'); plt.title(name);
    plt.tight_layout(); plt.savefig(f"{outdir}/{name}.png"); plt.close()


def rsa_cca_pair(eeg_emb,nat_emb,labels,tag,figdir,cca_k):
    rdm_eeg,rdm_nat=compute_rdm(eeg_emb),compute_rdm(nat_emb)
    visualize_rdm(f'RDM_EEG_{tag}',rdm_eeg,figdir)
    visualize_rdm(f'RDM_NAT_{tag}',rdm_nat,figdir)
    idx=np.triu_indices(len(labels),1)
    rho,p= spearmanr(rdm_eeg[idx],rdm_nat[idx])
    cca=CCA(n_components=cca_k).fit(eeg_emb,nat_emb)
    U,V=cca.transform(eeg_emb,nat_emb)
    corrs=[np.corrcoef(U[:,i],V[:,i])[0,1] for i in range(cca_k)]
    plot_cca(corrs,f'CCA_{tag}',figdir)
    return float(rho),float(p),corrs, cca


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['image','text','both'],required=True)
    p.add_argument('--eeg_image_dir',type=str)
    p.add_argument('--eeg_text_dir',type=str)
    p.add_argument('--text_models',nargs='+',default=['bert','roberta'])
    p.add_argument('--image_models',nargs='+',default=['vit','clip'])
    p.add_argument('--cca_components',type=int,default=5)
    p.add_argument('--output_csv',required=True)
    p.add_argument('--figures_dir',default='figs')
    p.add_argument('--cache_dir',default=None)
    p.add_argument('--embed_cache_dir',default='embed_cache')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--eeg_embedding_types',nargs='+',default=['cbramod','labram'])
    p.add_argument('--subject_aggregate',choices=['average','stack'],default='average')
    p.add_argument('--no_zscore',action='store_true')
    p.add_argument('--batch_size',type=int,default=32)
    p.add_argument('--test_size',type=float,default=0.2)
    args=p.parse_args()
    dev=args.device; zscore=not args.no_zscore
    results=[]

    if args.mode in ('image','both'):
        img_data=load_eeg_image_data(args.eeg_image_dir)
        n_sub=len(img_data)
        for et in args.eeg_embedding_types:
            paths,eeg_mat,_=aggregate_eeg_image(img_data,et,args.subject_aggregate,zscore)
            for m in args.image_models:
                nat_base=embed_images(paths,m,dev,args.cache_dir,args.batch_size,args.embed_cache_dir)
                if zscore: nat_base=_zscore_rows(nat_base)
                nat_emb=(np.repeat(nat_base,n_sub,axis=0) if args.subject_aggregate=='stack' else nat_base)
                tag=f'IMG_{et}_{m}_{args.subject_aggregate}'
                labels=paths if args.subject_aggregate=='average' else [f'S{s}_{p}' for s in range(n_sub) for p in paths]
                rho,pval,corrs, cca_model = rsa_cca_pair(eeg_mat,nat_emb,labels,tag,args.figures_dir,args.cca_components)
                # train/test pipeline
                idxs=np.arange(len(labels))
                tr,te = train_test_split(idxs,test_size=args.test_size,random_state=42)
                # compute mapping W: nat->eeg
                W = cca_model.y_weights_.dot(cca_model.x_weights_.T)
                eeg_pred = nat_emb[te].dot(W)
                mse = mean_squared_error(eeg_mat[te], eeg_pred)
                r2 = r2_score(eeg_mat[te], eeg_pred, multioutput='uniform_average')
                # RSA on test
                rdm_pred = compute_rdm(eeg_pred)
                rdm_true = compute_rdm(eeg_mat[te])
                rsa_test,_ = spearmanr(rdm_pred[np.triu_indices(len(te),1)], rdm_true[np.triu_indices(len(te),1)])
                print(f"[PIPE] {tag} test MSE={mse:.4f}, R2={r2:.4f}, RSA={rsa_test:.4f}")
                results.append({**dict(analysis='image', eeg_type=et, model=m, aggregate=args.subject_aggregate, rsa_rho=rho, rsa_p=pval, cca_corrs=corrs), 'test_mse':mse, 'test_r2':r2, 'test_rsa':rsa_test})

    if args.mode in ('text','both'):
        txt_data=load_eeg_text_data(args.eeg_text_dir)
        n_sub=len(txt_data)
        for et in args.eeg_embedding_types:
            sents,eeg_mat,_=aggregate_eeg_text(txt_data,et,args.subject_aggregate,zscore)
            for m in args.text_models:
                nat_base=embed_texts(sents,m,dev,args.cache_dir,args.batch_size,args.embed_cache_dir)
                if zscore: nat_base=_zscore_rows(nat_base)
                nat_emb=(np.repeat(nat_base,n_sub,axis=0) if args.subject_aggregate=='stack' else nat_base)
                tag=f'TXT_{et}_{m}_{args.subject_aggregate}'
                labels=sents if args.subject_aggregate=='average' else [f'S{s}_{t}' for s in range(n_sub) for t in sents]
                rho,pval,corrs, cca_model = rsa_cca_pair(eeg_mat,nat_emb,labels,tag,args.figures_dir,args.cca_components)
                idxs=np.arange(len(labels))
                tr,te = train_test_split(idxs,test_size=args.test_size,random_state=42)
                W = cca_model.y_weights_.dot(cca_model.x_weights_.T)
                eeg_pred = nat_emb[te].dot(W)
                mse = mean_squared_error(eeg_mat[te], eeg_pred)
                r2 = r2_score(eeg_mat[te], eeg_pred, multioutput='uniform_average')
                rdm_pred = compute_rdm(eeg_pred)
                rdm_true = compute_rdm(eeg_mat[te])
                rsa_test,_ = spearmanr(rdm_pred[np.triu_indices(len(te),1)], rdm_true[np.triu_indices(len(te),1)])
                print(f"[PIPE] {tag} test MSE={mse:.4f}, R2={r2:.4f}, RSA={rsa_test:.4f}")
                results.append({**dict(analysis='text', eeg_type=et, model=m, aggregate=args.subject_aggregate, rsa_rho=rho, rsa_p=pval, cca_corrs=corrs), 'test_mse':mse, 'test_r2':r2, 'test_rsa':rsa_test})

    df=pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"[DEBUG] Saved results with pipeline to {args.output_csv}")

if __name__=='__main__':
    main()
