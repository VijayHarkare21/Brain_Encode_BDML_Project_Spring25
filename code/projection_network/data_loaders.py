#!/usr/bin/env python3
"""
data_loader.py  —  v3  (ensemble-aware)
=======================================

What changed?
-------------
• new argument **eeg_types**  (str | Sequence[str])
      - default: "cbramod"  (backwards compatible)
      - give a tuple/list, e.g. ("cbramod", "labram"), to *concatenate*
        multiple EEG embedding spaces.
• all Dataset/Sampler logic unchanged – you still get
      { 'nat', 'eeg', 'subj', 'key' }
  but `eeg.shape[-1]` is now  D₁ + D₂ (+ …) in ensemble mode.

Everything else (lazy mmap, subsampling, contrastive sampler)
remains identical to v2.
"""
from __future__ import annotations
import hashlib, os, random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

# ------------------------------------------------------------------ #
#                        helper utilities                            #
# ------------------------------------------------------------------ #
def _md5(lines: Sequence[str]) -> str:
    return hashlib.md5("\n".join(lines).encode()).hexdigest()

def _zscore_rows(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(1, keepdims=True)
    sd = mat.std(1, ddof=1, keepdims=True) + 1e-9
    return (mat - mu) / sd

# ------------------------ raw EEG loaders ------------------------- #
def load_eeg_image_data(directory: str) -> Dict[str, dict]:
    data = {}
    for f in sorted(os.listdir(directory)):
        if f.endswith(".npy") and "test" not in f.lower():
            data[f.split("_")[0]] = np.load(Path(directory)/f, allow_pickle=True).item()
    if not data:
        raise FileNotFoundError(f"No training .npy in {directory}")
    return data

def load_eeg_text_data(directory: str) -> Dict[str, List[dict]]:
    data = {}
    for f in sorted(os.listdir(directory)):
        if f.endswith(".npy"):
            obj = np.load(Path(directory)/f, allow_pickle=True).item()
            sid = next(iter(obj))
            data[sid] = obj[sid]
    if not data:
        raise FileNotFoundError(f"No .npy in {directory}")
    return data

# ------------------- HF embedding (identical to v2) --------------- #
def embed_texts(texts, model_name, *, cache_dir, embed_cache, batch_size, device):
    embed_cache.mkdir(parents=True, exist_ok=True)
    path = embed_cache / f"text_{model_name}_{_md5(texts)}.npz"
    if path.exists():
        return np.load(path, mmap_mode="r")["emb"]
    from transformers import AutoTokenizer, AutoModel
    pretrained = {"bert": "bert-base-uncased", "roberta": "roberta-base"}[model_name]
    tok = AutoTokenizer.from_pretrained(pretrained, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(pretrained, cache_dir=cache_dir).to(device).eval()
    chunks, out = range(0, len(texts), batch_size), []
    with torch.no_grad():
        for i in chunks:
            inp = tok(texts[i:i+batch_size], truncation=True,
                      padding=True, return_tensors="pt").to(device)
            out.append(model(**inp).last_hidden_state[:, 0].cpu().numpy())
    emb = np.vstack(out); np.savez(path, emb=emb); return emb

def embed_images(paths, model_name, *, cache_dir, embed_cache, batch_size, device):
    embed_cache.mkdir(parents=True, exist_ok=True)
    path = embed_cache / f"image_{model_name}_{_md5(paths)}.npz"
    if path.exists():
        return np.load(path, mmap_mode="r")["emb"]
    from PIL import Image
    if model_name == "vit":
        from transformers import ViTModel, ViTImageProcessor
        proc  = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=cache_dir)
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=cache_dir).to(device).eval()
        forward = lambda x: model(**x).last_hidden_state[:, 0]
    else:
        from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
        proc  = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
        model = AutoModelForZeroShotImageClassification.from_pretrained(
                    "openai/clip-vit-base-patch32", cache_dir=cache_dir).to(device).eval()
        forward = model.get_image_features
    out, rng = [], range(0, len(paths), batch_size)
    with torch.no_grad():
        for i in rng:
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch_size]]
            out.append(forward(**proc(images=imgs, return_tensors="pt").to(device)).cpu().numpy())
    emb = np.vstack(out); np.savez(path, emb=emb); return emb

# -------------- common alignment for multiple EEG types ----------- #
def _align_image(img_data, eeg_type) -> Tuple[List[str], np.ndarray, List[int]]:
    by_path = defaultdict(list)
    subjects = sorted(img_data)
    for sidx, sid in enumerate(subjects):
        arr, key = img_data[sid], f"embeds_{eeg_type}"
        for p, e in zip(arr["img_paths"], arr[key]):
            by_path[p].append((sidx, e))
    rows, subs, keys = [], [], []
    for p in sorted(by_path):
        for sidx, e in by_path[p]:
            keys.append(p); rows.append(e); subs.append(sidx)
    return keys, np.vstack(rows), subs

def _align_text(txt_data, eeg_type) -> Tuple[List[str], np.ndarray, List[int]]:
    subjects = sorted(txt_data)
    common = sorted(set.intersection(*[
        {t["content"] for t in txt_data[sid] if f"embeds_{eeg_type}" in t}
        for sid in subjects]))
    if not common:
        raise RuntimeError("No shared sentences")
    rows, subs, keys = [], [], []
    for sidx, sid in enumerate(subjects):
        lut = {t["content"]: t[f"embeds_{eeg_type}"] for t in txt_data[sid]}
        for sent in common:
            rows.append(lut[sent]); subs.append(sidx); keys.append(sent)
    return keys, np.vstack(rows), subs

# -------------------------- DATASET ------------------------------- #
class EEGProjectionDataset(Dataset):
    """
    Pass **eeg_types** as a single string or a list/tuple.
    If >1 are given the resulting `eeg` vector is a concatenation.
    """
    def __init__(
        self,
        *,
        modality: str,
        eeg_types: Union[str, Sequence[str]] = "cbramod",
        nat_model: str,
        eeg_image_dir: Optional[str] = None,
        eeg_text_dir:  Optional[str] = None,
        cache_dir:     Optional[str] = None,
        embed_cache_dir: str = "embed_cache",
        batch_size: int = 32,
        device: str = "cpu",
        mmap: bool = False,
        zscore_eeg: bool = True,
        sample_frac: float = 1.0,
        max_per_subject: Optional[int] = None,
    ):
        self.modality = modality
        eeg_types = list(eeg_types) if isinstance(eeg_types, (list, tuple)) else [eeg_types]
        embed_cache = Path(embed_cache_dir)

        # ---------- NATURAL embeddings & alignment -----------------
        if modality == "image":
            if eeg_image_dir is None:
                raise ValueError("Need eeg_image_dir for image modality.")
            img_data = load_eeg_image_data(eeg_image_dir)
            # first EEG type sets the reference ordering
            keys, eeg_mat, subj_ids = _align_image(img_data, eeg_types[0])
            mats = [eeg_mat]
            for et in eeg_types[1:]:
                k2, m2, s2 = _align_image(img_data, et)
                assert k2 == keys and s2 == subj_ids, "EEG types mis-aligned!"
                mats.append(m2)
            eeg_mat = np.concatenate(mats, axis=1)
            nat_mat = embed_images(keys, nat_model, cache_dir=cache_dir,
                                   embed_cache=embed_cache, batch_size=batch_size,
                                   device=device)
        elif modality == "text":
            if eeg_text_dir is None:
                raise ValueError("Need eeg_text_dir for text modality.")
            txt_data = load_eeg_text_data(eeg_text_dir)
            keys, eeg_mat, subj_ids = _align_text(txt_data, eeg_types[0])
            mats = [eeg_mat]
            for et in eeg_types[1:]:
                k2, m2, s2 = _align_text(txt_data, et)
                assert k2 == keys and s2 == subj_ids, "EEG types mis-aligned!"
                mats.append(m2)
            eeg_mat = np.concatenate(mats, axis=1)
            nat_mat = embed_texts(keys, nat_model, cache_dir=cache_dir,
                                  embed_cache=embed_cache, batch_size=batch_size,
                                  device=device)
        else:
            raise ValueError("modality must be 'image' or 'text'")

        # ---------- optional subsampling ---------------------------
        rng = np.random.default_rng(0)
        keep = np.ones(len(keys), bool)
        if sample_frac < 1.0:
            keep &= rng.random(len(keys)) < sample_frac
        if max_per_subject is not None:
            subj = np.array(subj_ids)
            for sid in np.unique(subj):
                idx = np.where(subj == sid)[0]
                if len(idx) > max_per_subject:
                    keep[rng.choice(idx, size=len(idx)-max_per_subject, replace=False)] = False

        self.keys  = [k for k,m in zip(keys,keep) if m]
        self.subjs = torch.tensor([s for s,m in zip(subj_ids,keep) if m], dtype=torch.long)
        eeg_mat = eeg_mat[keep]; nat_mat = nat_mat[keep]

        if zscore_eeg:
            eeg_mat = _zscore_rows(eeg_mat)
        nat_mat = _zscore_rows(nat_mat)

        array_to_tensor = (lambda x: torch.from_numpy(x).float().share_memory_()) if mmap \
                          else (lambda x: torch.as_tensor(x, dtype=torch.float32))
        self.eeg = array_to_tensor(eeg_mat)
        self.nat = array_to_tensor(nat_mat)

    # ----------------- Dataset API --------------------------------
    def __len__(self):  return len(self.nat)
    def __getitem__(self, idx):
        return {"nat": self.nat[idx],
                "eeg": self.eeg[idx],
                "subj": self.subjs[idx],
                "key": self.keys[idx]}

# ---------------- convenience wrappers & sampler ----------------- #
def build_dataset(**kw) -> EEGProjectionDataset:
    return EEGProjectionDataset(**kw)

def make_loader(dataset: Dataset, **dl_kw) -> DataLoader:
    return DataLoader(dataset, **dl_kw)

# ----------------------------------------------------------------- #
#           A subject-balanced sampler for contrastive loss         #
# ----------------------------------------------------------------- #
class ContrastiveSampler(Sampler[List[int]]):
    """
    Batches with K distinct subjects × N items each (total K·N indices).

    Parameters
    ----------
    data       : EEGProjectionDataset
    K_subjects : int
    N_per_subject : int
    shuffle    : bool
    drop_last  : bool
    """

    def __init__(
        self,
        data: EEGProjectionDataset,
        *,
        K_subjects: int,
        N_per_subject: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.data = data
        self.K = K_subjects
        self.N = N_per_subject
        self.shuffle = shuffle
        self.drop_last = drop_last

        # index lists per subject
        self.by_subj: Dict[int, List[int]] = defaultdict(list)
        for idx, sid in enumerate(data.subjs.tolist()):
            self.by_subj[sid].append(idx)
        for lst in self.by_subj.values():
            random.shuffle(lst)

        self.subject_pool = list(self.by_subj)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.subject_pool)
            for lst in self.by_subj.values():
                random.shuffle(lst)

        batch: List[int] = []
        subj_cursor = {sid: 0 for sid in self.subject_pool}

        while True:
            if len(self.subject_pool) < self.K:
                break
            chosen = random.sample(self.subject_pool, self.K)
            batch.clear()
            for sid in chosen:
                cur = subj_cursor[sid]
                avail = self.by_subj[sid]
                if cur + self.N > len(avail):
                    self.subject_pool.remove(sid)
                    break
                batch.extend(avail[cur:cur+self.N])
                subj_cursor[sid] = cur + self.N
            else:  # only executed if the for-loop didn't `break`
                yield batch.copy()
                continue
            break  # insufficient data for another full batch

        if (not self.drop_last) and batch:
            yield batch

    def __len__(self):
        # Conservative estimate
        per_subj = [len(v) // self.N for v in self.by_subj.values()]
        return sum(per_subj) // self.K

# ---------------------------- demo ------------------------------- #
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--modality", required=True, choices=["image","text"])
    p.add_argument("--eeg_types", nargs="+", default=["cbramod","labram"])
    p.add_argument("--nat_model", required=True,
                   choices=["clip","vit","bert","roberta"])
    p.add_argument("--eeg_image_dir"); p.add_argument("--eeg_text_dir")
    args = p.parse_args()

    ds = build_dataset(
            modality=args.modality,
            eeg_types=args.eeg_types,
            nat_model=args.nat_model,
            eeg_image_dir=args.eeg_image_dir,
            eeg_text_dir=args.eeg_text_dir,
            mmap=True)
    print("Rows:", len(ds), "  EEG dim:", ds[0]["eeg"].shape[-1])
