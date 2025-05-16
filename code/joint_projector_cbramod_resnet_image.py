#!/usr/bin/env python3
"""
fine_tune_cbramod_image.py

Fine‑tune a CBraMod model (pre‑trained on images) on image‑EEG trials,
but pad/truncate each trial to the *actual* max # of 1‑s segments in your data,
rather than always 30.
"""
import os, math, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
import argparse

from cbramod import CBraMod
from pre_process_eeg_cbramod import process_eeg
from tqdm import trange, tqdm
import gc
from torchvision import models, transforms
from PIL import Image
import pandas as pd            #  ← at top of file
import random
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)     # stop on first NaN/Inf
DEBUG_NAN = True                            # turn off once training is stable

# ----------------------------------------------------------------------
# Replace original load_eeg_image_data with process_partition + loader
# ----------------------------------------------------------------------
def process_partition(eeg_fp, img_meta, partition_name, img_data_root):
    """
    eeg_fp: path to .npy for one partition
    img_meta: loaded image_metadata dict
    partition_name: 'training' or 'test'
    img_data_root: root dir containing 'training_images' & 'test_images'
    """
    data = np.load(eeg_fp, allow_pickle=True).item()
    eeg_data = data['preprocessed_eeg_data']  # (n_conditions, n_reps, 17, 200)

    # average over EEG repetitions (axis=1)
    eeg_mean = eeg_data.mean(axis=1)         # (n_conditions, 17, 200)

    # select corresponding image metadata
    if partition_name == 'training':
        img_concepts = img_meta['train_img_concepts']
        img_files    = img_meta['train_img_files']
        img_folder   = 'training_images'
    else:
        img_concepts = img_meta['test_img_concepts']
        img_files    = img_meta['test_img_files']
        img_folder   = 'test_images'

    n_conditions = eeg_mean.shape[0]
    assert n_conditions == len(img_concepts) == len(img_files), \
        f"Mismatch in #conditions vs metadata for {partition_name}"

    # build arrays for this subj+partition
    # replicate EEG mean as single "repetition" so shape matches downstream
    eeg_arr  = eeg_mean[:, None, :, :]     # (n_conditions, 1, 17, 200)
    return {
        'preprocessed_eeg_data': eeg_arr,
        'train_img_concepts':    list(img_concepts),
        'train_img_files':       list(img_files)
    }

def load_eeg_image_data(eeg_data_root: str, img_meta: dict, img_data_root: str):
    """
    Walk each subject folder under eeg_data_root, run process_partition
    on both training & test .npy, and collect into a dict.
    """
    print(f"[DEBUG] Loading EEG image data from: {eeg_data_root}")
    data = {}
    for subj in sorted(os.listdir(eeg_data_root)):
        subj_dir = os.path.join(eeg_data_root, subj)
        if not os.path.isdir(subj_dir):
            continue
        for partition in ['preprocessed_eeg_training.npy']:
            eeg_fp = os.path.join(subj_dir, partition)
            if not os.path.exists(eeg_fp):
                continue
            part_name = 'training' if 'training' in partition else 'test'
            print(f"[DEBUG]   Loading {subj} {part_name}")
            dd = process_partition(eeg_fp, img_meta, part_name, img_data_root)
            data[f"{subj}_{part_name}"] = dd
    return data

def load_image_labels(metadata_path: str, things_map_path: str):
    """Load image labels from metadata file."""
    print(f"[INFO] Loading image labels from {metadata_path}")
    print(f"[INFO] Loading high-level image labels from {things_map_path}")
    meta = np.load(metadata_path, allow_pickle=True).item()
    things_map = pd.read_csv(things_map_path, delimiter="\t")
    files = meta['train_img_files']
    concepts = meta['train_img_concepts']
    things_concepts = meta['train_img_concepts_THINGS']
    
    # Create a mapping from full path to concept label
    path_to_label = {}
    for things_concept, concept, fname in zip(things_concepts, concepts, files):
        # print(things_concept.split("_")[0])
        # print(things_map.iloc[int(things_concept.split("_")[0]) + 1])
        row = things_map.iloc[int(things_concept.split("_")[0]) - 1]
        high_concept = str(things_map.columns[row == 1][0]) if not (row == 0).all() else 'miscellaneous'
        path_key = os.path.join(concept, fname)
        path_to_label[path_key] = high_concept
        
    return path_to_label

# 1) Define 4 groups of EEG channels for pooling (17 channels total)
CHANNEL_GROUPS = [
    [0,1,2,3],
    [4,5,6,7],
    [8,9,10,11],
    [12,13,14,15,16]
]

# 2) Helper: compute the dataset's true max_segments (≤30)
def compute_max_segments(data, orig_sfreq, hard_cap=30):
    max_segs = 0
    for key, dd in data.items():
        arr = dd.get('preprocessed_eeg_data', None)
        if arr is None or arr.size == 0: continue
        # arr shape [#images × reps × channels × time]
        T = arr.shape[-1]
        segs = math.ceil(T / orig_sfreq)
        max_segs = max(max_segs, segs)
    return min(max_segs, hard_cap)

# ---------------------------------------------------------------
# JointDataset  —  ResNet50 features × EEG pairs with image labels
# ---------------------------------------------------------------
class JointDataset(Dataset):
    """
    For every (image × repetition) trial it returns
        img_feat : torch.FloatTensor [2048]
        eeg_proc : torch.FloatTensor [17, P, 200]
        lab_id   : int   (0 … #classes-1)
    If avg_subjects=True it averages the EEG across all subjects
    that saw the same image, so each image appears exactly
    once in the dataset.
    """
    def __init__(self,
                 data_dicts:    dict,
                 cnn,
                 metadata:      dict,          # path_key -> category string
                 cat2idx:       dict,          # category -> int
                 img_parent_dir:str,
                 *,
                 orig_sfreq:    int,
                 max_segments:  int,
                 avg_subjects:  bool=False,
                 batch_size:    int=1024):       # Batch size for feature extraction
        self.samples = []                  # final list of (feat, eeg, lab_id)
        self.samples_by_class   = defaultdict(list)
        self.class_counts       = Counter()
        feat_cache   = {}                  # path_key -> feature tensor
        eeg_bucket   = {}                  # path_key -> list[eeg slices]
        max_T = orig_sfreq * max_segments
        cnn.eval()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Collect all unique images to process in batches
        unique_images = {}  # path_key -> (img_path, category_id)
        
        for subj_key, dd in data_dicts.items():
            arr = dd.get('preprocessed_eeg_data')
            files = dd.get('train_img_files')
            concepts = dd.get('train_img_concepts')
            if arr is None or files is None or concepts is None:
                continue
            N_images, reps, C, T = arr.shape
            for i in range(N_images):
                path_key = os.path.join(concepts[i], files[i])
                if path_key not in metadata:  # skip unlabeled
                    continue
                cat = metadata[path_key]
                lab_id = cat2idx[cat]
                img_path = os.path.join(img_parent_dir, concepts[i], files[i])
                unique_images[path_key] = (img_path, lab_id)
        
        # Process images in batches
        with torch.no_grad():
            batch_inputs = []
            batch_paths = []
            device = next(cnn.parameters()).device
            
            # Function to process a batch of images
            def process_batch():
                if not batch_inputs:
                    return
                
                print(f"Processing batch of {len(batch_inputs)} images...")
                batch_tensor = torch.stack(batch_inputs).to(device)
                batch_features = cnn(batch_tensor).cpu()
                
                for i, path_key in enumerate(batch_paths):
                    feat_cache[path_key] = batch_features[i]
                
                batch_inputs.clear()
                batch_paths.clear()
            
            # Prepare all images in batches
            for path_key, (img_path, _) in unique_images.items():
                if path_key in feat_cache:  # Skip if already processed
                    continue
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    inp = preprocess(img)
                    batch_inputs.append(inp)
                    batch_paths.append(path_key)
                    
                    # Process when batch is full
                    if len(batch_inputs) >= batch_size:
                        process_batch()
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            # Process remaining images
            process_batch()
        
        # Now build the dataset using the cached features
        for subj_key, dd in data_dicts.items():
            arr = dd.get('preprocessed_eeg_data')
            files = dd.get('train_img_files')
            concepts = dd.get('train_img_concepts')
            if arr is None or files is None or concepts is None:
                continue
            N_images, reps, C, T = arr.shape
            for i in range(N_images):
                for r in range(reps):
                    raw = arr[i, r, :, :]   # [channels × time]
                    sl = raw[:, :max_T]
                    if sl.shape[1] < max_T:
                        pad = np.zeros((C, max_T - sl.shape[1]))
                        sl  = np.hstack([sl, pad])
                    path_key = os.path.join(concepts[i], files[i])
                    if path_key not in metadata:  # skip unlabeled
                        continue
                    cat = metadata[path_key]
                    lab_id = cat2idx[cat]
                    
                    if avg_subjects:
                        eeg_bucket.setdefault(path_key, []).append(sl)
                    else:
                        idx = len(self.samples)
                        self.samples.append(
                            (feat_cache[path_key], sl, lab_id)
                        )
                        self.samples_by_class[lab_id].append(idx)
                        self.class_counts[lab_id] += 1
                        
        # -------- subject-average option --------
        if avg_subjects:
            for path_key, lst in eeg_bucket.items():
                img_feat = feat_cache[path_key]
                eeg_avg  = np.mean(lst, axis=0)
                lab_id   = cat2idx[metadata[path_key]]
                idx      = len(self.samples)
                self.samples.append((img_feat, eeg_avg, lab_id))
                self.samples_by_class[lab_id].append(idx)
                self.class_counts[lab_id] += 1
                
        print(f"[INFO] JointDataset: {len(self.samples)} samples, "
              f"{len(cat2idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, arr, lab_id = self.samples[idx]
        proc   = process_eeg(arr.T, 500)            # ➜ [1,17,P,200]
        x_eeg  = torch.from_numpy(proc).float().squeeze(0)
        return feat.float(), x_eeg, lab_id


# -------- supervised InfoNCE on embeddings -----------------

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, oversample_factor=1.0, undersample_factor=0.25, indices=None):
        """
        Parameters:
        -----------
        dataset : JointDataset
            Must have samples_by_class attribute
        batch_size : int
            Size of each batch
        oversample_factor : float
            Factor to oversample minority classes. If >1.0, minority classes will be oversampled.
        undersample_factor : float
            Factor to undersample the single majority class. If <1.0, the majority class will be reduced.
        indices : list
            List of indices to use (for using with a subset)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.undersample_factor = undersample_factor
        self.oversample_factor = oversample_factor
        
        # Get class distribution
        self.n_classes = len(dataset.samples_by_class)
        
        # If indices are provided, filter samples_by_class to only include those indices
        if indices is not None:
            # Convert indices to a set for O(1) lookup
            indices_set = set(indices)
            # Filter samples by class to only include indices in the subset
            self.samples_by_class = {
                cls: [idx for idx in indices_list if idx in indices_set]
                for cls, indices_list in dataset.samples_by_class.items()
            }
        else:
            self.samples_by_class = dataset.samples_by_class
        
        # Calculate target samples per class
        min_class_size = min(len(samples) for samples in self.samples_by_class.values())
        
        # Keep track of original class sizes for undersampling calculation
        class_sizes = {cls: len(samples) for cls, samples in self.samples_by_class.items()}
        
        # If we're oversampling, use the majority class size as basis
        if oversample_factor > 1.0:
            max_class_size = max(len(samples) for samples in self.samples_by_class.values())
            # Oversample classes to reach a target size between min and max
            self.target_samples_per_class = {
                cls: min(
                    int(len(samples) * oversample_factor),  # Oversample by factor
                    max(
                        len(samples),                       # Never undersample
                        int(min_class_size * oversample_factor)  # Target for minority classes
                    )
                )
                for cls, samples in self.samples_by_class.items()
            }
        else:
            # If not oversampling, use a consistent number of samples per class
            samples_per_class = max(1, batch_size // self.n_classes)
            self.target_samples_per_class = {
                cls: len(samples)  # Use original class sizes
                for cls, samples in self.samples_by_class.items()
            }
        
        # Apply undersampling to the majority class if requested
        if self.undersample_factor < 1.0:
            # Pick the class with the largest original size
            majority_cls = max(class_sizes, key=class_sizes.get)
            original = class_sizes[majority_cls]
            reduced = max(1, int(original * self.undersample_factor))
            # Never exceed what we had already targeted via oversampling
            self.target_samples_per_class[majority_cls] = min(
                self.target_samples_per_class[majority_cls],
                reduced
        )
            
        # Calculate length: how many complete batches can we make
        total_samples = sum(self.target_samples_per_class.values())
        self.length = total_samples // batch_size
        
        print(f"[INFO] BalancedBatchSampler initialized with {self.n_classes} classes")
        print(f"[INFO] Original class distribution: {[len(samples) for samples in self.samples_by_class.values()]}")
        print(f"[INFO] Target samples per class: {self.target_samples_per_class}")
        print(f"[INFO] Will generate {self.length} balanced batches of size {batch_size}")
    
    def __iter__(self):
        """Yield balanced batches of indices"""
        # Create a pool of samples for each class with replacement if needed
        sample_pools = {}
        for cls, indices in self.samples_by_class.items():
            target_size = self.target_samples_per_class[cls]
            if target_size <= len(indices):
                # If we don't need to oversample, just shuffle the original indices
                pool = random.sample(indices, target_size)
            else:
                # Oversample with replacement
                pool = random.choices(indices, k=target_size)
            random.shuffle(pool)  # Ensure randomness within class
            sample_pools[cls] = pool
        
        # Flatten indices
        all_indices = []
        # Use the actual class keys instead of range(self.n_classes)
        for cls in self.samples_by_class.keys():
            all_indices.extend(sample_pools.get(cls, []))
        
        # Shuffle to mix classes while maintaining overall balance
        random.shuffle(all_indices)
        
        # Yield batches
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:  # Only yield complete batches
                yield batch
    
    def __len__(self):
        return self.length


# ---------------------------------------------------------------------#
# 4 · Numerically-stable InfoNCE with NaN tracker
# ---------------------------------------------------------------------#
def info_nce(z, y, temperature=0.1, *, debug=DEBUG_NAN):
    def _check(t, tag, *, ignore=None):
        if not debug: return
        bad = torch.isnan(t) | torch.isinf(t)
        if ignore is not None:
            bad &= ~ignore
        if bad.any():
            print(f"\n[NaN-tracker] {tag} -> {t[bad][:5].tolist()} ...")
            raise RuntimeError(f"NaN/Inf in {tag}")

    z = F.normalize(z, dim=1, eps=1e-6)            # B,D
    _check(z, "z-norm")

    B        = z.size(0)
    self_m   = torch.eye(B, dtype=torch.bool, device=z.device)
    pos_m    = (y[:, None] == y[None, :]) & ~self_m
    keep_row = pos_m.sum(1) > 0
    if not keep_row.any():
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    z, y, pos_m = z[keep_row], y[keep_row], pos_m[keep_row][:, keep_row]
    self_m      = torch.eye(z.size(0), dtype=torch.bool, device=z.device)

    sim = torch.matmul(z, z.T) / temperature
    _check(sim, "raw-sim")
    sim.masked_fill_(self_m, float('-inf'))
    _check(sim, "sim-masked", ignore=self_m)

    row_max,_ = sim.max(1, keepdim=True)
    logits    = sim - row_max
    _check(logits, "logits-shifted", ignore=self_m)

    exp_logits = torch.exp(logits)
    denom      = exp_logits.sum(1, keepdim=True).clamp_min(1e-8)
    log_p      = logits - torch.log(denom)
    _check(log_p, "log-prob", ignore=self_m)

    log_p = log_p.masked_fill(self_m, 0.0)
    mean_pos = (pos_m * log_p).sum(1) / pos_m.sum(1)
    _check(mean_pos, "mean-pos")

    return -mean_pos.mean()



# ------------------------------------------------------------------------
# 4) Model: loads only the backbone from your image‑fine‑tuned checkpoint
# ------------------------------------------------------------------------
class CBraModTextClassifier(nn.Module):
    """
    ϕ_backbone  → ϕ_embedder (returns z: [B,200]) → g_classifier (logits)
    We contrast on  z  and classify on  logits.
    """
    def __init__(self, param):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12, nhead=8
        )
        self.backbone.proj_out = nn.Identity()

        flat_dim   = param.channels * param.patches * param.d_model
        hidden_dim = param.patches  * param.d_model
        
        self.classifier = nn.Sequential(
            nn.Linear(len(CHANNEL_GROUPS)*param.d_model, 512),
            nn.ELU(), nn.Dropout(param.dropout),
            nn.Linear(512, param.num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)   # [B, C, P, D]
        region_vecs = []
        for idxs in CHANNEL_GROUPS:
            sub = feats[:, idxs, :, :]       # [B, |group|, P, D]
            rv = sub.mean(dim=(1,2))         # [B, D]
            region_vecs.append(rv)
        z = torch.cat(region_vecs, dim=1)    # [B, groups·D]
        logits = self.classifier(z)
        return z, logits


class Text2EEG(nn.Module):
    def __init__(self, in_dim: int = 2048, out_dim: int = 800):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        return self.net(x)


class JointModel(nn.Module):
    def __init__(self, eeg_encoder: CBraModTextClassifier, projector: Text2EEG):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.projector   = projector
    def forward(self, x_eeg, x_img):
        z_eeg, logits = self.eeg_encoder(x_eeg)
        z_txt         = self.projector(x_img)
        return z_eeg, z_txt, logits


# ------------------------------------------------------------------
# 3 · Training / evaluation helpers
# ------------------------------------------------------------------
CE_W, NCE_W, MSE_W, COS_W = 0.2, 1.0, 1.0, 0.5

def run_epoch(model, loader, opt, sched, device, ce_loss):
    model.train()
    tot, correct, N = 0.0, 0, 0
    step = 0
    for img, eeg, lab in tqdm(loader, leave=False, desc="Train", dynamic_ncols=True):
        step += 1
        img, eeg, lab = img.to(device), eeg.to(device), lab.to(device)

        opt.zero_grad()
        z_eeg, z_img, logits = model(eeg, img)

        # ----- component losses ----------------------------------
        eps = 1e-6
        z_eeg = z_eeg + eps
        z_img = z_img + eps
        cos   = F.cosine_similarity(z_eeg, z_img).mean()
        ce    = ce_loss(logits, lab)
        nce   = info_nce(z_eeg, lab)
        mse   = F.mse_loss(z_eeg, z_img)
        loss  = CE_W * ce + NCE_W * nce + MSE_W * mse + COS_W * (1 - cos)

        # print every mini-batch (like text script)
        print(f"ce={ce.item():.4f}  nce={nce.item():.4f}  "
              f"mse={mse.item():.4f}  cos={cos.item():.4f}")

        # NaN guards ------------------------------------------------
        if torch.isnan(loss):
            raise RuntimeError("NaN loss – see component values above")
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                raise RuntimeError(f"Gradient NaN in {n}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        tot     += loss.item() * lab.size(0)
        correct += (logits.argmax(1) == lab).sum().item()
        N       += lab.size(0)

        # progress heartbeat every 5 steps
        if step % 5 == 0:
            print(f"\nstep {step}  loss {loss.item():.4f}\n")

    return tot / len(loader.dataset), correct / N


@torch.no_grad()
def evaluate(model, loader, device, criterion_ce):
    model.eval(); ce_tot, correct, Z, Y = 0,0,[],[]
    steps = 0
    for img_feat, eeg_raw, lab in tqdm(loader, total=len(loader),desc=f"Eval", leave=False, dynamic_ncols=True):
        steps += 1
        img_feat, eeg_raw, lab = img_feat.to(device), eeg_raw.to(device), lab.to(device)
        z_eeg, z_img, logits = model(eeg_raw, img_feat)
        ce_tot += criterion_ce(logits, lab).item()*len(lab)
        correct+= (logits.argmax(1)==lab).sum().item()
        Z.append(z_eeg.cpu()); Y.append(lab.cpu())
    Z = torch.cat(Z); Y = torch.cat(Y)
    knn = knn_metrics(Z, Y, k=5)
    return ce_tot/len(loader.dataset), correct/len(loader.dataset), knn['1nn'], knn['recall@5']
    
from sklearn.neighbors import NearestNeighbors

@torch.no_grad()
def knn_metrics(embeddings, labels, k=5, metric='cosine'):
    z = torch.nn.functional.normalize(embeddings, dim=1).cpu().numpy()
    y = labels.cpu().numpy()
    nnm = NearestNeighbors(n_neighbors=k+1, metric=metric)
    nnm.fit(z)
    dist, idx = nnm.kneighbors(z)
    idx = idx[:, 1:]
    neigh_labels = y[idx]
    acc_1nn = (neigh_labels[:, 0] == y).mean()
    hits = (neigh_labels == y[:, None]).any(axis=1)
    recall_k = hits.mean()
    return {'1nn': acc_1nn, f'recall@{k}': recall_k}

from collections import defaultdict
from torch.utils.data import SubsetRandomSampler

# def balanced_sampler_from_subset(subset, repeat: int = 4):
#     buckets = defaultdict(list)
#     for local_idx, global_idx in enumerate(subset.indices):
#         lab_id = subset.dataset.samples[global_idx][2]
#         buckets[lab_id].append(local_idx)
#     rng  = np.random.default_rng()
#     flat = []
#     for _ in range(repeat):
#         for idxs in buckets.values():
#             flat.extend(rng.permutation(idxs))
#     return SubsetRandomSampler(flat)

def main(args):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # ------------ image metadata & raw EEG --------------
    img_meta = np.load(args.metadata, allow_pickle=True).item()
    raw_dict = load_eeg_image_data(args.eeg_dir, img_meta, args.img_parent_dir)
    max_seg  = compute_max_segments(raw_dict, args.orig_sfreq)

    path_to_label = load_image_labels(args.metadata, args.things_map)
    uniq_cats   = sorted(set(path_to_label.values()))
    cat2idx     = {c:i for i,c in enumerate(uniq_cats)}
    num_classes = len(uniq_cats)
    print(f"[INFO] Detected {num_classes} image classes:", uniq_cats)

    # ------ image feature extractor -------
    resnet = models.resnet50(pretrained=True).eval().to(device)
    resnet.fc = nn.Identity()

    dummy = argparse.Namespace(channels=17, patches=max_seg,
                               d_model=200, dim_feedforward=800,
                               n_layer=12, nhead=8,
                               dropout=args.dropout,
                               num_classes=num_classes)
    eeg_enc = CBraModTextClassifier(dummy)

    # load pretrained backbone weights
    state = torch.load(args.eeg_ckpt, map_location=device)
    bk = {k:v for k,v in state.items() if not k.startswith("proj_out.")}
    eeg_enc.backbone.load_state_dict(bk)
    print("Model loaded and ready ...")
    
    print("=== All CBraMod backbone parameter names ===")
    for name, p in eeg_enc.backbone.named_parameters():
        print(name)

    # unfreeze last 6 layers
    for n,p in eeg_enc.backbone.named_parameters():
        p.requires_grad = any(n.startswith(f"encoder.layers.{i}") for i in (6,7,8,9,10,11))
    total, trainable = 0, 0
    for n,p in eeg_enc.backbone.named_parameters():
        total += p.numel()
        if p.requires_grad: 
            trainable += p.numel()
            print("Unfrozen:", n)
    print(f"Backbone params: {trainable:,}/{total:,} trainable")  
    
    for p in eeg_enc.classifier.parameters():
        p.requires_grad = True

    model = JointModel(eeg_enc, Text2EEG()).to(device)
    
    print("=== Projector parameters ===")
    total_proj, trainable_proj = 0, 0
    for name, p in model.projector.named_parameters():
        n = p.numel()
        total_proj += n
        if p.requires_grad:
            trainable_proj += n
        print(f"{name} | shape={tuple(p.shape)} | trainable={p.requires_grad}")
    
    print(f"\nProjector: {trainable_proj:,}/{total_proj:,} parameters are trainable")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    dataset = JointDataset(raw_dict, resnet, path_to_label, cat2idx,
                           args.img_parent_dir,
                           orig_sfreq=args.orig_sfreq,
                           max_segments=max_seg,
                           avg_subjects=args.avg_subjects)
    tr_len  = int(0.9*len(dataset))
    tr_ds, va_ds = random_split(dataset, [tr_len, len(dataset)-tr_len])
    sampler = BalancedBatchSampler(dataset, args.batch, 12.0, indices=tr_ds.indices)
    tr_dl = DataLoader(dataset, batch_sampler=sampler)
    va_dl = DataLoader(va_ds, batch_size=args.batch)

    opt  = optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                       lr=args.lr, weight_decay=1e-2)
    print("Optimizer knows about", sum(p.numel() for p in opt.param_groups[0]['params']), "params")
    print(" vs. model has", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable params")

    total_steps  = len(tr_dl) * args.epochs
    warmup_steps = int(total_steps * 0.03)
    sched = optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            optim.lr_scheduler.LinearLR(opt, start_factor=1e-2, total_iters=warmup_steps),
            optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_steps - warmup_steps, eta_min=1e-6
            )
        ],
        milestones=[warmup_steps]
    )
    ce = nn.CrossEntropyLoss()

    metrics_path = os.path.join(args.model_dir, "metrics_cbramod_image_cpu_new.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "tr_loss", "va_loss", "va_acc", "va_1nn", "va_r5"])

    best1 = 0
    for ep in trange(1, args.epochs+1, desc="Epoch"):
        tr_loss, tr_acc = run_epoch(model, tr_dl, opt, sched, device, ce)
        va_loss, va_acc, va_1nn, va_r5 = evaluate(model, va_dl, device, ce)
        print(f"Ep{ep:02d} | CE_tr {tr_loss:.4f} | CE_va {va_loss:.4f} "
              f"| CE_acc {va_acc:.2%} | 1-NN {va_1nn:.2%} | R@5 {va_r5:.2%}")
        tqdm.write(
            f"Ep{ep:02d} | CE_tr {tr_loss:.4f} | CE_va {va_loss:.4f} "
            f"| CE_acc {va_acc:.2%} | 1-NN {va_1nn:.2%} | R@5 {va_r5:.2%}"
        )
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, tr_loss, va_loss, va_acc, va_1nn, va_r5])
        if va_1nn > best1:
            best1 = va_1nn
            torch.save(model.state_dict(), args.out)
            tqdm.write("saved best model")

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--eeg_dir",        required=True,
                   help="root dir of sub-*/preprocessed_eeg_*.npy files")
    P.add_argument("--metadata",       required=True,
                   help=".npy metadata file with train/test img arrays")
    P.add_argument("--things_map",     required=True,
                   help="TSV mapping THINGS concepts to high-level labels")
    P.add_argument("--img_parent_dir", required=True,
                   help="parent directory for training_images/<concept>/<file> and test_images")
    P.add_argument("--eeg_ckpt",       required=True)
    P.add_argument("--out",            required=True)
    P.add_argument("--model_dir",      required=True)
    P.add_argument("--epochs",     type=int, default=20)
    P.add_argument("--batch",      type=int, default=64)
    P.add_argument("--lr",         type=float, default=3e-4)
    P.add_argument("--orig_sfreq", type=int, default=500)
    P.add_argument("--dropout",    type=float, default=0.2)
    P.add_argument("--cuda",       type=int, default=0)
    P.add_argument("--avg_subjects", action="store_true",
                   help="average EEG embeddings across subjects")
    main(P.parse_args())
