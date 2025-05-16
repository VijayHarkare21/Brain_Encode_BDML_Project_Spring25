#!/usr/bin/env python3
"""
fine_tune_cbramod_text.py

Fine‑tune a CBraMod model (pre‑trained on images) on text‑EEG trials,
but pad/truncate each trial to the *actual* max # of 1‑s segments in your data,
rather than always 30.
"""
import os, math, csv
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)        # catches NaNs/Infs in fwd/backward
torch.autograd.profiler.profile()              # optional: richer traces
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Sampler, BatchSampler
import argparse

from cbramod import CBraMod
from pre_process_eeg_cbramod import process_eeg
from eeg_embeds_utils import load_eeg_text_data
from tqdm import trange, tqdm
import gc
from transformers import AutoTokenizer, AutoModel
import pandas as pd            #  ← at top of file
import psutil
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
import random
import torch.nn.functional as F

def get_ram_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # in MB

def load_text_labels(csv_path: str):
    """
    Returns {normalized_sentence : category_string}
    """
    print(f"[INFO] Loading text labels from {csv_path}")
    df = pd.read_csv(csv_path, delimiter=";")
    sentences  = df.iloc[:, 2].astype(str).str.strip().str.lower()
    categories = df["category"].astype(str).str.strip()
    return dict(zip(sentences, categories))



# -----------------------------------------------------------------------------
# 1) Channel‐selection (same as before)
# -----------------------------------------------------------------------------
chanlocs = [
    'E2','E3','E4','E5','E6','E7','E9','E10','E11','E12','E13','E15','E16','E18',
    'E19','E20','E22','E23','E24','E26','E27','E28','E29','E30','E31','E33','E34',
    'E35','E36','E37','E38','E39','E40','E41','E42','E43','E44','E45','E46','E47',
    'E50','E51','E52','E53','E54','E55','E57','E58','E59','E60','E61','E62','E64',
    'E65','E66','E67','E69','E70','E71','E72','E74','E75','E76','E77','E78','E79',
    'E80','E82','E83','E84','E85','E86','E87','E89','E90','E91','E92','E93','E95',
    'E96','E97','E98','E100','E101','E102','E103','E104','E105','E106','E108','E109',
    'E110','E111','E112','E114','E115','E116','E117','E118','E120','E121','E122',
    'E123','E124','Cz'
]
keep_channels = [
    # Broca
    'E13','E19','E20','E22','E23','E24','E26','E27','E28','E29','E33','E34','E38','E43',
    # Wernicke/SMG
    'E30','E31','E35','E36','E37','E39','E40','E41','E44',
    # VWFA/N170
    'E42','E45','E46','E47','E50','E51','E52','E53','E54',
    'E57','E58','E59','E60','E61','E64','E65','E66','E69','E70',
    # N400/P600
    'E5','E6','E7','E10','E11','E12','E15','E16','E18',
    'E55','E62','E67','E71','E72','E75','E76','E77','E106','Cz'
]
global_keep_idx = [chanlocs.index(ch) for ch in keep_channels]

# 1) Define each region’s channel names
BROCA = [
    'E13','E19','E20','E22','E23','E24','E26','E27','E28','E29',
    'E33','E34','E38','E43'
]
WERNICKE_SMG = [
    'E30','E31','E35','E36','E37','E39','E40','E41','E44'
]
VWFA_N170 = [
    'E42','E45','E46','E47','E50','E51','E52','E53','E54',
    'E57','E58','E59','E60','E61','E64','E65','E66','E69','E70'
]
N400_P600 = [
    'E5','E6','E7','E10','E11','E12','E15','E16','E18',
    'E55','E62','E67','E71','E72','E75','E76','E77','E106','Cz'
]

# 3) Build region→indices *into* keep_channels
REGIONS = {
    'broca':    [keep_channels.index(ch) for ch in BROCA],
    'wernicke': [keep_channels.index(ch) for ch in WERNICKE_SMG],
    'vwfa_n170':[keep_channels.index(ch) for ch in VWFA_N170],
    'n400_p600':[keep_channels.index(ch) for ch in N400_P600],
}
# -----------------------------------------------------------------------------
# 2) Helper: compute the dataset's true max_segments (≤30)
# -----------------------------------------------------------------------------
def compute_max_segments(data, eeg_text_dir, orig_sfreq, hard_cap=30):
    max_segs = 0
    for subj, trials in data.items():
        for tr in trials:
            if not tr: continue
            raw = tr.get('rawData', None)
            if raw is None or raw.shape == () or raw.size == 0: continue
            segs = math.ceil(raw.shape[0] / orig_sfreq)
            max_segs = max(max_segs, segs)
    return min(max_segs, hard_cap)

# ---------------------------------------------------------------
# JointDataset  —  BERT × EEG pairs with 8-class text labels
# ---------------------------------------------------------------
class JointDataset(Dataset):
    """
    For every (subject × repetition) trial it returns
        bert_vec : torch.FloatTensor [768]
        eeg_proc : torch.FloatTensor [61, P, 200]
        lab_id   : int   (0 … #classes-1)
    If avg_subjects=True it averages the EEG across all subjects
    that saw the same sentence, so each sentence appears exactly
    once in the dataset.
    
    Lazy loading: Data is loaded only when needed during __getitem__ calls.
    """
    def __init__(self,
                 data_dicts:   dict,
                 tokenizer,
                 bert,
                 sent2cat:     dict,          # sentence -> category string
                 cat2idx:      dict,          # category -> int
                 *,
                 orig_sfreq:   int,
                 max_segments: int,
                 avg_subjects: bool=False,
                 max_bert_cache: int=1000):   # Limit the BERT cache size
        self.tokenizer = tokenizer
        self.bert = bert
        self.sent2cat = sent2cat
        self.cat2idx = cat2idx
        self.max_T = orig_sfreq * max_segments
        self.bert_cache = {}  # Limited cache
        self.bert_cache_keys = []  # To track LRU for cache eviction
        self.max_bert_cache = max_bert_cache
        self.data_dicts = data_dicts
        
        # Instead of storing processed data, store references to data
        self.sample_refs = []  # List of references to data locations
        
        # Keep track of class distribution
        self.class_counts = Counter()
        self.samples_by_class = defaultdict(list)
        
        # For averaging across subjects
        if avg_subjects:
            self.eeg_refs_by_sent = {}  # sentence -> list of (subject, trial_idx)
            self.sent_list = []  # List of unique sentences
        
        # Prepare data references
        for subj, trials in data_dicts.items():
            for tr_idx, tr in enumerate(trials):
                if not tr: continue
                sent = tr.get('content')
                raw = tr.get('rawData')
                if raw is None or sent is None or raw.size == 0 or raw.shape == ():
                    continue
                sent_norm = str(sent).strip().lower()
                if sent_norm not in sent2cat:  # skip unlabeled
                    continue
                
                cat = sent2cat[sent_norm]
                lab_id = cat2idx[cat]
                
                if avg_subjects:
                    self.eeg_refs_by_sent.setdefault(sent_norm, []).append((subj, tr_idx))
                    if sent_norm not in self.sent_list:
                        self.sent_list.append(sent_norm)
                else:
                    sample_ref = {
                        'subject': subj,
                        'trial_idx': tr_idx,
                        'sentence': sent_norm,
                        'label_id': lab_id
                    }
                    self.sample_refs.append(sample_ref)
                    self.class_counts[lab_id] += 1
                    self.samples_by_class[lab_id].append(len(self.sample_refs) - 1)
        
        # For averaged subjects, create sample references based on unique sentences
        if avg_subjects:
            for sent_norm in self.sent_list:
                lab_id = cat2idx[sent2cat[sent_norm]]
                sample_ref = {
                    'sentence': sent_norm,
                    'eeg_refs': self.eeg_refs_by_sent[sent_norm],
                    'label_id': lab_id
                }
                self.sample_refs.append(sample_ref)
                self.class_counts[lab_id] += 1
                self.samples_by_class[lab_id].append(len(self.sample_refs) - 1)
            
            # We don't need the full reference dictionary anymore
            self.eeg_refs_by_sent = None
            self.sent_list = None
            
        print(f"[INFO] JointDataset: {len(self.sample_refs)} samples prepared for lazy loading, "
              f"{len(cat2idx)} classes")
        print("[INFO] Class distribution:", dict(self.class_counts))
        
        # Calculate class weights for loss weighting (inverse frequency)
        total_samples = sum(self.class_counts.values())
        self.class_weights = {
            cls: total_samples / (len(self.class_counts) * count)
            for cls, count in self.class_counts.items()
        }
        print("[INFO] Class weights:", self.class_weights)
            
    def __len__(self):
        return len(self.sample_refs)
    
    def get_class_weights(self):
        """Return class weights as a tensor for weighted loss"""
        sorted_weights = [self.class_weights[i] for i in range(len(self.class_weights))]
        return torch.tensor(sorted_weights)
    
    def _get_bert_vec(self, sentence):
        """Helper to get or compute BERT vector for a sentence with LRU cache"""
        if sentence not in self.bert_cache:
            # Manage cache size
            if len(self.bert_cache) >= self.max_bert_cache:
                # Remove least recently used item
                oldest_key = self.bert_cache_keys.pop(0)
                del self.bert_cache[oldest_key]
            
            # Compute new embedding
            with torch.no_grad():
                self.bert.eval()
                toks = self.tokenizer(sentence,
                                     return_tensors="pt",
                                     truncation=True, max_length=64)
                embedding = self.bert(**toks).last_hidden_state[:, 0].squeeze(0)
                self.bert_cache[sentence] = embedding
                self.bert_cache_keys.append(sentence)
        else:
            # Update LRU order
            self.bert_cache_keys.remove(sentence)
            self.bert_cache_keys.append(sentence)
            
        return self.bert_cache[sentence]
    
    def _load_eeg_data(self, subject, trial_idx):
        """Load EEG data for a specific trial"""
        raw = self.data_dicts[subject][trial_idx].get('rawData')
        # Slice and pad
        sl = raw[:self.max_T, global_keep_idx]
        if sl.shape[0] < self.max_T:
            pad = np.zeros((self.max_T - sl.shape[0], sl.shape[1]))
            sl = np.vstack([sl, pad])
        return sl
    
    def __getitem__(self, idx):
        ref = self.sample_refs[idx]
        sentence = ref['sentence']
        label_id = ref['label_id']
        
        # Get BERT vector (will be cached)
        bert_vec = self._get_bert_vec(sentence)
        
        # Load EEG data - this is where the lazy loading happens
        if 'eeg_refs' in ref:  # This is an averaged sample
            eeg_slices = []
            for subj, tr_idx in ref['eeg_refs']:
                eeg_slices.append(self._load_eeg_data(subj, tr_idx))
            eeg_data = np.mean(eeg_slices, axis=0)
        else:  # Individual sample
            eeg_data = self._load_eeg_data(ref['subject'], ref['trial_idx'])
        
        # Process EEG data
        proc = process_eeg(eeg_data, 500)  # ➜ [1,61,P,200]
        x_eeg = torch.from_numpy(proc).float().squeeze(0)
        
        return bert_vec.float(), x_eeg, label_id

    def clear_bert_cache(self):
        """Clear BERT embedding cache to free memory"""
        self.bert_cache = {}
        self.bert_cache_keys = []
        gc.collect()  # Force garbage collection

    def get_class_distribution(self):
        """Return the class distribution in the dataset"""
        return self.class_counts
    
def info_nce(z, y, temperature=0.1, *, debug=True):
    """
    Supervised InfoNCE loss with safety checks.
    Set debug=True to have it assert-and-print the moment
    any intermediate goes non-finite.
    """
    # ------------------------------------------------------------
    def _check(tensor, tag, *, ignore_mask=None):
        """
        Fail if tensor contains NaN/±Inf **outside** the positions marked
        True in `ignore_mask`.
        """
        if not debug:
            return
    
        bad_nan = torch.isnan(tensor)
        bad_inf = torch.isinf(tensor)
    
        if ignore_mask is not None:
            bad_inf = bad_inf & ~ignore_mask      # allow our own -inf on diagonal
    
        bad = bad_nan | bad_inf
        if bad.any():
            vals = tensor[bad][:5].tolist()
            print(f"\n[NaN-tracker]  {tag} -> bad values: {vals} …")
            raise RuntimeError(f"NaN/Inf detected in {tag}")
    # ------------------------------------------------------------

    # 1. normalise embeddings
    z = F.normalize(z, dim=1, eps=1e-6)
    _check(z, "z-norm")

    # 2. filter rows w/o positives
    B = z.size(0)
    self_m = torch.eye(B, dtype=torch.bool, device=z.device)
    pos_m  = (y.view(1, -1) == y.view(-1, 1)) & ~self_m
    keep   = (pos_m.sum(1) > 0).nonzero(as_tuple=True)[0]
    if keep.numel() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    z, y   = z[keep], y[keep]
    B      = z.size(0)
    self_m = torch.eye(B, dtype=torch.bool, device=z.device)
    pos_m  = (y.view(1, -1) == y.view(-1, 1)) & ~self_m

    # 3. similarity, mask self, row-wise stabilisation
    sim = torch.matmul(z, z.T).div_(temperature)
    _check(sim, "raw-sim")

    sim.masked_fill_(self_m, float('-inf'))
    _check(sim, "sim-masked", ignore_mask=self_m)

    row_max, _ = sim.max(1, keepdim=True)          # finite because at least one non-inf
    logits = sim - row_max
    _check(logits, "logits-shifted", ignore_mask=self_m)

    # 4. log-softmax (stable)
    exp_logits = torch.exp(logits)
    denom  = exp_logits.sum(1, keepdim=True).clamp_min(1e-8)
    log_p  = logits - torch.log(denom)
    _check(log_p, "log-prob", ignore_mask=self_m)

    # 5. mean log-prob of positives
    log_p = log_p.masked_fill(self_m, 0.0)   # neutral element for the sum
    mean_pos = (pos_m * log_p).sum(1) / pos_m.sum(1)
    _check(mean_pos, "mean-pos")

    return -mean_pos.mean()



# -----------------------------------------------------------------------------
# 4) Model: loads only the backbone from your image‑fine‑tuned checkpoint
# -----------------------------------------------------------------------------
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
            seq_len=30,        # ← variable #segments
            n_layer=12, nhead=8
        )
        self.backbone.proj_out = nn.Identity()

        flat_dim   = param.channels * param.patches * param.d_model
        hidden_dim = param.patches  * param.d_model
        
        # final proj from (#regions × d_model) → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(len(REGIONS)*param.d_model, 512),
            nn.ELU(), nn.Dropout(param.dropout),
            nn.Linear(512, param.num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)   # [B, C, P, D]
        region_vecs = []
        for idxs in REGIONS.values():
            # select only that region’s channels:
            sub = feats[:, idxs, :, :]       # [B, |regs|, P, D]
            # average over channels & patches:
            rv = sub.mean(dim=(1,2))         # [B, D]
            region_vecs.append(rv)
        z = torch.cat(region_vecs, dim=1)    # [B, R·D]
        logits = self.classifier(z)
        return z, logits

    #     self.embedder = nn.Sequential(
    #         nn.Linear(flat_dim, hidden_dim),
    #         nn.ELU(), nn.Dropout(param.dropout),
    #         nn.Linear(hidden_dim, param.d_model),
    #         nn.ELU()
    #     )
    #     self.classifier = nn.Sequential(
    #         nn.Dropout(param.dropout),
    #         nn.Linear(param.d_model, param.num_classes)
    #     )

    # def forward(self, x):
    #     feats = self.backbone(x)
    #     B,C,P,D = feats.shape
    #     flat = feats.contiguous().view(B, C*P*D)
    #     z    = self.embedder(flat)      # [B,200]
    #     logits = self.classifier(z)
    #     return z, logits

class Text2EEG(nn.Module):
    def __init__(self, in_dim: int = 768, out_dim: int = 800):
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
    def forward(self, x_eeg, x_bert):
        z_eeg, logits = self.eeg_encoder(x_eeg)   # [B,800]
        z_txt         = self.projector(x_bert)    # [B,800]
        return z_eeg, z_txt, logits

# ------------------------------------------------------------------
# 3 · Training / evaluation helpers
# ------------------------------------------------------------------
CE_WEIGHT, NCE_WEIGHT, MSE_WEIGHT, COS_WEIGHT = 0.2, 1.0, 1.0, 0.5

from torch.cuda.amp import autocast, GradScaler
# -----------------------------------------------------------------------------
# 5) Training/Eval loops (standard)
# -----------------------------------------------------------------------------

def run_epoch(model, loader, opt, sched, device, criterion_ce):
    """One training epoch. Returns avg CE‑loss."""
    model.train(); tot_loss = 0; correct=0; N=0
    steps = 0
    for bvec, eeg_raw, lab in tqdm(loader, total=len(loader),desc=f"Train", leave=False, dynamic_ncols=True):
        steps += 1
        bvec, eeg_raw, lab = bvec.to(device), eeg_raw.to(device), lab.to(device)
        opt.zero_grad()
        z_eeg, z_txt, logits = model(eeg_raw, bvec)
        eps = 1e-6
        z_eeg = z_eeg + eps
        z_txt = z_txt + eps
        cos_sim   = F.cosine_similarity(z_eeg, z_txt).mean()
        ce      = criterion_ce(logits, lab)
        nce     = info_nce(z_eeg, lab)
        mse     = F.mse_loss(z_eeg, z_txt)
        loss    = (CE_WEIGHT * ce +
                   NCE_WEIGHT * nce +
                   MSE_WEIGHT * mse +
                   COS_WEIGHT * (1 - cos_sim))
        print(f"ce={ce.item()}  nce={nce.item()}  mse={mse.item()}  cos={cos_sim.item()}")
        if torch.isnan(loss):
            print(f"ce={ce.item()}  nce={nce.item()}  mse={mse.item()}  cos={cos_sim.item()}")
            raise RuntimeError("NaN loss – see component values above")
        for name, p in model.named_parameters():
            if torch.isnan(p).any():
                raise RuntimeError(f"Parameter NaN in {name}")
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                raise RuntimeError(f"Gradient NaN in {name}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
        tot_loss += loss.item()*len(lab)
        correct  += (logits.argmax(1)==lab).sum().item(); N += len(lab)
        if steps % 5 == 0:
            print(f"\nstep {steps} loss {loss.item():.4f}\n")
            # print(f"\ncuda mem alloc: {torch.cuda.memory_allocated()}\n")
            # print(f"\ncuda mem res: {torch.cuda.memory_reserved()}\n")
            # print(f"RAM usage during train step: {get_ram_usage()} MB")
            # break
    return tot_loss/len(loader.dataset), correct/N

# @torch.no_grad()
def evaluate(model, loader, device, criterion_ce):
    model.eval(); ce_tot, correct, Z, Y = 0,0,[],[]
    steps = 0
    with torch.no_grad():
        for bvec, eeg_raw, lab in tqdm(loader, total=len(loader),desc=f"Eval", leave=False, dynamic_ncols=True):
            steps += 1
            bvec, eeg_raw, lab = bvec.to(device), eeg_raw.to(device), lab.to(device)
            z_eeg, z_txt, logits = model(eeg_raw, bvec)
            ce_tot += criterion_ce(logits, lab).item()*len(lab)
            correct+= (logits.argmax(1)==lab).sum().item()
            Z.append(z_eeg.cpu()); Y.append(lab.cpu())
            if steps % 5 == 0:
                # print(f"\nstep {steps} loss {loss.item():.4f}\n")
                print(f"RAM usage during evaluation: {get_ram_usage()} MB")
                # break
    Z = torch.cat(Z); Y = torch.cat(Y)
    knn = knn_metrics(Z, Y, k=5)
    return ce_tot/len(loader.dataset), correct/len(loader.dataset), knn['1nn'], knn['recall@5']
    
from sklearn.neighbors import NearestNeighbors

@torch.no_grad()
def knn_metrics(embeddings, labels, k=5, metric='cosine'):
    """
    embeddings : Tensor [N, d]
    labels     : Tensor [N]
    Returns dict with 1-NN accuracy and Recall@k.
    Computes *leave-one-out* k-NN on the validation set itself.
    """
    # L2-normalise for cosine metric
    z = torch.nn.functional.normalize(embeddings, dim=1).cpu().numpy()
    y = labels.cpu().numpy()

    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)  # +1 for the sample itself
    nn.fit(z)
    dist, idx = nn.kneighbors(z)                           # idx shape [N, k+1]

    idx = idx[:, 1:]                                       # drop self-match at col 0
    neigh_labels = y[idx]                                  # shape [N, k]

    # 1-NN accuracy
    acc_1nn = (neigh_labels[:, 0] == y).mean()

    # Recall@k (hit if *any* of top-k neighbours shares the label)
    hits = (neigh_labels == y[:, None]).any(axis=1)
    recall_k = hits.mean()

    return {'1nn': acc_1nn, f'recall@{k}': recall_k}
    
# from collections import Counter, defaultdict
# def balanced_sampler(labels, repeat=4):
#     buckets = defaultdict(list)
#     for idx, lab in enumerate(labels):
#         buckets[lab].append(idx)
#     idxs = []
#     # draw 'repeat' copies of each label index list shuffled
#     for _ in range(repeat):
#         for lab, arr in buckets.items():
#             idxs.extend(np.random.permutation(arr))
#     return torch.utils.data.SubsetRandomSampler(idxs)
from collections import defaultdict
from torch.utils.data import SubsetRandomSampler

# ----------------------------------------------------------
# def balanced_sampler_from_subset(subset, repeat: int = 4):
#     """
#     Build a SubsetRandomSampler that yields a class-balanced index
#     stream *local* to a torch.utils.data.Subset.

#     Parameters
#     ----------
#     subset : torch.utils.data.Subset
#         The training split returned by random_split.
#     repeat : int, default=4
#         How many times to iterate over the label buckets when
#         composing a single epoch’s index list.  Increase if you
#         want longer training epochs.

#     Returns
#     -------
#     sampler : torch.utils.data.SubsetRandomSampler
#         Can be passed directly to DataLoader(..., sampler=sampler).
#     """
#     # ---- bucket local indices by label id -------------------
#     buckets = defaultdict(list)      # lab_id → list[local_idx]

#     for local_idx, global_idx in enumerate(subset.indices):
#         # JointDataset stores label_id in position 2 of each tuple
#         lab_id = subset.dataset.sample_refs[global_idx]["label_id"]
#         buckets[lab_id].append(local_idx)

#     # ---- flatten buckets with in-bucket shuffling ------------
#     rng  = np.random.default_rng()
#     flat = []
#     for _ in range(repeat):
#         for idxs in buckets.values():
#             flat.extend(rng.permutation(idxs))

#     return SubsetRandomSampler(flat)
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, oversample_factor=1.0, indices=None):
        """
        Parameters:
        -----------
        dataset : JointDataset
            Must have samples_by_class attribute
        batch_size : int
            Size of each batch
        oversample_factor : float
            Factor to oversample minority classes. If >1.0, minority classes will be oversampled.
        indices : list
            List of indices to use (for using with a subset)
        """
        self.dataset = dataset
        self.batch_size = batch_size
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
        for cls in range(self.n_classes):
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

def main(args):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # ------------ data --------------
    raw_dict = load_eeg_text_data(args.eeg_dir)
    max_seg  = compute_max_segments(raw_dict, args.eeg_dir, args.orig_sfreq)

    tok  = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").eval()
    
    sent2cat = load_text_labels(args.label_csv)
    uniq_cats = sorted(set(sent2cat.values()))
    cat2idx   = {c:i for i,c in enumerate(uniq_cats)}
    num_classes = len(uniq_cats)
    print(f"[INFO] Detected {num_classes} text classes:", uniq_cats)


    dummy = argparse.Namespace(channels=len(global_keep_idx), patches=max_seg,
                               d_model=200, dim_feedforward=800, n_layer=12,
                               nhead=8, dropout=args.dropout, num_classes=num_classes)
    eeg_enc = CBraModTextClassifier(dummy)
    # eeg_enc.load_state_dict(torch.load(args.eeg_ckpt, map_location="cpu"))
    

    # 4) load pretrained *backbone* weights
    state = torch.load(args.eeg_ckpt, map_location=device)
    # bk = {k.split("backbone.")[1]:v for k,v in state.items() if k.startswith("backbone.")}
    bk = {k:v for k,v in state.items() if not k.startswith("proj_out.")}
    eeg_enc.backbone.load_state_dict(bk)

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

    print("Model loaded and ready ...")
    
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Trainable parameters only
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    dataset = JointDataset(raw_dict, tok, bert, sent2cat, cat2idx, orig_sfreq=args.orig_sfreq,
                           max_segments=max_seg, avg_subjects=args.avg_subjects)
    # tr_len  = int(0.9*len(dataset))
    # tr_ds, va_ds = random_split(dataset, [tr_len, len(dataset)-tr_len])
    orig_len = len(dataset)
    frac = 1.0
    if frac < 1.0:
        small_n = int(orig_len * frac)
        dataset, _ = random_split(dataset, [small_n, orig_len - small_n])
        print(f"[INFO] Using only {small_n}/{orig_len} samples ({frac*100:.1f}%)")
    tr_len  = int(0.9*len(dataset))
    tr_ds, va_ds = random_split(dataset, [tr_len, len(dataset)-tr_len])
    # sampler = balanced_sampler_from_subset(tr_ds, repeat=2)
    sampler = BalancedBatchSampler(dataset, args.batch, 3.0, tr_ds.indices)
    tr_dl = DataLoader(dataset, batch_sampler=sampler)
    va_dl = DataLoader(va_ds, batch_size=args.batch)

    opt  = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                             lr=args.lr, weight_decay=1e-2)
    # sched= torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs*len(tr_dl))
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
    ce   = nn.CrossEntropyLoss()

    metrics_path = os.path.join(args.model_dir, "metrics_cbramod_text_cpu_latest.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "tr_loss", "va_loss", "va_acc", "va_1nn", "va_r5"])

    print(f"RAM usage before training starts: {get_ram_usage()} MB")
    def nan_checker(name):
        def hook(_, inp, out):
            # check both input and output tensors
            for t in (*inp, out):
                if torch.is_tensor(t) and torch.isnan(t).any():
                    raise RuntimeError(f"NaN detected in {name}")
        return hook
    
    # register the hook on every sub-module
    for name, module in model.named_modules():
        module.register_forward_hook(nan_checker(name))

    
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

# -------------------------------------------------------------------
if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--eeg_dir",    required=True)
    P.add_argument("--eeg_ckpt",   required=True)
    P.add_argument("--out",        required=True)
    P.add_argument("--model_dir",        required=True)
    P.add_argument("--epochs",     type=int, default=20)
    P.add_argument("--batch",      type=int, default=64)
    P.add_argument("--lr",         type=float, default=3e-4)
    P.add_argument("--orig_sfreq", type=int, default=500)
    P.add_argument("--dropout",    type=float, default=0.2)
    P.add_argument("--cuda",       type=int, default=0)
    P.add_argument("--avg_subjects", action="store_true",
                   help="average EEG embeddings across subjects")
    P.add_argument("--label_csv", required=True,
               help="semicolon-separated file with <…;…;sentence;…;category>")

    main(P.parse_args())
