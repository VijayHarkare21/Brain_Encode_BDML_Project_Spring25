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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from cbramod import CBraMod
from pre_process_eeg_cbramod import process_eeg
from eeg_embeds_utils import load_eeg_text_data
from tqdm import trange, tqdm
import gc


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

# -----------------------------------------------------------------------------
# 3) Dataset  –  lazy slice-on-demand using data_dicts already in RAM
# -----------------------------------------------------------------------------
class EEGTextDataset(Dataset):
    """
    data_dicts : dict[subj → list[trial_dict]]
                 Each trial_dict contains keys 'rawData' (np.ndarray [T, #all_ch])
                 and  'content'  (string label).

    We DO NOT copy or preprocess arrays at build time.
    self.index holds tuples (raw_array, row0, row1, label_int).
    Channel-selection, padding and float-casting happen in __getitem__.
    """
    def __init__(self,
                 data_dicts:   dict,
                 orig_sfreq:   int,
                 max_segments: int):
        self.orig_sfreq   = orig_sfreq
        self.max_segments = max_segments
        self.max_samples  = orig_sfreq * max_segments

        index_tuples  = []     # temporary list before label map exists
        all_labels    = []     # collect strings first

        for subj, trials in data_dicts.items():
            print(f"Scanning subject {subj}")
            for tr in trials:
                if not tr: continue
                raw = tr.get('rawData', None)
                lbl = tr.get('content', None)
                if raw is None or lbl is None or raw.size == 0 or raw.shape == ():
                    continue

                n_rows = raw.shape[0]
                row0 = 0
                while row0 < n_rows:
                    row1 = min(row0 + self.max_samples, n_rows)
                    index_tuples.append((raw, row0, row1, lbl))
                    all_labels.append(lbl)
                    row0 = row1

        # ---- label → int mapping ----
        uniq = sorted(set(all_labels))
        self.label2idx   = {lab: i for i, lab in enumerate(uniq)}
        self.num_classes = len(uniq)
        self.labels_int  = [self.label2idx[l] for l in all_labels]

        # ---- final index with integer labels ----
        self.index = [
            (raw, r0, r1, self.label2idx[lbl])
            for (raw, r0, r1, lbl) in index_tuples
        ]
        assert len(self.index) == len(self.labels_int)

        # free no-longer-needed lists
        del index_tuples, all_labels

    # ----------------------------------------------------------
    def __len__(self):
        return len(self.index)

    # ----------------------------------------------------------
    def __getitem__(self, idx: int):
        raw_full, row0, row1, label_int = self.index[idx]

        # --- slice ONLY the needed rows (view, not copy) ----
        slice_ = raw_full[row0:row1, :]                  # (T, #all_channels)

        # --- select language-relevant channels (makes a *copy* but small) ---
        slice_ = slice_[:, global_keep_idx]              # (T, 61)

        # --- pad if needed to exactly max_samples ----------
        T = slice_.shape[0]
        if T < self.max_samples:
            pad = np.zeros((self.max_samples - T, slice_.shape[1]),
                           dtype=slice_.dtype)
            slice_ = np.vstack([slice_, pad])
        # slice_ = slice_.astype(np.float16, copy=False)
        # ---- preprocess → [1, 61, segments, 200] ---------
        proc = process_eeg(slice_, self.orig_sfreq)

        # x = torch.from_numpy(proc).to(dtype=torch.float16).squeeze(0)    # (61, segs, 200)
        x = torch.from_numpy(proc).float().squeeze(0)    # (61, segs, 200)
        y = label_int
        return x, y

        
# -------- supervised InfoNCE on embeddings -----------------
def info_nce(z, y, temperature=0.07):
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature          # [B,B]
    labels = y.unsqueeze(0) == y.unsqueeze(1)         # positives mask
    logits_mask = torch.ones_like(sim, dtype=torch.bool)
    logits_mask.fill_diagonal_(0)
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
    loss = -(log_prob * labels.float()).sum(1) / labels.sum(1).clamp(min=1)
    return loss.mean()


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

from torch.cuda.amp import autocast, GradScaler
# -----------------------------------------------------------------------------
# 5) Training/Eval loops (standard)
# -----------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, crit, device):
    lambda_ce  = 0.2          # weight for CE
    lambda_nce = 1.0          # weight for InfoNCE
    temperature= 0.07         # same as SimCLR
    model.train()
    scaler = GradScaler()
    total_loss = 0.
    steps = 0
    for x, y in tqdm(loader,
                     total=len(loader),
                     desc=f"Train",
                     leave=False,
                     dynamic_ncols=True):
        steps += 1
        
        # x,y = x.to(device, dtype=torch.float16), y.to(device)
        x,y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # with autocast(dtype=torch.float16):
        #     z, logits = model(x)
        #     loss_ce  = crit(logits, y)
        #     loss_nce = info_nce(z, y, temperature)
        #     loss     = lambda_ce * loss_ce + lambda_nce * loss_nce
        # scaler.scale(loss).backward()               # ← NEW
        # scaler.step(optimizer)                      # ← NEW
        # scaler.update()
        
        z, logits = model(x)
        loss_ce  = crit(logits, y)
        loss_nce = info_nce(z, y, temperature)
        loss     = lambda_ce * loss_ce + lambda_nce * loss_nce
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        total_loss += loss.item() * x.size(0)
        # if steps % 5 == 0:
        #     # torch.cuda.empty_cache()
        #     break
        if steps % 5 == 0:
            print(f"step {steps} loss {loss.item():.4f}")
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, crit, device, k=5):
    model.eval()
    ce_tot, correct, N = 0., 0, 0
    Z, Y = [], []
    # step = 0
    for x,y in tqdm(loader,
                     total=len(loader),
                     desc=f"Eval",
                     leave=False,
                     dynamic_ncols=True):
        # x,y = x.to(device, dtype=torch.float16), y.to(device)
        x,y = x.to(device), y.to(device)
        # with autocast(dtype=torch.float16):
        #     z, logits = model(x)
        z, logits = model(x)
        ce_tot += crit(logits,y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        N += x.size(0)
        Z.append(z); Y.append(y)
        # if step % 5 == 0:
        #     # torch.cuda.empty_cache()
        #     break
    Z = torch.cat(Z,0); Y = torch.cat(Y,0)
    knn = knn_metrics(Z, Y, k=k)          # {'1nn':…, 'recall@5':…}
    return ce_tot/N, correct/N, knn['1nn'], knn['recall@5']
    
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
def balanced_sampler_from_subset(subset, repeat=4):
    """
    subset : torch.utils.data.Subset produced by random_split
    repeat : how many times to iterate over the label buckets
             (repeat ≥ 2 guarantees at least two positives / batch
              for most labels when batch_size ≥ len(labels))

    Returns a SubsetRandomSampler whose indices are 0 … len(subset)-1
    """
    # full-label list lives in the *underlying* dataset
    all_labels = subset.dataset.labels_int

    # build buckets keyed by label
    buckets = defaultdict(list)          # label_int → list[local_idx]
    for local_idx, global_idx in enumerate(subset.indices):
        lab = all_labels[global_idx]
        buckets[lab].append(local_idx)

    # flatten with shuffling inside each bucket
    flat = []
    rng = np.random.default_rng()
    for _ in range(repeat):
        for idxs in buckets.values():
            flat.extend(rng.permutation(idxs))
    return SubsetRandomSampler(flat)


# -----------------------------------------------------------------------------
# 6) Main: glue everything together
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_text_dir",    required=True)
    parser.add_argument("--pretrained_model", required=True) # here we will used pretrained weights itself
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--orig_sfreq",  type=int,   default=500)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--cuda",        type=int,   default=0)
    args = parser.parse_args()

    
    data = load_eeg_text_data(args.eeg_text_dir)
    # 1) figure out dynamic max_segments
    args.patches = compute_max_segments(data, args.eeg_text_dir,
                                        args.orig_sfreq,
                                        hard_cap=30)

    print(f"→ Padding/truncating to {args.patches} segments (≤30).")
    # 2) dataset & split
    dataset = EEGTextDataset(data,
                             args.orig_sfreq,
                             max_segments=args.patches)
    print("Dataset created ...")           
    del data
    gc.collect()
    n_val   = int(len(dataset)*0.2)
    n_tr    = len(dataset)-n_val
    tr_ds, va_ds = random_split(dataset, [n_tr, n_val])
    # sampler = balanced_sampler(dataset.labels_int)
    sampler = balanced_sampler_from_subset(tr_ds, repeat=4)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size)
    
    print("Data loaders created ...")

    # 3) build model
    args.channels        = len(global_keep_idx)  # 61
    args.d_model         = 200
    args.dim_feedforward = 800
    args.n_layer         = 12
    args.nhead           = 8
    args.num_classes = dataset.num_classes

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model  = CBraModTextClassifier(args).to(device)
    
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Trainable parameters only
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # 4) load pretrained *backbone* weights
    state = torch.load(args.pretrained_model, map_location=device)
    # bk = {k.split("backbone.")[1]:v for k,v in state.items() if k.startswith("backbone.")}
    bk = {k:v for k,v in state.items() if not k.startswith("proj_out.")}
    model.backbone.load_state_dict(bk)
    print("Model loaded and ready ...")
    # 5) train
    optimizer    = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9,0.999))
    total_steps  = len(tr_dl) * args.epochs
    warmup_steps = len(tr_dl) * 3
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
            )
        ],
        milestones=[warmup_steps]
    )
    print("Training starts ...")
    criterion= nn.CrossEntropyLoss()
    best_acc = 0.
    
    metrics_path = os.path.join(args.model_dir, "metrics_cbramod_text.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "tr_loss", "va_loss", "va_acc", "va_1nn", "va_r5"])
    
    for ep in trange(1, args.epochs + 1,
                 desc="Epochs",
                 unit="epoch",
                 dynamic_ncols=True):
        tr_loss = train_epoch(model, tr_dl, optimizer, scheduler, criterion, device)
        va_loss, va_acc, va_1nn, va_r5 = eval_epoch(model, va_dl, criterion, device, k=5)
        tqdm.write(
            f"Ep{ep:02d} | CE_tr {tr_loss:.4f} | CE_va {va_loss:.4f} "
            f"| CE_acc {va_acc:.2%} | 1-NN {va_1nn:.2%} | R@5 {va_r5:.2%}"
        )
        print(f"Ep{ep:02d} | CE_tr {tr_loss:.4f} | CE_va {va_loss:.4f} "
              f"| CE_acc {va_acc:.2%} | 1-NN {va_1nn:.2%} | R@5 {va_r5:.2%}")
              
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, tr_loss, va_loss, va_acc, va_1nn, va_r5])
    
        if va_1nn > best_acc:                 # early-stop on embedding quality
            best_acc = va_1nn
            torch.save(model.state_dict(), os.path.join(args.model_dir, "best_cbramod_text_model.pth"))
 
    print(f"Done. Best 1-NN acc: {best_acc:.2%}")
