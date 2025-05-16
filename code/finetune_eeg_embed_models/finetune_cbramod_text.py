#!/usr/bin/env python3
"""
fine_tune_cbramod_text.py

Fine‑tune a CBraMod model (pre‑trained on images) on text‑EEG trials,
but pad/truncate each trial to the *actual* max # of 1‑s segments in your data,
rather than always 30.
"""
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from cbramod import CBraMod
from pre_process_eeg_cbramod import process_eeg
from rsa_cca_grouped_analysis_with_ensemble import load_eeg_text_data

# -----------------------------------------------------------------------------
# 1) Channel‐selection (same as before)
# -----------------------------------------------------------------------------
chanlocs = [ ... ]   # your full list of 146 channels
keep_channels = [ ... ]  # your 61‐channel subset
global_keep_idx = [chanlocs.index(ch) for ch in keep_channels]

# -----------------------------------------------------------------------------
# 2) Helper: compute the dataset's true max_segments (≤30)
# -----------------------------------------------------------------------------
def compute_max_segments(eeg_text_dir, orig_sfreq, hard_cap=30):
    data = load_eeg_text_data(eeg_text_dir)
    max_segs = 0
    for subj, trials in data.items():
        for tr in trials:
            raw = tr.get('rawData', None)
            if raw is None: continue
            segs = math.ceil(raw.shape[0] / orig_sfreq)
            max_segs = max(max_segs, segs)
    return min(max_segs, hard_cap)

# -----------------------------------------------------------------------------
# 3) Dataset: now takes max_segments as an argument
# -----------------------------------------------------------------------------
class EEGTextDataset(Dataset):
    def __init__(self, eeg_text_dir, orig_sfreq, max_segments):
        self.orig_sfreq  = orig_sfreq
        self.max_segments = max_segments
        self.max_samples = orig_sfreq * max_segments

        data_dicts = load_eeg_text_data(eeg_text_dir)
        raws, labels = [], []

        for subj, trials in data_dicts.items():
            for tr in trials:
                raw = tr.get('rawData', None)
                lbl = tr.get('content', None)
                if raw is None or lbl is None: 
                    continue

                # select channels
                sel = raw[:, global_keep_idx]                  # (T,61)

                # truncate or pad to exactly max_samples
                L = sel.shape[0]
                if L > self.max_samples:
                    sel = sel[:self.max_samples]
                elif L < self.max_samples:
                    pad = np.zeros((self.max_samples - L, sel.shape[1]))
                    sel = np.vstack([sel, pad])

                raws.append(sel)
                labels.append(lbl)

        # build label→idx
        uniq = sorted(set(labels))
        self.label2idx   = {lab:i for i,lab in enumerate(uniq)}
        self.num_classes = len(uniq)
        self.raws        = raws
        self.labels      = [self.label2idx[l] for l in labels]

    def __len__(self):
        return len(self.raws)

    def __getitem__(self, idx):
        raw = self.raws[idx]          # (max_samples, 61)
        proc = process_eeg(raw, self.orig_sfreq)  # → [1,61,segments,200]
        x    = torch.from_numpy(proc).float().squeeze(0)
        y    = self.labels[idx]
        return x, y

# -----------------------------------------------------------------------------
# 4) Model: loads only the backbone from your image‑fine‑tuned checkpoint
# -----------------------------------------------------------------------------
class CBraModTextClassifier(nn.Module):
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
        hidden_dim = param.patches * param.d_model

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ELU(), nn.Dropout(param.dropout),
            nn.Linear(hidden_dim, param.d_model),
            nn.ELU(), nn.Dropout(param.dropout),
            nn.Linear(param.d_model, param.num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        B,C,P,D = feats.shape
        out = feats.contiguous().view(B, C*P*D)
        return self.classifier(out)

# -----------------------------------------------------------------------------
# 5) Training/Eval loops (standard)
# -----------------------------------------------------------------------------
def train_epoch(model, loader, optim, crit, device):
    model.train()
    total_loss = 0.
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optim.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    total_loss, correct = 0., 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += crit(logits,y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return total_loss/len(loader.dataset), correct/len(loader.dataset)

# -----------------------------------------------------------------------------
# 6) Main: glue everything together
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_text_dir",    required=True)
    parser.add_argument("--finetuned_model", required=True) # here we will used pretrained weights itself
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--orig_sfreq",  type=int,   default=500)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--cuda",        type=int,   default=0)
    args = parser.parse_args()

    # 1) figure out dynamic max_segments
    args.patches = compute_max_segments(args.eeg_text_dir,
                                        args.orig_sfreq,
                                        hard_cap=30)
    print(f"→ Padding/truncating to {args.patches} segments (≤30).")

    # 2) dataset & split
    dataset = EEGTextDataset(args.eeg_text_dir,
                             args.orig_sfreq,
                             max_segments=args.patches)
    n_val   = int(len(dataset)*0.2)
    n_tr    = len(dataset)-n_val
    tr_ds, va_ds = random_split(dataset, [n_tr, n_val])
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size)

    # 3) build model
    args.channels        = len(global_keep_idx)  # 61
    args.d_model         = 200
    args.dim_feedforward = 800
    args.n_layer         = 12
    args.nhead           = 8

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model  = CBraModTextClassifier(args).to(device)

    # 4) load pretrained *backbone* weights
    state = torch.load(args.finetuned_model, map_location=device)
    bk = {k.split("backbone.")[1]:v for k,v in state.items() if k.startswith("backbone.")}
    model.backbone.load_state_dict(bk)

    # 5) train
    optim    = optim.Adam(model.parameters(), lr=args.lr)
    criterion= nn.CrossEntropyLoss()
    best_acc = 0.
    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, tr_dl, optim, criterion, device)
        va_loss, va_acc = eval_epoch(model, va_dl, criterion, device)
        print(f"Ep{ep:02d}  tr_loss:{tr_loss:.4f}  va_loss:{va_loss:.4f}  va_acc:{va_acc:.2%}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), "best_cbramod_text_model.pth")

    print(f"Done! Best val acc = {best_acc:.2%}")
