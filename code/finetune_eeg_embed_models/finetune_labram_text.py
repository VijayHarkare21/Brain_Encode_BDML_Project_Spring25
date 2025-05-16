#!/usr/bin/env python3
"""
fine_tune_labram_text.py

Fine‑tune LaBraM on EEG‑text classification.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from pre_process_eeg_labram import process_eeg                              # :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
from labram_embeddings_generator_image import get_model_init_args, \
    load_model_from_checkpoint, CHANNELS                                 # :contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}
import utils
from rsa_cca_grouped_analysis_with_ensemble import load_eeg_text_data       # :contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17}
from classifiers_eeg_embeds import load_text_labels                        # :contentReference[oaicite:18]{index=18}:contentReference[oaicite:19]{index=19}

# -----------------------------------------------------------------------------
# 1) Dataset for EEG‑Text
# -----------------------------------------------------------------------------
class EEGTextDatasetLabram(Dataset):
    def __init__(self, eeg_root, text_csv, subject_handling="stack", orig_sfreq=200):
        assert subject_handling in ("stack", "average")
        self.handling = subject_handling
        self.orig_sfreq = orig_sfreq

        # 1a) load all subjects' text trials
        txt_data = load_eeg_text_data(eeg_root)

        # 1b) build lists of (raw_eeg, sentence) pairs
        raws, sents = [], []
        for sid, trials in txt_data.items():
            for t in trials:
                if not t or "rawData" not in t or "content" not in t:
                    continue
                raws.append(t["rawData"])   # shape (T, C_all)
                sents.append(t["content"].strip().lower())

        # 1c) map sentences → labels
        label_map = load_text_labels(text_csv)
        labs = []
        for sent in sents:
            if sent not in label_map:
                raise KeyError(f"No label for sentence {sent!r}")
            labs.append(label_map[sent])

        # 1d) optionally average across subjects (not typical for text)
        if self.handling == "average":
            # group by sentence, average EEG
            from collections import defaultdict
            grp = defaultdict(list)
            for raw, lab in zip(raws, labs):
                grp[lab].append(raw)
            self.raws   = [ np.stack(grp[l],0).mean(0) for l in grp ]
            self.labels = [ l for l in grp ]
        else:
            self.raws   = raws
            self.labels = labs

        # 1e) finalize integer labels
        uniq = sorted(set(self.labels))
        self.lab2idx = {l:i for i,l in enumerate(uniq)}
        self.num_classes = len(uniq)
        self.labels = [ self.lab2idx[l] for l in self.labels ]

        # compute max segments over all trials
        seg_counts = []
        for raw in self.raws:
            T = raw.shape[0]
            n_tgt = int(T * 200 / self.orig_sfreq)
            n_seg = int(np.ceil(n_tgt/200))
            seg_counts.append(n_seg)
        self.max_seg = max(seg_counts)

        # precompute channel indices
        self.input_chans = utils.get_input_chans([c.upper() for c in CHANNELS])

    def __len__(self):
        return len(self.raws)

    def __getitem__(self, idx):
        raw = self.raws[idx]               # (T, C_all)
        proc = process_eeg(raw, self.orig_sfreq)  # → [1, C_all, n_seg, 200]
        proc = proc.squeeze(0)             # [C_all, n_seg, 200]

        # pad/truncate along segment dimension:
        C_all, n_seg, W = proc.shape
        if n_seg < self.max_seg:
            pad = np.zeros((C_all, self.max_seg-n_seg, W), dtype=proc.dtype)
            proc = np.concatenate([proc, pad], axis=1)
        elif n_seg > self.max_seg:
            proc = proc[:, :self.max_seg, :]

        x = torch.from_numpy(proc).float()           # [C_all, max_seg, 200]
        y = self.labels[idx]
        return x, y

# -----------------------------------------------------------------------------
# 2) Fine‑tuning Module
# -----------------------------------------------------------------------------
class LaBraMForTextClassification(nn.Module):
    def __init__(self, checkpoint_path, device, num_classes):
        super().__init__()
        # 2a) load backbone
        init_args = get_model_init_args()
        self.backbone = load_model_from_checkpoint(checkpoint_path, device)
        # 2b) attach new head
        embed_dim = self.backbone.embed_dim
        self.classifier = nn.Linear(embed_dim, num_classes)
        # 2c) fixed channel selection
        self.input_chans = utils.get_input_chans([c.upper() for c in CHANNELS])

    def forward(self, x):
        # x: [B, C_all, n_seg, 200]
        feats = self.backbone.forward_features(x, input_chans=self.input_chans)
        return self.classifier(feats)

# -----------------------------------------------------------------------------
# 3) Training / Evaluation
# -----------------------------------------------------------------------------
def train_epoch(model, loader, opt, sched, crit, device):
    model.train()
    total_loss = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss   = crit(logits, y)
        loss.backward()
        opt.step()
        sched.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    total_loss, correct = 0, 0
    for x,y in loader:
        x,y    = x.to(device), y.to(device)
        logits = model(x)
        total_loss += crit(logits, y).item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds==y).sum().item()
    return total_loss/len(loader.dataset), correct/len(loader.dataset)

# -----------------------------------------------------------------------------
# 4) Main
# -----------------------------------------------------------------------------
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_text_dir", required=True,
                        help="dir of .npy EEG‑text files")
    parser.add_argument("--text_csv",    required=True,
                        help="CSV of sentence→category")
    parser.add_argument("--checkpoint",  required=True,
                        help="LaBraM pretrained .pth")
    parser.add_argument("--subject_handling", choices=["stack","average"],
                        default="stack")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--orig_sfreq", type=int, default=200)
    parser.add_argument("--cuda",       type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)

    # dataset & split
    ds = EEGTextDatasetLabram(
        args.eeg_text_dir, args.text_csv,
        subject_handling=args.subject_handling,
        orig_sfreq=args.orig_sfreq
    )
    n_val = int(len(ds)*0.2)
    tr_ds, va_ds = random_split(ds, [len(ds)-n_val, n_val])
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model = LaBraMForTextClassification(
        args.checkpoint, device, ds.num_classes
    ).to(device)

    # optimizer & scheduler (3‑epoch warmup + cosine)
    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=5e-4, betas=(0.9,0.999), weight_decay=0.05
    )
    total_steps  = len(tr_loader)*args.epochs
    warmup_steps = len(tr_loader)*3
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps-warmup_steps, eta_min=1e-6)
        ],
        milestones=[warmup_steps]
    )
    criterion = nn.CrossEntropyLoss()

    # training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, tr_loader, optimizer, scheduler, criterion, device)
        va_loss, va_acc = eval_epoch(model, va_loader, criterion, device)
        print(f"Epoch {epoch:02d}  tr_loss:{tr_loss:.4f}  va_loss:{va_loss:.4f}  va_acc:{va_acc:.2%}")
        if va_acc>best_acc:
            best_acc=va_acc
            torch.save(model.state_dict(), f"best_labram_text_{args.subject_handling}.pth")

    print(f"Done! Best text val accuracy = {best_acc:.2%}")
