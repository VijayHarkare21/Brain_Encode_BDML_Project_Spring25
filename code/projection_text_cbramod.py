#!/usr/bin/env python3
"""
train_proj_text2eeg.py  —  v2 with *optional* subject‑averaging
----------------------------------------------------------------
Learns an MLP that projects frozen **BERT‑CLS embeddings** (ℝ768)
into the **CBraMod EEG embedding space** (ℝ800).

Use the same ZuCo2 text‑EEG dataset you used for finetuning the
CBraMod encoder.  Pass `--avg_subjects` to average the EEG vectors
across subjects *before* the projector sees them (i.e.
population‑level training).  Omitting the flag keeps every
(subject × trial) sample (single‑subject training).

Example
-------
```bash
python train_proj_text2eeg.py \
  --eeg_dir   /scratch/.../ZuCo2/NR/npy_file/embeds \
  --eeg_ckpt  best_cbramod_text_model.pth \
  --model_out text2eeg_proj.pth \
  --epochs 20 --batch 128 --avg_subjects
```
"""
# -------------------------------------------------------------
# 0 · Imports
# -------------------------------------------------------------
import argparse, os, math, gc, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel

from finetune_cbramod_text import (
    CBraModTextClassifier,  EEGTextDataset, process_eeg,
    compute_max_segments,   load_eeg_text_data,
    global_keep_idx,        REGIONS
)

# -------------------------------------------------------------
# 1 · Projection head  f: ℝ768 → ℝ800
# -------------------------------------------------------------
class Text2EEG(nn.Module):
    def __init__(self, in_dim=768, out_dim=800):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------
# 2 · Dataset that yields (bert_vec, eeg_vec) pairs
# -------------------------------------------------------------
class BERT_EEG_Pairs(Dataset):
    """Pre‑computes pairs so training dataloader is fast."""
    def __init__(self, data_dicts, tokenizer, bert, eeg_enc,
                 orig_sfreq, max_segments, avg_subjects=False,
                 device="cuda"):

        self.samples = []   # will store (bert_vec, eeg_vec)
        cache = {}          # sentence → list[eeg_vec]
        bert.eval(); eeg_enc.eval()

        with torch.no_grad():
            for subj, trials in data_dicts.items():
                print(f"Encoding subject {subj}")
                for tr in trials:
                    raw = tr.get('rawData'); sent = tr.get('content')
                    if raw is None or sent is None or raw.size == 0:
                        continue

                    # ---- BERT CLS vec (once per sentence) ----
                    if sent not in cache:
                        toks = tokenizer(sent, return_tensors="pt",
                                         truncation=True, max_length=64)
                        cache[sent] = {
                            "bert": bert(**{k:v.to(device) for k,v in toks.items()})
                                          .last_hidden_state[:,0].squeeze(0).cpu(),
                            "eeg_list": []
                        }

                    # ---- EEG → z (subject‑specific) ----
                    slice_ = raw[:orig_sfreq*max_segments, global_keep_idx]
                    if slice_.shape[0] < orig_sfreq*max_segments:
                        pad = np.zeros((orig_sfreq*max_segments-slice_.shape[0],
                                        slice_.shape[1]))
                        slice_ = np.vstack([slice_, pad])
                    proc = process_eeg(slice_, orig_sfreq)
                    eeg_in = torch.from_numpy(proc).unsqueeze(0).half().to(device)
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        z,_ = eeg_enc(eeg_in)
                    cache[sent]["eeg_list"].append(z.squeeze(0).float().cpu())

        # -------- flatten according to avg_subjects flag ---------
        for sent, d in cache.items():
            if avg_subjects:
                eeg_vec = torch.stack(d["eeg_list"],0).mean(0)
                self.samples.append((d["bert"], eeg_vec))
            else:
                for ev in d["eeg_list"]:
                    self.samples.append((d["bert"], ev))

        print(f"Total paired samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# -------------------------------------------------------------
# 3 · Training routine
# -------------------------------------------------------------
@torch.no_grad()
def eval_dl(model, loader, mse, cos, device):
    model.eval(); tot_mse=tot_cos=0
    for bvec, evec in loader:
        pred = model(bvec.to(device))
        tot_mse += mse(pred, evec.to(device)).item()*bvec.size(0)
        tot_cos += cos(pred, evec.to(device)).sum().item()
    N=len(loader.dataset)
    return tot_mse/N, tot_cos/N

# -------------------------------------------------------------
# 4 · Main entry
# -------------------------------------------------------------
def main(args):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # --- raw data & segment length ---
    raw_dict = load_eeg_text_data(args.eeg_dir)
    max_seg  = compute_max_segments(raw_dict, args.eeg_dir, args.orig_sfreq)

    # --- frozen EEG encoder ---
    dummy = argparse.Namespace(channels=len(global_keep_idx), patches=max_seg,
                               d_model=200, dim_feedforward=800,
                               n_layer=12, nhead=8, dropout=0.0, num_classes=1)
    eeg_enc = CBraModTextClassifier(dummy).to(device)
    eeg_enc.load_state_dict(torch.load(args.eeg_ckpt, map_location=device))
    eeg_enc.half(); eeg_enc.eval()

    # --- frozen BERT encoder ---
    tok  = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()

    # --- build pair dataset ---
    pair_ds = BERT_EEG_Pairs(raw_dict, tok, bert, eeg_enc,
                             args.orig_sfreq, max_seg,
                             avg_subjects=args.avg_subjects,
                             device=device)
    tr_len = int(0.9*len(pair_ds)); va_len = len(pair_ds)-tr_len
    tr_ds, va_ds = random_split(pair_ds, [tr_len, va_len])
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch)

    # --- projector to train ---
    proj = Text2EEG().to(device)
    opt  = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=1e-2)
    mse  = nn.MSELoss(); cos = nn.CosineSimilarity(dim=1)

    best = 1e9
    for ep in trange(1, args.epochs+1, desc="Epoch"):
        proj.train(); tot=0
        for bvec,evec in tr_dl:
            bvec,evec = bvec.to(device), evec.to(device)
            opt.zero_grad()
            pred = proj(bvec)
            loss = mse(pred,evec)+0.5*(1-cos(pred,evec)).mean()
            loss.backward(); opt.step(); tot+=loss.item()*bvec.size(0)
        tr_mse = tot/len(tr_ds)
        va_mse, va_cos = eval_dl(proj, va_dl, mse, cos, device)
        tqdm.write(f"Ep{ep:02d}  MSE_tr {tr_mse:.4f} | MSE_va {va_mse:.4f} | cos_va {va_cos:.3f}")
        if va_mse<best:
            best=va_mse; torch.save(proj.state_dict(), args.model_out)

# -------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eeg_dir",   required=True)
    p.add_argument("--eeg_ckpt",  required=True)
    p.add_argument("--model_out", required=True)
    p.add_argument("--epochs",    type=int, default=20)
    p.add_argument("--batch",     type=int, default=128)
    p.add_argument("--lr",        type=float, default=2e-4)
    p.add_argument("--orig_sfreq",type=int, default=500)
    p.add_argument("--cuda",      type=int, default=0)
    p.add_argument("--avg_subjects", action="store_true",
                   help="average EEG embeddings across subjects before training")
    main(p.parse_args())
