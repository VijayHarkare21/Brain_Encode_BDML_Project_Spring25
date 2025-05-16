#!/usr/bin/env python3
"""
fine_tune_labram_image_contrastive.py
Fine‑tune LaBraM on THINGS image EEG (train split) using contrastive learning.
Supports two subject modes: 'stack' or 'average'.
Uses contrastive learning to create an embedding space where EEG from the same image
are closer together regardless of subject variability.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pre_process_eeg_labram import process_eeg
from eeg_embeds_utils import get_model_init_args, \
    load_model_from_checkpoint, CHANNELS, load_image_labels
import utils  # for get_input_chans

# -----------------------------------------------------------------------------
# 1) Dataset - Enhanced with image ID tracking for contrastive learning
# -----------------------------------------------------------------------------
class EEGImageDatasetLabram(Dataset):
    def __init__(self, eeg_root, img_meta_path, things_map_path,
                 subject_handling="stack", orig_sfreq=200):
        assert subject_handling in ("stack", "average")
        self.handling = subject_handling
        self.orig_sfreq = orig_sfreq
        
        # 1a) load THINGS metadata + label map
        meta = np.load(img_meta_path, allow_pickle=True).item()
        concepts = meta["train_img_concepts"]
        files    = meta["train_img_files"]
        path2label = load_image_labels(img_meta_path, things_map_path)
        
        # 1b) collect raw EEG per (cond, rep) or per-image averages
        raws, labels, image_ids = [], [], []  # Added image_ids to track same-image samples
        per_img = {i: [] for i in range(len(concepts))}
        
        for subj in sorted(os.listdir(eeg_root)):
            fp = os.path.join(eeg_root, subj, "preprocessed_eeg_training.npy")
            if not os.path.exists(fp): continue
            
            data = np.load(fp, allow_pickle=True).item()
            eeg  = data["preprocessed_eeg_data"]     # shape [n_cond, n_rep, C, T]
            n_cond, n_rep, C, T = eeg.shape
            
            if self.handling == "stack":
                for i in range(n_cond):
                    key = f"{concepts[i]}/{files[i]}".lower()
                    lbl = path2label[key]
                    for r in range(n_rep):
                        raws.append(eeg[i, r])        # (C, T)
                        labels.append(lbl)
                        image_ids.append(i)           # Track which image this EEG is from
            else:
                subj_mean = eeg.mean(axis=1)         # [n_cond, C, T]
                for i in range(n_cond):
                    per_img[i].append(subj_mean[i])
                    
        if self.handling == "average":
            for i, mats in per_img.items():
                mean_all = np.stack(mats, 0).mean(0)  # (C, T)
                raws.append(mean_all)
                key = f"{concepts[i]}/{files[i]}".lower()
                labels.append(path2label[key])
                image_ids.append(i)
                
        # 1c) finalize label→idx
        uniq = sorted(set(labels))
        self.lab2idx = {l:i for i,l in enumerate(uniq)}
        self.num_classes = len(uniq)
        
        self.raws = raws
        self.labels = [self.lab2idx[l] for l in labels]
        self.image_ids = image_ids  # Store image IDs for contrastive learning
        
    def __len__(self):
        return len(self.raws)
        
    def __getitem__(self, idx):
        raw = self.raws[idx]                  # (C, T)
        proc = process_eeg(raw.T, self.orig_sfreq)  # → [1, C, n_seg, 200]
        x    = torch.from_numpy(proc).float().squeeze(0)  # [C, n_seg, 200]
        y    = self.labels[idx]
        img_id = self.image_ids[idx]
        
        return x, y, img_id  # Return image ID alongside data and label

# -----------------------------------------------------------------------------
# 2) Fine‑tuning module with projector for contrastive learning
# -----------------------------------------------------------------------------
class LaBraMForContrastiveImageClassification(nn.Module):
    def __init__(self, checkpoint_path, device, num_classes, embed_dim=200, proj_dim=512):
        super().__init__()
        # 2a) load backbone exactly as in your embedding script
        init_args = get_model_init_args()  
        self.backbone = load_model_from_checkpoint(checkpoint_path, device)
        
        # 2b) fix in‑channels selection
        self.input_chans = utils.get_input_chans([c.upper() for c in CHANNELS])
        
        # 2c) embedding dimension from backbone
        self.embed_dim = self.backbone.embed_dim
        
        # 2d) projection head for contrastive learning 
        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, proj_dim)
        )
        
        # 2e) classifier head
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, x, get_embeddings=False, get_projections=False):
        # x: [B, C, n_seg, 200]
        # Process through backbone to get embeddings
        embeddings = self.backbone(x, input_chans=self.input_chans)  # [B, embed_dim]
        
        # For contrastive loss, get normalized projections
        projections = F.normalize(self.projector(embeddings), dim=1)
        
        # For classification task
        logits = self.classifier(embeddings)
        
        if get_embeddings and get_projections:
            return logits, embeddings, projections
        elif get_embeddings:
            return logits, embeddings
        elif get_projections:
            return logits, projections
        else:
            return logits

# -----------------------------------------------------------------------------
# 3) Contrastive Loss Implementation - NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)
# -----------------------------------------------------------------------------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
    def forward(self, projections, image_ids):
        """
        projections: normalized embeddings [batch_size, proj_dim]
        image_ids: image identifiers to determine positive pairs [batch_size]
        """
        device = projections.device
        batch_size = projections.shape[0]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # Create mask for positive pairs (same image_id but different indices)
        image_ids = image_ids.view(-1, 1)
        pos_mask = (image_ids == image_ids.T).float()
        # Remove self-similarity from positive pairs
        pos_mask.fill_diagonal_(0)
        
        # For each anchor, find its positive pairs
        # If an anchor has no positive pairs, we'll skip it in the loss calculation
        loss = 0
        valid_anchors = 0
        
        for anchor_idx in range(batch_size):
            pos_indices = pos_mask[anchor_idx].nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0:
                continue  # Skip anchors with no positive pairs
                
            valid_anchors += 1
            
            # For each anchor, all other samples are treated as negatives
            # except those with the same image_id
            neg_mask = torch.ones(batch_size, device=device)
            neg_mask[anchor_idx] = 0  # exclude self
            neg_mask[pos_indices] = 0  # exclude positives
            neg_indices = neg_mask.nonzero(as_tuple=True)[0]
            
            # If there are no negatives, skip this anchor
            if len(neg_indices) == 0:
                valid_anchors -= 1
                continue
                
            # Prepare logits and targets for this anchor
            logits = sim_matrix[anchor_idx]
            
            # Construct target: for each positive pair
            for pos_idx in pos_indices:
                # Create anchor-positive pair
                pos_logit = logits[pos_idx].unsqueeze(0)
                # All logits for this anchor
                pair_logits = torch.cat([pos_logit, logits[neg_indices]])
                
                # Target is always 0 (the positive sample is at index 0)
                target = torch.zeros(1, device=device, dtype=torch.long)
                
                # Calculate loss for this positive pair
                loss += self.criterion(pair_logits.unsqueeze(0), target)
        
        # Average loss over valid anchors and their positive pairs
        if valid_anchors > 0:
            loss = loss / valid_anchors
        else:
            # If no valid anchors (unusual case), return zero loss
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        return loss

# -----------------------------------------------------------------------------
# 4) Combined Loss Function
# -----------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    def __init__(self, temperature=0.5, contrastive_weight=0.5):
        super().__init__()
        self.contrastive_loss = NTXentLoss(temperature)
        self.classification_loss = nn.CrossEntropyLoss()
        self.contrastive_weight = contrastive_weight
        
    def forward(self, logits, labels, projections, image_ids):
        # Classification loss
        cls_loss = self.classification_loss(logits, labels)
        
        # Contrastive loss
        con_loss = self.contrastive_loss(projections, image_ids)
        
        # Combine losses
        total_loss = (1 - self.contrastive_weight) * cls_loss + self.contrastive_weight * con_loss
        
        return total_loss, cls_loss, con_loss

# -----------------------------------------------------------------------------
# 5) Train / eval loops
# -----------------------------------------------------------------------------
def train_epoch(model, loader, opt, sched, crit, device, contrastive_weight):
    model.train()
    total_loss, cls_total, con_total = 0.0, 0.0, 0.0
    
    for x, y, img_ids in loader:
        x, y = x.to(device), y.to(device)
        img_ids = img_ids.to(device)
        
        opt.zero_grad()
        logits, projections = model(x, get_projections=True)
        
        loss, cls_loss, con_loss = crit(logits, y, projections, img_ids)
        
        loss.backward()
        opt.step()
        sched.step()
        
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        cls_total += cls_loss.item() * batch_size
        con_total += con_loss.item() * batch_size
        
    n = len(loader.dataset)
    return total_loss / n, cls_total / n, con_total / n

@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    total_loss, cls_total, con_total = 0.0, 0.0, 0.0
    correct = 0
    
    for x, y, img_ids in loader:
        x, y = x.to(device), y.to(device)
        img_ids = img_ids.to(device)
        
        logits, projections = model(x, get_projections=True)
        
        loss, cls_loss, con_loss = crit(logits, y, projections, img_ids)
        
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        cls_total += cls_loss.item() * batch_size
        con_total += con_loss.item() * batch_size
        
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        
    n = len(loader.dataset)
    return total_loss / n, cls_total / n, con_total / n, correct / n

# -----------------------------------------------------------------------------
# 6) Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_root", required=True,
                        help="EEG root dir w/ preprocessed_eeg_training.npy per subject")
    parser.add_argument("--img_meta", required=True,
                        help="path/to/image_metadata.npy")
    parser.add_argument("--things_map", required=True,
                        help="path/to/things_map.tsv")
    parser.add_argument("--checkpoint", required=True,
                        help="LaBraM pretrained checkpoint (.pth)")
    parser.add_argument("--subject_handling", choices=["stack","average"],
                        default="stack")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--orig_sfreq", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature scaling for contrastive loss")
    parser.add_argument("--contrastive_weight", type=float, default=0.5, 
                        help="Weight of contrastive loss vs classification loss")
    parser.add_argument("--proj_dim", type=int, default=128,
                        help="Projection dimension for contrastive learning")
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()
    
    # Reproducibility
    torch.manual_seed(0); np.random.seed(0)
    
    # Dataset & split
    ds = EEGImageDatasetLabram(
        args.eeg_root, args.img_meta, args.things_map,
        subject_handling=args.subject_handling,
        orig_sfreq=args.orig_sfreq
    )
    
    n_val = int(len(ds) * 0.2)
    n_tr = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])
    
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size)
    
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model = LaBraMForContrastiveImageClassification(
        args.checkpoint, device, ds.num_classes, proj_dim=args.proj_dim
    ).to(device)
    
    # Criterion
    criterion = CombinedLoss(
        temperature=args.temperature,
        contrastive_weight=args.contrastive_weight
    )
    
    # Optimizer & scheduler - now training both projector and classifier
    optimizer = optim.AdamW(
        list(model.projector.parameters()) + list(model.classifier.parameters()),
        lr=5e-4, betas=(0.9, 0.999), weight_decay=0.05
    )
    
    total_steps = len(tr_loader) * args.epochs
    warmup_steps = len(tr_loader) * 3
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
    
    # Training loop
    best_acc = 0.0
    print(f"Starting training with contrastive learning (weight={args.contrastive_weight}, temp={args.temperature})")
    
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_cls, tr_con = train_epoch(
            model, tr_loader, optimizer, scheduler, criterion, device, args.contrastive_weight
        )
        va_loss, va_cls, va_con, va_acc = eval_epoch(model, va_loader, criterion, device)
        
        print(f"Epoch {epoch:02d} | Train: loss={tr_loss:.4f} cls={tr_cls:.4f} con={tr_con:.4f} | "
              f"Val: loss={va_loss:.4f} cls={va_cls:.4f} con={va_con:.4f} acc={va_acc:.2%}")
        
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': va_acc,
                'embed_dim': model.embed_dim,
                'proj_dim': args.proj_dim,
                'temperature': args.temperature,
                'contrastive_weight': args.contrastive_weight
            }, f"best_labram_image_contrastive_{args.subject_handling}.pth")
    
    print(f"Finished! Best val accuracy = {best_acc:.2%}")

# -----------------------------------------------------------------------------
# 7) Embedding Visualization Helper (can be run separately)
# -----------------------------------------------------------------------------
def visualize_embeddings(model_path, dataset, device, output_file='embeddings_plot.png'):
    """
    Visualize the learned embedding space using t-SNE.
    Color points by image ID to show how well contrastive learning worked.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        
        # Load saved model
        checkpoint = torch.load(model_path, map_location=device)
        
        # Recreate model architecture
        model = LaBraMForContrastiveImageClassification(
            args.checkpoint, device, dataset.num_classes,
            embed_dim=checkpoint['embed_dim'],
            proj_dim=checkpoint['proj_dim']
        ).to(device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create DataLoader with no shuffling to keep track of indices
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        # Collect embeddings
        all_embeddings = []
        all_img_ids = []
        all_labels = []
        
        with torch.no_grad():
            for x, y, img_id in loader:
                x = x.to(device)
                _, embeddings = model(x, get_embeddings=True)
                all_embeddings.append(embeddings.cpu().numpy())
                all_img_ids.extend(img_id.numpy())
                all_labels.extend(y.numpy())
                
        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot - color by image ID
        plt.figure(figsize=(12, 10))
        
        # Convert image IDs to a categorical colormap
        unique_ids = np.unique(all_img_ids)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_ids)))
        id_to_color = {id: colors[i] for i, id in enumerate(unique_ids)}
        
        # Plot points
        for img_id in unique_ids:
            mask = np.array(all_img_ids) == img_id
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                color=id_to_color[img_id], 
                alpha=0.7,
                s=30,
                label=f"Image {img_id}" if img_id < 10 else None  # Limit legend entries
            )
            
        plt.title(f"t-SNE Visualization of Contrastive Embeddings\nColored by Image ID")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        
        # Add legend for the first few images only to avoid clutter
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), ncol=1)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Embeddings visualization saved to {output_file}")
        
    except ImportError:
        print("Visualization requires matplotlib and scikit-learn. Please install with:")
        print("pip install matplotlib scikit-learn")