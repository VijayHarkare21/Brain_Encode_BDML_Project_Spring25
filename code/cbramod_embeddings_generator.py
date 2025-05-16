#!/usr/bin/env python3
import os
import numpy as np
import torch
from cbramod import CBraMod
from pre_process_eeg_cbramod import process_eeg

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
input_dir  = r"/scratch/vjh9526/bdml_2025/project/datasets/ZuCo2/2urht/osfstorage/task1 - NR/npy_file"
embed_dir  = os.path.join(input_dir, 'embeds')
weights_fp = r"/scratch/vjh9526/bdml_2025/project/code/CBraMod/pretrained_weights/pretrained_weights.pth"
sfreq_original = 500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
model = CBraMod().to(device)
model.load_state_dict(torch.load(weights_fp, map_location=device))
model.eval()

# -----------------------------------------------------------------------------
# POOLING UTILS
# -----------------------------------------------------------------------------
def fixed_attention_pool(feats):
    # feats: [B, C, P, D] -> [B, D]
    scores = feats.norm(dim=-1, keepdim=True)            # [B, C, P, 1]
    w = torch.softmax(scores.view(feats.size(0), -1), dim=-1).view_as(scores)
    return (feats * w).sum(dim=(1, 2))

# -----------------------------------------------------------------------------
# EMBEDDING
# -----------------------------------------------------------------------------
def embed_subject(raws):
    embeds = []
    for raw in raws:
        if raw is None:
            continue
        # ensure numpy array
        raw_np = raw.cpu().numpy() if torch.is_tensor(raw) else raw
        # preprocess EEG
        processed = process_eeg(raw_np, sfreq_original)
        processed = np.ascontiguousarray(processed)
        prepped = torch.from_numpy(processed).to(device).float()
        print(prepped.shape)
        with torch.no_grad():
            feats     = model(prepped)                       # [B, C, P, D]
            mean_pool = feats.mean(dim=(1, 2))               # [B, D]
            max_pool  = feats.amax(dim=(1, 2))               # [B, D]
            attn_pool = fixed_attention_pool(feats)          # [B, D]
            embed     = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)

        embed_np = embed.detach().cpu().numpy()
        embeds.append(embed_np)
        print(embed.shape)

        del prepped, feats, embed
        torch.cuda.empty_cache()
    return embeds
    
chanlocs = ['E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10', 'E11', 'E12', 'E13', 'E15', 'E16', 'E18', 'E19', 'E20',
            'E22',
            'E23', 'E24', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39',
            'E40',
            'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58',
            'E59',
            'E60', 'E61', 'E62', 'E64', 'E65', 'E66', 'E67', 'E69', 'E70', 'E71', 'E72', 'E74', 'E75', 'E76', 'E77',
            'E78',
            'E79', 'E80', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E89', 'E90', 'E91', 'E92', 'E93', 'E95', 'E96',
            'E97',
            'E98', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E108', 'E109', 'E110', 'E111', 'E112',
            'E114',
            'E115', 'E116', 'E117', 'E118', 'E120', 'E121', 'E122', 'E123', 'E124', 'Cz']
            
keep_channels = [
    # Left frontal – Broca
    'E13','E19','E20','E22','E23','E24','E26','E27','E28','E29','E33','E34','E38','E43',
    # Left temporal / parietal – Wernicke / SMG
    'E30','E31','E35','E36','E37','E39','E40','E41','E44',
    # Left occipito-temporal – VWFA & visual word-form N170
    'E42','E45','E46','E47','E50','E51','E52','E53','E54',
    'E57','E58','E59','E60','E61','E64','E65','E66','E69','E70',
    # Midline centro-parietal – N400 / P600 global integration
    'E5','E6','E7','E10','E11','E12','E15','E16','E18',
    'E55','E62','E67','E71','E72','E75','E76','E77','E106','Cz'
]

# 1) indices of channels we want to keep, *in the order listed above*
global_keep_idx = [chanlocs.index(ch) for ch in keep_channels]


# -----------------------------------------------------------------------------
# PROCESS SINGLE SUBJECT'S DATA LIST
# -----------------------------------------------------------------------------
def process_subject_list(data_list):
    valid_raws = []
    valid_idxs = []
    valid_contents = []

    for i, item in enumerate(data_list):
        if not item or 'rawData' not in item:
            continue
        raw = item['rawData']
        content = item['content']
        if raw is None or not isinstance(raw, (np.ndarray,)) or raw.size == 0 or len(raw.shape) == 0:
            continue
        
        print(raw.shape)
        
        # 2) slice the data
        eeg_subset = raw[:, global_keep_idx]
        
        # eeg_subset now has shape (61, time_points)
        print(eeg_subset.shape)
        valid_raws.append(eeg_subset)
        valid_idxs.append(i)
        valid_contents.append(content)

    if not valid_raws:
        return data_list

    embeds = embed_subject(valid_raws)
    for idx, emb, con in zip(valid_idxs, embeds, valid_contents):
        data_list[idx]['embeds_cbramod'] = emb
        data_list[idx]['content'] = con
    return data_list

# -----------------------------------------------------------------------------
# MAIN LOOP OVER ALL .npy FILES
# -----------------------------------------------------------------------------
def main():
    os.makedirs(embed_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):
        in_path = os.path.join(input_dir, fname)
        # skip directories or non-.npy
        if not os.path.isfile(in_path) or not fname.endswith('.npy'):
            continue

        subj = fname.replace('.npy', '')
        out_fname = f"{subj}_embeds.npy"
        out_path = os.path.join(embed_dir, out_fname)

        # if os.path.exists(out_path):
        #     print(f"SKIPPING existing embeddings: {out_path}")
        #     continue

        print(f"Loading: {in_path}")
        data_dict = np.load(in_path, allow_pickle=True).item()
        # assume single key per file
        subject_key = next(iter(data_dict))
        print(f"Processing subject: {subject_key}")

        data_list = data_dict[subject_key]
        data_dict[subject_key] = process_subject_list(data_list)

        print(f"Saving embeddings to: {out_path}")
        np.save(out_path, data_dict, allow_pickle=True)

    print("All done.")

if __name__ == '__main__':
    main()
