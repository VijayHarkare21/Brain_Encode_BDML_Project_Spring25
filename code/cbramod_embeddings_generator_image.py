#!/usr/bin/env python3
import os
import numpy as np
import torch
from cbramod import CBraMod
from pre_process_eeg_cbramod import process_eeg

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Root directory containing 'eeg_dataset/preprocessed_data' and 'image_set'
gdrive_data_parent_dir = r"/scratch/vjh9526/bdml_2025/project/datasets/THINGS-EEG"
eeg_data_root = os.path.join(gdrive_data_parent_dir, 'eeg_dataset', 'preprocessed_data')
img_data_root = os.path.join(gdrive_data_parent_dir, 'image_set')
# Directory where embeddings will be saved
embed_dir = os.path.join(gdrive_data_parent_dir, 'eeg_dataset', 'embeds')
# Pretrained weights for CBraMod
weights_fp = r"/scratch/vjh9526/bdml_2025/project/code/CBraMod/pretrained_weights/pretrained_weights.pth"
# Original sampling frequency (for process_eeg)
sfreq_original = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load CBraMod model
model = CBraMod().to(device)
model.load_state_dict(torch.load(weights_fp, map_location=device))
model.eval()

# -----------------------------------------------------------------------------
# POOLING UTILITIES
# -----------------------------------------------------------------------------
def fixed_attention_pool(feats):
    """
    feats: [B, C, P, D] tensor -> pooled [B, D]
    """
    scores = feats.norm(dim=-1, keepdim=True)  # [B, C, P, 1]
    w = torch.softmax(scores.view(feats.size(0), -1), dim=-1).view_as(scores)
    return (feats * w).sum(dim=(1, 2))

# -----------------------------------------------------------------------------
# EMBEDDING UTILS
# -----------------------------------------------------------------------------
def embed_subject(raws):
    """
    raw inputs: list of numpy arrays, each shape (channels, time_points)
    returns: numpy array of shape (n_samples, embedding_dim)
    """
    embeds = []

    for raw in raws:
        raw_np = raw.cpu().numpy() if torch.is_tensor(raw) else raw
        raw_np = raw_np.T
        # Ensure contiguous
        processed = process_eeg(raw_np, sfreq_original)
        processed = np.ascontiguousarray(processed)
        prepped = torch.from_numpy(processed).to(device).float()

        with torch.no_grad():
            feats = model(prepped)                     # [B, C, P, D]
            mean_pool = feats.mean(dim=(1, 2))         # [B, D]
            max_pool = feats.amax(dim=(1, 2))          # [B, D]
            attn_pool = fixed_attention_pool(feats)    # [B, D]
            embed = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)  # [B, 3D]

        embed_np = embed.detach().cpu().numpy()
        embeds.append(embed_np)  # list of shape (B, 3D) arrays

        # free GPU memory
        del prepped, feats, embed
        torch.cuda.empty_cache()

    # concatenate all embeddings along batch dimension
    all_embeds = np.concatenate(embeds, axis=0)
    return all_embeds

# -----------------------------------------------------------------------------
# PROCESS ONE PARTITION (TRAINING OR TEST)
# -----------------------------------------------------------------------------
def process_partition(eeg_fp, img_meta, partition_name):
    """
    eeg_fp: path to .npy for one partition
    img_meta: loaded image_metadata dict
    partition_name: 'training' or 'test'
    """
    data = np.load(eeg_fp, allow_pickle=True).item()
    eeg_data = data['preprocessed_eeg_data']  # shape: (n_conditions, n_reps, 17, 200)

    # average over EEG repetitions (axis=1)
    eeg_mean = eeg_data.mean(axis=1)         # shape: (n_conditions, 17, 200)

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
        f"Mismatch in number of conditions and image metadata for {partition_name}"

    # build raws and paths
    raws = []
    img_paths = []
    for idx in range(n_conditions):
        raw_i = eeg_mean[idx]  # (17,200)
        raws.append(raw_i)
        # full image path
        img_paths.append(
            os.path.join(img_data_root, img_folder,
                         img_concepts[idx], img_files[idx])
        )

    # compute embeddings
    embeddings = embed_subject(raws)  # (n_conditions, embedding_dim)

    # pack into save dict
    save_dict = {
        'embeds_cbramod': embeddings,
        'img_paths': img_paths,
        'ch_names': data['ch_names'],
        'times': data['times']
    }
    return save_dict

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    os.makedirs(embed_dir, exist_ok=True)

    # load image metadata once
    img_meta_fp = os.path.join(img_data_root, 'image_metadata.npy')
    img_meta = np.load(img_meta_fp, allow_pickle=True).item()

    # iterate over subjects in preprocessed_data
    for subj in sorted(os.listdir(eeg_data_root)):
        subj_dir = os.path.join(eeg_data_root, subj)
        if not os.path.isdir(subj_dir):
            continue

        # process both training and test for each subject
        for partition in ['preprocessed_eeg_training.npy', 'preprocessed_eeg_test.npy']:
            eeg_fp = os.path.join(subj_dir, partition)
            if not os.path.exists(eeg_fp):
                continue
            part_name = 'training' if 'training' in partition else 'test'
            out_fname = f"{subj}_{part_name}_embeds.npy"
            out_fp = os.path.join(embed_dir, out_fname)

            if os.path.exists(out_fp):
                print(f"SKIPPING existing embeddings: {out_fp}")
                continue

            print(f"Processing {eeg_fp} ({part_name})...")
            save_dict = process_partition(eeg_fp, img_meta, part_name)

            print(f"Saving embeddings to: {out_fp}")
            np.save(out_fp, save_dict, allow_pickle=True)

    print("All done.")

if __name__ == '__main__':
    main()
