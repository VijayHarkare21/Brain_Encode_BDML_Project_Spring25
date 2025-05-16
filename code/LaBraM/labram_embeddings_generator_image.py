#!/usr/bin/env python3
import os
import numpy as np
import torch
from pathlib import Path
import argparse
from collections import OrderedDict

from pre_process_eeg_labram import process_eeg
from modeling_finetune import labram_base_patch200_200
import utils

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
gdrive_data_parent_dir = r"/scratch/vjh9526/bdml_2025/project/datasets/THINGS-EEG"
eeg_data_root = os.path.join(gdrive_data_parent_dir, 'eeg_dataset', 'preprocessed_data')
img_data_root = os.path.join(gdrive_data_parent_dir, 'image_set')
# Use the same directory where CBraMod saved its embeddings
embed_dir = os.path.join(gdrive_data_parent_dir, 'eeg_dataset', 'embeds')
weights_fp = r"/scratch/vjh9526/bdml_2025/project/code/LaBraM/checkpoints/labram-base.pth"
sfreq_original = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CHANNELS = ["Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2"]

# -----------------------------------------------------------------------------
# PARSE MODEL INIT ARGUMENTS (maintaining original options)
# -----------------------------------------------------------------------------
def get_model_init_args():
    parser = argparse.ArgumentParser('Extract arguments for model initialization', add_help=False)
    parser.add_argument('--model', default='labram_base_patch200_200', type=str,
                        help='Name of the model to initialize')
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='Number of classification types')
    parser.add_argument('--drop', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0,
                        help='Attention dropout rate')
    parser.add_argument('--drop_path', type=float, default=0.1,
                        help='Drop path rate')
    parser.add_argument('--drop_block_rate', default=None,
                        help='Drop block rate (not specified, set to None)')
    parser.add_argument('--use_mean_pooling', action='store_true', default=True,
                        help='Use mean pooling')
    parser.add_argument('--init_scale', type=float, default=0.001,
                        help='Initialization scale')
    parser.add_argument('--rel_pos_bias', action='store_true', default=False,
                        help='Use relative position bias')
    parser.add_argument('--abs_pos_emb', action='store_true', default=False,
                        help='Use absolute position embedding')
    parser.add_argument('--layer_scale_init_value', type=float, default=0.1,
                        help='Layer scale initialization value')
    parser.add_argument('--qkv_bias', action='store_true', default=True,
                        help='Use QKV bias')
    args, _ = parser.parse_known_args()
    return args

args = get_model_init_args()

# -----------------------------------------------------------------------------
# LOAD LABRAM MODEL WITH DIAGNOSTIC PRINTS
# -----------------------------------------------------------------------------
def load_model_from_checkpoint(checkpoint_path, device):
    model = labram_base_patch200_200(
        pretrained=True,
        num_classes=args.nb_classes if args.nb_classes > 0 else 0,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        init_values=args.layer_scale_init_value,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt = checkpoint.get('model', checkpoint)

    # Strip 'student.' prefix from keys
    stripped_map = {k.replace('student.', ''): v for k, v in ckpt.items()}
    total_ckpt_keys = list(stripped_map.keys())

    loaded_keys = []
    dropped_keys = []
    new_ckpt = OrderedDict()

    # Match shapes and load
    for name, weight in stripped_map.items():
        if name in model.state_dict() and weight.shape == model.state_dict()[name].shape:
            new_ckpt[name] = weight
            loaded_keys.append(name)
        else:
            dropped_keys.append(name)

    # Diagnostics
    print(f"Loaded {len(loaded_keys)}/{len(total_ckpt_keys)} checkpoint parameters into the model.")
    if dropped_keys:
        print("Dropped the following checkpoint parameters (mismatched or unused):")
        for k in dropped_keys:
            print("  -", k)

    missing_keys = [k for k in model.state_dict().keys() if k not in new_ckpt]
    if missing_keys:
        print(f"Model parameters not initialized from checkpoint and left at default ({len(missing_keys)}):")
        for k in missing_keys:
            print("  -", k)

    # Load what we can
    utils.load_state_dict(model, new_ckpt)
    model.eval()
    return model

model = load_model_from_checkpoint(weights_fp, device)

# -----------------------------------------------------------------------------
# EMBEDDING UTILITIES
# -----------------------------------------------------------------------------
def embed_labram(raws):
    """
    raws: list of np.ndarray, each (channels, time_points)
    returns: np.ndarray (n_samples, embed_dim)
    """
    embeds = []
    for raw in raws:
        raw_np = raw.cpu().numpy() if torch.is_tensor(raw) else raw
        raw_np = raw_np.T
        processed = process_eeg(raw_np, sfreq_original)
        processed = np.ascontiguousarray(processed)
        tensor = torch.from_numpy(processed).to(device).float()
        input_chans = utils.get_input_chans([c.upper() for c in CHANNELS])
        with torch.no_grad():
            emb = model.forward_features(tensor, input_chans=input_chans)
        emb_np = emb.cpu().numpy()
        if emb_np.ndim == 2 and emb_np.shape[0] == 1:
            emb_np = emb_np[0:1, :]
        embeds.append(emb_np)
        del tensor, emb
        torch.cuda.empty_cache()
    return np.concatenate(embeds, axis=0)

# -----------------------------------------------------------------------------
# COMPUTE LABRAM EMBEDDINGS FOR A SUBJECT/PARTITION
# -----------------------------------------------------------------------------
def compute_labram_for_partition(subj, partition_name, img_meta):
    subj_dir = os.path.join(eeg_data_root, subj)
    eeg_fp = os.path.join(subj_dir, f'preprocessed_eeg_{partition_name}.npy')
    data = np.load(eeg_fp, allow_pickle=True).item()
    eeg_mean = data['preprocessed_eeg_data'].mean(axis=1)
    files    = img_meta['train_img_files'] if partition_name=='training' else img_meta['test_img_files']
    n = eeg_mean.shape[0]
    raws = [eeg_mean[i] for i in range(n)]
    return embed_labram(raws)

# -----------------------------------------------------------------------------
# MAIN: MERGE CBraMod + LABRAM EMBEDDINGS
# -----------------------------------------------------------------------------
def main():
    img_meta_fp = os.path.join(img_data_root, 'image_metadata.npy')
    img_meta = np.load(img_meta_fp, allow_pickle=True).item()

    for embed_fp in sorted(Path(embed_dir).glob('*_*_embeds.npy')):
    # for embed_fp in sorted(os.listdir(embed_dir)):
        subj, part, _ = embed_fp.stem.split('_', 2)
        partition_name = 'training' if 'training' in embed_fp.stem else 'test'
        print(f"Loading CBraMod embeddings: {embed_fp}")
        save_dict = np.load(embed_fp, allow_pickle=True).item()

        print(f"Computing LaBraM embeddings for {subj} ({partition_name})...")
        labram_emb = compute_labram_for_partition(subj, partition_name, img_meta)
        save_dict['embeds_labram'] = labram_emb

        print(f"Saving combined embeddings back to {embed_fp}")
        np.save(embed_fp, save_dict, allow_pickle=True)

    print("All done: CBraMod + LaBraM embeddings saved.")

if __name__ == '__main__':
    main()
