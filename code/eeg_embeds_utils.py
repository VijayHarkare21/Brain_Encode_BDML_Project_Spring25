#!/usr/bin/env python3
import os
import numpy as np
import torch
from pathlib import Path
import argparse
from collections import OrderedDict

import pickle
import hashlib
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# from pre_process_eeg_labram import process_eeg
# from modeling_finetune import labram_base_patch200_200
# import utils

CHANNELS = ["Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2"]

# def get_model_init_args():
#     parser = argparse.ArgumentParser('Extract arguments for model initialization', add_help=False)
#     parser.add_argument('--model', default='labram_base_patch200_200', type=str,
#                         help='Name of the model to initialize')
#     parser.add_argument('--nb_classes', default=0, type=int,
#                         help='Number of classification types')
#     parser.add_argument('--drop', type=float, default=0.0,
#                         help='Dropout rate')
#     parser.add_argument('--attn_drop_rate', type=float, default=0.0,
#                         help='Attention dropout rate')
#     parser.add_argument('--drop_path', type=float, default=0.1,
#                         help='Drop path rate')
#     parser.add_argument('--drop_block_rate', default=None,
#                         help='Drop block rate (not specified, set to None)')
#     parser.add_argument('--use_mean_pooling', action='store_true', default=True,
#                         help='Use mean pooling')
#     parser.add_argument('--init_scale', type=float, default=0.001,
#                         help='Initialization scale')
#     parser.add_argument('--rel_pos_bias', action='store_true', default=False,
#                         help='Use relative position bias')
#     parser.add_argument('--abs_pos_emb', action='store_true', default=False,
#                         help='Use absolute position embedding')
#     parser.add_argument('--layer_scale_init_value', type=float, default=0.1,
#                         help='Layer scale initialization value')
#     parser.add_argument('--qkv_bias', action='store_true', default=True,
#                         help='Use QKV bias')
#     args, _ = parser.parse_known_args()
#     return args
    
# def load_model_from_checkpoint(checkpoint_path, device):
#     model = labram_base_patch200_200(
#         pretrained=True,
#         num_classes=args.nb_classes if args.nb_classes > 0 else 0,
#         drop_rate=args.drop,
#         drop_path_rate=args.drop_path,
#         attn_drop_rate=args.attn_drop_rate,
#         drop_block_rate=None,
#         use_mean_pooling=args.use_mean_pooling,
#         init_scale=args.init_scale,
#         init_values=args.layer_scale_init_value,
#     ).to(device)

#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     ckpt = checkpoint.get('model', checkpoint)

#     # Strip 'student.' prefix from keys
#     stripped_map = {k.replace('student.', ''): v for k, v in ckpt.items()}
#     total_ckpt_keys = list(stripped_map.keys())

#     loaded_keys = []
#     dropped_keys = []
#     new_ckpt = OrderedDict()

#     # Match shapes and load
#     for name, weight in stripped_map.items():
#         if name in model.state_dict() and weight.shape == model.state_dict()[name].shape:
#             new_ckpt[name] = weight
#             loaded_keys.append(name)
#         else:
#             dropped_keys.append(name)

#     # Diagnostics
#     print(f"Loaded {len(loaded_keys)}/{len(total_ckpt_keys)} checkpoint parameters into the model.")
#     if dropped_keys:
#         print("Dropped the following checkpoint parameters (mismatched or unused):")
#         for k in dropped_keys:
#             print("  -", k)

#     missing_keys = [k for k in model.state_dict().keys() if k not in new_ckpt]
#     if missing_keys:
#         print(f"Model parameters not initialized from checkpoint and left at default ({len(missing_keys)}):")
#         for k in missing_keys:
#             print("  -", k)

#     # Load what we can
#     utils.load_state_dict(model, new_ckpt)
#     model.eval()
#     return model
    
def load_image_labels(metadata_path: str, things_map_path: str) -> Dict[str, str]:
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


# def load_text_labels(csv_path: str) -> Dict[str, str]:
#     """Load text labels from CSV file."""
#     print(f"[INFO] Loading text labels from {csv_path}")
#     df = pd.read_csv(csv_path, delimiter=";")
#     # Map sentences to their categories
#     return dict(zip(df.iloc[:, 2], df['category']))
def load_text_labels(csv_path: str) -> Dict[str, str]:
    print(f"[INFO] Loading text labels from {csv_path}")
    df = pd.read_csv(csv_path, delimiter=";")
    # assume the sentence is in column 2
    sentences = df.iloc[:, 2].astype(str).str.strip().str.lower()
    categories = df["category"].astype(str).str.strip()
    return dict(zip(sentences, categories))
    
import gc
def load_eeg_text_data(directory: str) -> Dict[str, List[dict]]:
    """Load EEG text data from .npy files in a directory."""
    print(f"[INFO] Loading EEG text data from: {directory}")
    data = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.npy'):
            continue
        
        print(f"[INFO]   Loading file: {fname}")
        arr = np.load(os.path.join(directory, fname), allow_pickle=True).item()
        subject = next(iter(arr.keys()))
        data[subject] = arr[subject]
        # del arr
        # gc.collect()
    return data