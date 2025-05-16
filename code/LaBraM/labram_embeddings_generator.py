#!/usr/bin/env python
# ------------------------------------------------------------------
#  LaBraM embedding pipeline  (DEBUG build)
#  * 61 measured HydroCel-129 electrodes + Cz
#  * nearest-neighbour 10-20 ROI, spline-interpolated
#  * ≤16×200-sample windows (LaBraM-base spec)
#  This version prints key shapes and channel lists at every step.
# ------------------------------------------------------------------
import argparse
from pathlib import Path
import numpy as np
import torch
import mne

from pre_process_eeg_labram import process_eeg
from modeling_finetune import labram_base_patch200_200
from utils import get_input_chans, load_state_dict           # LaBraM repo utils

DEBUG = True          # flip to False for silent run
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sfreq_original = 500.0

DATA_DIR  = Path("/scratch/vjh9526/bdml_2025/project/datasets/ZuCo2/2urht/osfstorage/task1 - NR/npy_file/embeds")
CKPT_PATH = Path("/scratch/vjh9526/bdml_2025/project/code/LaBraM/checkpoints/labram-base.pth")

# ────────────────────────────────  channel sets  ────────────────────────────────
KEEP_HYDRO = [
    'E13','E19','E20','E22','E23','E24','E26','E27','E28','E29','E33','E34','E38','E43',
    'E30','E31','E35','E36','E37','E39','E40','E41','E44',
    'E42','E45','E46','E47','E50','E51','E52','E53','E54',
    'E57','E58','E59','E60','E61','E64','E65','E66','E69','E70',
    'E5','E6','E7','E10','E11','E12','E15','E16','E18',
    'E55','E62','E67','E71','E72','E75','E76','E77','E106',
    'Cz'
]

CHANLOCS_105 = [  # original recording order (ZuCo-2)
    'E2','E3','E4','E5','E6','E7','E9','E10','E11','E12','E13','E15','E16','E18','E19','E20',
    'E22','E23','E24','E26','E27','E28','E29','E30','E31','E33','E34','E35','E36','E37','E38',
    'E39','E40','E41','E42','E43','E44','E45','E46','E47','E50','E51','E52','E53','E54','E55',
    'E57','E58','E59','E60','E61','E62','E64','E65','E66','E67','E69','E70','E71','E72','E74',
    'E75','E76','E77','E78','E79','E80','E82','E83','E84','E85','E86','E87','E89','E90','E91',
    'E92','E93','E95','E96','E97','E98','E100','E101','E102','E103','E104','E105','E106','E108',
    'E109','E110','E111','E112','E114','E115','E116','E117','E118','E120','E121','E122','E123',
    'E124','Cz'
]
IDX_61 = [CHANLOCS_105.index(ch) for ch in KEEP_HYDRO]

# ───────────────────────────────  ROI mapping (once) ────────────────────────────
# Find nearest 10-20 channels for each HydroCel channel, then unique-preserve order
_hydro, _std20 = (mne.channels.make_standard_montage(m)
                  for m in ("GSN-HydroCel-129", "standard_1020"))
ph, ps = (_m.get_positions()["ch_pos"] for _m in (_hydro, _std20))
roi20 = []
for ch in KEEP_HYDRO:
    p = ph[ch]
    nearest = min(_std20.ch_names, key=lambda s: np.linalg.norm(ps[s] - p))
    roi20.append(nearest)
ROI_1020 = []
seen = set()
for ch in roi20:
    if ch not in seen:
        seen.add(ch)
        ROI_1020.append(ch)
if DEBUG:
    print(f"ROI 10-20 ({len(ROI_1020)} ch): {ROI_1020}")
del _hydro, _std20, ph, ps, roi20

# ────────────────────────────  visualisation helpers  ─────────────────────────────
import matplotlib
matplotlib.use("Agg")            # head-less: create files, never pop a window
import matplotlib.pyplot as plt


def _save_montage_with_highlights(montage, highlight, out_file,
                                  title=None, size=6, dot=120):
    pos3d = montage.get_positions()["ch_pos"]
    names = montage.ch_names
    xy = np.array([pos3d[ch][:2] for ch in names])  # drop z
    x, y = xy[:, 0], xy[:, 1]

    fig, ax = plt.subplots(figsize=(size, size), dpi=dot)
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    ax.scatter(x, y, s=14, color="#bbbbbb", zorder=1)
    mask = [ch in highlight for ch in names]
    ax.scatter(x[mask], y[mask], s=40, color="#e31a1c", zorder=2)

    for xx, yy, ch, m in zip(x, y, names, mask):
        if m:
            ax.text(xx, yy, ch, color="#e31a1c", fontsize=6,
                    ha="center", va="center", zorder=3)
    if title:
        ax.set_title(title, fontsize=10)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    if DEBUG:
        print(f"saved {out_file.resolve()}")


def save_roi_visualisations(save_dir="figs"):
    """
    Writes two images into *save_dir*:
        hydro_roi.png – 129-ch HydroCel + highlighted KEEP_HYDRO
        std20_roi.png – 10-20 montage   + highlighted ROI_1020
    """
    save_dir = Path(save_dir)
    hydro = mne.channels.make_standard_montage("GSN-HydroCel-129")
    std20 = mne.channels.make_standard_montage("standard_1020")

    _save_montage_with_highlights(
        hydro, KEEP_HYDRO,
        save_dir / "hydro_roi.png",
        title="HydroCel-129 – recorded electrodes"
    )
    _save_montage_with_highlights(
        std20, ROI_1020,
        save_dir / "std20_roi.png",
        title="Standard-10-20 – ROI channels"
    )

# ───────────────────────────────  interpolation helper  ───────────────────────────
def interpolate_h61_to_roi20(
    eeg61: np.ndarray,
    in_names: list,
    sfreq: float,
    roi: list = ROI_1020
):
    """
    Interpolate 61-channel HydroCel EEG onto the predefined 10-20 ROI channels.

    Returns:
    -------
    data  : np.ndarray, shape (n_roi, n_times)
    roi   : list of channel names (ROI_1020)
    """
    # 1) Load montages and positions
    hydro = mne.channels.make_standard_montage("GSN-HydroCel-129")
    std20 = mne.channels.make_standard_montage("standard_1020")
    pos_hydro = hydro.get_positions()["ch_pos"]
    pos_std20 = std20.get_positions()["ch_pos"]

    # 2) Build Raw with measured data
    info = mne.create_info(in_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg61.T, info, verbose="ERROR")
    raw.set_montage(hydro, match_case=False)

    # 3) Add zero data for any ROI channels not measured
    missing = [ch for ch in roi if ch not in in_names]
    if missing:
        zeros = np.zeros((len(missing), raw.n_times), dtype=raw._data.dtype)
        mi    = mne.create_info(missing, sfreq, ch_types="eeg")
        raw.add_channels([mne.io.RawArray(zeros, mi)], force_update_info=True)

    # 4) Assign combined montage so each channel has correct 3D loc
    all_pos = {**pos_hydro, **pos_std20}
    combined_montage = mne.channels.make_dig_montage(
        ch_pos=all_pos,
        coord_frame='head'
    )
    raw.set_montage(combined_montage, on_missing="ignore")

    # 5) Interpolate the missing channels
    raw.info["bads"] = missing
    raw.interpolate_bads(reset_bads=True, mode="accurate")

    # 6) Reorder to ROI list
    raw.reorder_channels(roi)

    # 7) Return interpolated data (n_roi, n_times) and ROI names
    return raw.get_data(), roi

# ─────────────── model initialization & loading (unchanged) ──────────────────
def get_model_init_args():
    parser = argparse.ArgumentParser('Extract arguments for model initialization', add_help=False)
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, help='Name of the model to train')
    parser.add_argument('--nb_classes', default=0, type=int, help='Number of classification types')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate')
    parser.add_argument('--drop_block_rate', default=None, help='Drop block rate (not specified, set to None)')
    parser.add_argument('--use_mean_pooling', action='store_true', default=True, help='Use mean pooling')
    parser.add_argument('--init_scale', type=float, default=0.001, help='Initialization scale')
    parser.add_argument('--rel_pos_bias', action='store_true', default=False, help='Use relative position bias')
    parser.add_argument('--abs_pos_emb', action='store_true', default=False, help='Use absolute position embedding')
    parser.add_argument('--layer_scale_init_value', type=float, default=0.1, help='Layer scale initialization value')
    parser.add_argument('--qkv_bias', action='store_true', default=True, help='Use QKV bias')
    args, _ = parser.parse_known_args()
    return args


def load_model(ckpt, device):
    args = get_model_init_args()
    model = labram_base_patch200_200(
        pretrained=True,
        num_classes=6,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        init_values=args.layer_scale_init_value,
    ).to(device)
    ckpt = torch.load(ckpt, map_location=device)['model']
    state = {k.replace('student.', ''): v for k, v in ckpt.items() if k.startswith('student.')}
    for k in ('head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'):
        state.pop(k, None)
    load_state_dict(model, state)
    model.eval()
    if DEBUG:
        print("✓ LaBraM model loaded")
    return model

model = load_model(CKPT_PATH, device)

# ─────────────────────────────── per-subject processing ─────────────────────────
def process_subject(subj_dict):
    for t, trial in enumerate(subj_dict):
        if not trial:
            print(f"skipping trial {t} ...")
            continue
        raw_data = trial.get("rawData", None)
        if raw_data is None:
            print(f"skipping trial {t} ...")
            continue
        raw_np = raw_data.cpu().numpy() if torch.is_tensor(raw_data) else raw_data
        if not isinstance(raw_data, (np.ndarray,)) or raw_data.size == 0 or len(raw_data.shape) == 0:
            print(f"skipping trial {t} ...")
            continue
        if DEBUG:
            print(f"\n⋯ trial {t}  raw shape {raw_np.shape}")
        eeg61 = raw_np[:, IDX_61]                         # (time, 61)
        data_roi, roi_names = interpolate_h61_to_roi20(eeg61, KEEP_HYDRO, sfreq_original)
        if DEBUG:
            print(f"    after spline  {data_roi.shape[0]}×{data_roi.shape[1]}  (ROI)")
        eeg_roi = data_roi.T                              # (n_times, C)
        proc = np.ascontiguousarray(process_eeg(eeg_roi, sfreq_original))
        tensor = torch.from_numpy(proc).float().to(device)   # [1,C,A,200]
        if DEBUG:
            print(f"    after preprocess tensor {tuple(tensor.shape)}")
        max_p = model.time_embed.shape[1]                 # 16
        if tensor.shape[2] > max_p:
            tensor = tensor[:, :, :max_p, :]
            if DEBUG:
                print(f"    cropped to {tuple(tensor.shape)}")
        input_chans = get_input_chans([c.upper() for c in roi_names])
        if DEBUG:
            print(f"    input_chans len={len(input_chans)}  first5={input_chans[:5]}")
        with torch.no_grad():
            embed = model.forward_features(tensor, input_chans=input_chans)
        trial["embeds_labram"] = embed.cpu()
        if DEBUG:
            print(f"    embed shape {tuple(embed.shape)}")
    return subj_dict

# ─────────────────────────────── main execution ─────────────────────────────────
if __name__ == "__main__":
    save_roi_visualisations("/scratch/vjh9526/bdml_2025/project/code/LaBraM/figs")
    for npy_file in sorted(DATA_DIR.glob("*.npy")):
        print(f"\n=== {npy_file.name} ===")
        data = np.load(npy_file, allow_pickle=True).item()
        subj_id = next(iter(data))
        if subj_id not in ['YSD', 'YSL', 'YTL']:
            continue
        data[subj_id] = process_subject(data[subj_id])
        np.save(npy_file, data, allow_pickle=True)
        print(f"✓ saved {npy_file.name}")
    print("\nAll done.")