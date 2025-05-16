import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mne

# Set to True to get a console log when figures are written
DEBUG = False

def save_montage_with_highlights(
    montage_name: str,
    highlights: list,
    out_file: str,
    title: str = None,
    size: int = 6,
    dpi: int = 120
):
    """
    Create and save a single 2D topomap of the specified montage,
    highlighting the channels in `highlights`.

    Parameters
    ----------
    montage_name : str
        Name of the standard montage (e.g. "GSN-HydroCel-129" or "standard_1020").
    highlights : list of str
        Channel names to highlight.
    out_file : str
        Path (file name) to save the figure to.
    title : str, optional
        Figure title.
    size : int, optional
        Figure size in inches (square).
    dpi : int, optional
        Figure resolution in dots per inch.
    """
    # Load the named montage
    montage = mne.channels.make_standard_montage(montage_name)

    # Extract 2D positions
    pos3d = montage.get_positions()["ch_pos"]
    ch_names = montage.ch_names
    xy = np.array([pos3d[ch][:2] for ch in ch_names])
    x, y = xy[:, 0], xy[:, 1]

    # Set up figure
    fig, ax = plt.subplots(figsize=(size, size), dpi=dpi)
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    # Plot all electrodes in grey
    ax.scatter(x, y, s=14, color="#bbbbbb", zorder=1)

    # Overlay highlighted electrodes in red
    mask = np.array([ch in highlights for ch in ch_names])
    ax.scatter(x[mask], y[mask], s=40, color="#e31a1c", zorder=2)

    # Label only the highlighted ones
    for xi, yi, ch, m in zip(x, y, ch_names, mask):
        if m:
            ax.text(
                xi, yi, ch,
                color="#e31a1c",
                fontsize=6,
                ha="center", va="center",
                zorder=3
            )

    # Optional title
    if title:
        ax.set_title(title, fontsize=10)

    # Ensure output directory exists
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save and close
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    if DEBUG:
        print(f"Saved montage to {out_path.resolve()}")


if __name__ == "__main__":
    # Example definitions of your channel sets:
    # KEEP_HYDRO = [
    #     # ... list your 129 HydroCel channels here ...
    #     "E13", "E14", "E20",  # etc.
    # ]
    ROI_1020 = [
        "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2"
    ]

    # Example: save the HydroCel-129 montage
    # save_montage_with_highlights(
    #     montage_name="GSN-HydroCel-129",
    #     highlights=KEEP_HYDRO,
    #     out_file="figs/hydro_roi.png",
    #     title="HydroCel-129 – recorded electrodes"
    # )

    # Example: save the standard‑10‑20 montage
    save_montage_with_highlights(
        montage_name="standard_1020",
        highlights=ROI_1020,
        out_file="/scratch/vjh9526/bdml_2025/project/figs/std20_roi_image.png",
        title="Standard‑10‑20 – ROI channels"
    )
