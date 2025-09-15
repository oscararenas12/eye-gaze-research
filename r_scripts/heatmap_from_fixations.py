#!/usr/bin/env python3
"""
Heatmap Generator from Fixation Data
---------------------------------------------
Given a CSV of fixation coordinates -> generate heatmap images.

Quick usage (single CSV):
    python heatmap_from_fixations.py --csv whatever_valid_fixations.csv --out ./heatmaps

Add an overlay:
    python heatmap_from_fixations.py --csv whatever_valid_fixations.csv --overlay overlay.png --out ./heatmaps

Batch by per-row stimulus column (e.g., "image" / "stimulus" / "slide"):
    python heatmap_from_fixations.py --csv all_fixations.csv --batch-by image --stim-root ./slides --out ./heatmaps

Options:
    --xcol, --ycol       : manually set column names if auto-detect fails
    --width, --height    : stimulus size in pixels (default 1920x1080)
    --bins               : grid bins as "WxH" or scalar (default "192x108")
    --sigma              : Gaussian blur in bin units if SciPy available (default 1.0)
    --alpha              : overlay opacity for heatmap (0..1, default 0.6)
    --cmap               : matplotlib colormap for heatmap (default "hot")
    --dpi                : output DPI (default 150)
    --normalized         : if your x/y are in [0,1], scale by width/height first
    --filter             : optional pandas query string to filter rows
    --prefix             : filename prefix for outputs (default "")

Assumptions:
- Coordinates are in pixel space, origin at top-left (x rightwards, y downwards).
- If your data is normalized (0..1), use --normalized to scale to width/height.

Outputs PNG files to the given --out directory.
"""
import argparse
import os
import sys
import math
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# try optional smoothing
try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True

except Exception:
    _HAS_SCIPY = False

# column auto detection
X_CANDIDATES = [
    "x", "X",
    "fix_x", "FixX", "FixationX", "fixation_x",
    "GazeX", "gaze_x", "gazepos_x", "gx",
    "pos_x", "PositionX", "screen_x"
]
Y_CANDIDATES = [
    "y", "Y",
    "fix_y", "FixY", "FixationY", "fixation_y",
    "GazeY", "gaze_y", "gazepos_y", "gy",
    "pos_y", "PositionY", "screen_y"
]
STIM_CANDIDATES = [
    "image", "stimulus", "slide", "scene", "frame", "frame_image", "stim"
]


def autodetect_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    colset = {c.lower() for c in columns}
    for cand in candidates:
        if cand.lower() in colset:
            return [c for c in columns if c.lower() == cand.lower()][0]
        
    return None


def load_fixation_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV loaded but has no rows.")
    
    return df


def make_heatmap(
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
    bins: Tuple[int, int] = (192, 108),
    sigma: float = 1.0,
) -> np.ndarray:
    """
    build a 2d histogram in screen coordinates and smooth it
    xs ys are pixel coords with origin top left
    """
    # clip coordinates to screen
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)

    # histogram (note: numpy.histogram2d expects first array as x, second as y; we map to image correctly after)
    H, xedges, yedges = np.histogram2d(xs, ys, bins=bins, range=[[0, width], [0, height]])

    # H shape: (bins_x, bins_y). we want (rows, cols) -> (bins_y, bins_x) for imshow
    H = H.T  # transpose so rows = y, cols = x

    if _HAS_SCIPY and sigma > 0:
        H = gaussian_filter(H, sigma=sigma)

    # Normalize
    if H.max() > 0:
        H = H / H.max()

    return H


def draw_and_save_heatmap(
    heatmap: np.ndarray,
    out_path: str,
    overlay_path: Optional[str] = None,
    alpha: float = 0.6,
    cmap: str = "hot",
    width: int = 1920,
    height: int = 1080,
    dpi: int = 150,
) -> None:
    """
    save heatmap (optionally overlaid on an image)
    """
    fig_w = width / dpi
    fig_h = height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])  # fill figure -> no margins
    ax.set_axis_off()

    if overlay_path and os.path.isfile(overlay_path):
        img = plt.imread(overlay_path)
        ax.imshow(img, extent=[0, width, height, 0])  # extent flips y to top left origin

    ax.imshow(
        heatmap,
        extent=[0, width, height, 0],
        interpolation='bilinear',
        alpha=alpha,
        cmap=cmap
    )

    # no borders
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def process_dataframe(
    df: pd.DataFrame,
    xcol: Optional[str],
    ycol: Optional[str],
    batch_by: Optional[str],
    normalized: bool,
    width: int,
    height: int,
    bins: Tuple[int, int],
    sigma: float,
    overlay: Optional[str],
    stim_root: Optional[str],
    out_dir: str,
    prefix: str,
    alpha: float,
    cmap: str,
    dpi: int,
) -> List[str]:
    """
    create heatmaps
    if batch_by is provided and present in the DF, group by it and
    make one map per group (trying to find a matching overlay under stim_root if present)
    """
    outputs = []

    # auto detect columns if not given
    xcol = xcol or autodetect_column(list(df.columns), X_CANDIDATES)
    ycol = ycol or autodetect_column(list(df.columns), Y_CANDIDATES)

    if xcol is None or ycol is None:
        raise ValueError(
            "Could not auto-detect x/y columns. "
            "Please pass --xcol and --ycol explicitly."
        )

    # optional filter to drop NaNs and insane coords
    df = df.dropna(subset=[xcol, ycol])

    # normalize if necessary
    def get_coords(sub: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        xs = sub[xcol].to_numpy(dtype=float)
        ys = sub[ycol].to_numpy(dtype=float)
        if normalized:
            xs = xs * width
            ys = ys * height

        return xs, ys

    if batch_by and batch_by in df.columns:
        for key, sub in df.groupby(batch_by):
            xs, ys = get_coords(sub)
            if len(xs) == 0:
                continue
            H = make_heatmap(xs, ys, width=width, height=height, bins=bins, sigma=sigma)

            # determine overlay for this batch
            ov = None
            if overlay:
                ov = overlay
            elif stim_root:
                # Look for a file that matches the key
                base = os.path.join(stim_root, str(key))
                for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                    if os.path.isfile(base + ext):
                        ov = base + ext
                        break

            safe_key = str(key).replace("/", "_").replace("\\", "_").replace(" ", "_")
            out_path = os.path.join(out_dir, f"{prefix}{safe_key}_heatmap.png")
            draw_and_save_heatmap(H, out_path, overlay_path=ov, alpha=alpha, cmap=cmap, width=width, height=height, dpi=dpi)
            outputs.append(out_path)
    else:
        xs, ys = get_coords(df)
        if len(xs) == 0:
            raise ValueError("No fixation rows found after filtering.")
        H = make_heatmap(xs, ys, width=width, height=height, bins=bins, sigma=sigma)
        out_path = os.path.join(out_dir, f"{prefix}heatmap.png")
        draw_and_save_heatmap(H, out_path, overlay_path=overlay, alpha=alpha, cmap=cmap, width=width, height=height, dpi=dpi)
        outputs.append(out_path)

    return outputs


def parse_bins(bins_str: str) -> Tuple[int, int]:
    if "x" in bins_str.lower():
        a, b = bins_str.lower().split("x")
        return (int(a), int(b))
    else:
        b = int(bins_str)

        # keep aspect ratio of width:height ~ 16:9 -> translate scalar to (cols, rows)
        cols = b
        rows = max(1, int(round(b * 9/16)))
        return (cols, rows)


def main():
    ap = argparse.ArgumentParser(description="Generate heatmaps from fixation CSVs.")
    ap.add_argument("--csv", required=True, help="Path to fixation CSV file.")
    ap.add_argument("--out", required=True, help="Output directory for heatmap images.")
    ap.add_argument("--xcol", default=None, help="Name of X column (auto-detect if omitted).")
    ap.add_argument("--ycol", default=None, help="Name of Y column (auto-detect if omitted).")
    ap.add_argument("--batch-by", default=None, help="Optional column to group by (e.g., image/stimulus).")
    ap.add_argument("--overlay", default=None, help="Single image path to overlay heatmap onto.")
    ap.add_argument("--stim-root", default=None, help="Directory containing per-stimulus images (used with --batch-by).")
    ap.add_argument("--width", type=int, default=1920, help="Stimulus width in pixels (default 1920).")
    ap.add_argument("--height", type=int, default=1080, help="Stimulus height in pixels (default 1080).")
    ap.add_argument("--bins", default="192x108", help="Grid bins as 'WxH' or scalar (default 192x108).")
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian blur in bin units if SciPy available (default 1.0).")
    ap.add_argument("--alpha", type=float, default=0.6, help="Heatmap overlay opacity (0..1, default 0.6).")
    ap.add_argument("--cmap", default="hot", help="Matplotlib colormap (default 'hot').")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI (default 150).")
    ap.add_argument("--normalized", action="store_true", help="Set if x/y are normalized [0,1].")
    ap.add_argument("--filter", default=None, help="Optional pandas query string to filter rows (e.g., \"duration > 100\").")
    ap.add_argument("--prefix", default="", help="Prefix for output filenames.")
    args = ap.parse_args()

    # load data
    df = load_fixation_csv(args.csv)

    # optional filter
    if args.filter:
        try:
            df = df.query(args.filter)
        except Exception as e:
            print(f"[WARN] Failed to apply filter '{args.filter}': {e}", file=sys.stderr)

    # resolve batch by column name (auto detect common names if user passed placeholder)
    batch_by = args.batch_by
    if batch_by == "auto" or (batch_by is None):
        # try to auto detect a stimulus column if not explicitly set
        found = autodetect_column(list(df.columns), STIM_CANDIDATES)
        if args.batch_by == "auto":
            batch_by = found
        elif args.batch_by is None:
            # keep None unless we find something strong
            batch_by = found

    bins = parse_bins(args.bins)

    outputs = process_dataframe(
        df=df,
        xcol=args.xcol,
        ycol=args.ycol,
        batch_by=batch_by,
        normalized=args.normalized,
        width=args.width,
        height=args.height,
        bins=bins,
        sigma=args.sigma,
        overlay=args.overlay,
        stim_root=args.stim_root,
        out_dir=args.out,
        prefix=args.prefix,
        alpha=args.alpha,
        cmap=args.cmap,
        dpi=args.dpi,
    )

    print("Wrote heatmaps:")
    for p in outputs:
        print(" -", p)


if __name__ == "__main__":
    main()
