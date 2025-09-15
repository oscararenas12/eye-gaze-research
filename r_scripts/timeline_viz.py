#!/usr/bin/env python3
"""
Timeline Heatmap Visualizer
---------------------------------------
Create incremental/windowed heatmaps from eye tracking fixations, with optional
point (duration scaled) and edge (scanpath) overlays. Saves a sequence of PNG frames
you can drop into a report or turn into a GIF/MP4 later.

Quick start (will auto-detect common column names):
    python timeline_viz.py \
        --csv valid_fixations.csv \
        --out out/P23_T2 \
        --window-mode cumulative \
        --draw-points --draw-edges

Window modes (optional, most of the time BG will be used to pre-proccess the data into windows):
    cumulative                growing window from the start (aka "expanding")
    tumbling                  fixed, non-overlapping windows by time
    hopping                   fixed windows that advance by a hop (can overlap)

Common flags:
    --overlay stimulus.png    background image behind the heatmap
    --width 1920 --height 1080 size of the stimulus/coordinate space
    --bins 192x108            histogram grid (WxH) or single number (keeps 16:9)
    --sigma 1.0               Gaussian blur (in *bin* units) if SciPy available
    --normalized              if x/y are in [0,1], scale by width/height first
    --ts-ms                   timestamps are milliseconds (default assumes seconds)
    --global-norm             normalize all frames by the same max (prettier)
    --step 5                  render every Nth frame to save time

Outputs are written into:  <out>/frames/frame_00000.png, ...
"""

from __future__ import annotations
import argparse
import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# auto detect column helpers
X_CANDS = ["x","X","fix_x","FixX","FixationX","fixation_x","GazeX","gaze_x","gx","pos_x","PositionX","screen_x"]
Y_CANDS = ["y","Y","fix_y","FixY","FixationY","fixation_y","GazeY","gaze_y","gy","pos_y","PositionY","screen_y"]
T_CANDS = ["timestamp","time","t","ts","ms","Time","Timestamp"]
DUR_CANDS = ["duration","fix_dur","FixationDuration","dur","Duration","dur_ms","fixation_duration"]


def autodetect(cols: List[str], cands: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for cand in cands:
        if cand.lower() in low:
            return low[cand.lower()]
    return None



# core heatmap + drawing
def parse_bins(bins_str: str) -> Tuple[int,int]:
    s = str(bins_str).lower()
    if "x" in s:
        a,b = s.split("x")
        return (int(a), int(b))
    # scalar -> keep 16:9 rows approximation
    cols = int(s)
    rows = max(1, int(round(cols * 9/16)))
    return (cols, rows)


def make_heatmap(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    width: int,
    height: int,
    bins: Tuple[int, int],
    sigma: float,
    global_max: Optional[float],
    weights: Optional[np.ndarray] = None,  # optional: e.g., durations
) -> np.ndarray:
    # clean: drop NaNs and out-of-bounds instead of clipping
    mask = (
        np.isfinite(xs) & np.isfinite(ys) &
        (xs >= 0) & (xs < width) &
        (ys >= 0) & (ys < height)
    )
    if weights is not None:
        mask = mask & np.isfinite(weights)
        w = weights[mask]
    else:
        w = None

    xs = xs[mask]
    ys = ys[mask]

    # histogram in image coordinates
    H, _, _ = np.histogram2d(xs, ys, bins=bins, range=[[0, width], [0, height]], weights=w)
    H = H.T  # rows = y, cols = x for imshow

    # optional smoothing
    if _HAS_SCIPY and sigma > 0:
        H = gaussian_filter(H, sigma=sigma)

    # normalize
    if global_max is not None and global_max > 0:
        H = H / global_max
    else:
        m = H.max()
        if m > 0:
            H = H / m

    return H


def draw_points(ax, xs: np.ndarray, ys: np.ndarray, durations: Optional[np.ndarray]):
    if durations is None or len(durations) == 0:
        sizes = 30
    else:
        # map durations to a reasonable size range (px^2 for scatter)
        dmin, dmax = float(np.nanmin(durations)), float(np.nanmax(durations))
        if dmax <= dmin:
            sizes = 30
        else:
            sizes = np.interp(durations, (dmin, dmax), (15, 90))
    ax.scatter(xs, ys, s=sizes, c='cyan', alpha=0.75, linewidths=0.5, edgecolors='k')


def draw_edges(ax, xs: np.ndarray, ys: np.ndarray):
    if len(xs) >= 2:
        ax.plot(xs, ys, color='lime', linewidth=2.0, alpha=0.7)


def render_frame(out_path: str, H: np.ndarray, *, width: int, height: int,
                 overlay: Optional[str], alpha: float, cmap: str,
                 xs: Optional[np.ndarray]=None, ys: Optional[np.ndarray]=None,
                 durations: Optional[np.ndarray]=None,
                 draw_points_flag: bool=False, draw_edges_flag: bool=False,
                 dpi: int=150):
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = plt.axes([0,0,1,1])
    ax.set_axis_off()

    if overlay and os.path.isfile(overlay):
        bg = plt.imread(overlay)
        ax.imshow(bg, extent=[0,width,height,0])

    ax.imshow(H, extent=[0,width,height,0], cmap=cmap, interpolation='bilinear', alpha=alpha)

    if xs is not None and ys is not None:
        if draw_edges_flag:
            draw_edges(ax, xs, ys)
        if draw_points_flag:
            draw_points(ax, xs, ys, durations)

    ax.set_xlim([0,width]); ax.set_ylim([height,0])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



# window makers (time-based)
def make_frames_indices(times: np.ndarray, *, mode: str, window_size: float,
                        hop_size: float, step: int) -> List[np.ndarray]:
    """Return a list of index arrays for each frame.
       times must be sorted ascending and in *seconds*.
    """
    n = len(times)
    if n == 0:
        return []

    frames: List[np.ndarray] = []

    if mode == 'cumulative':
        for i in range(0, n, step):
            frames.append(np.arange(0, i+1, dtype=int))
        return frames

    if mode == 'tumbling':
        start = times[0]
        end = times[-1]
        t = start
        while t <= end:
            win_end = t + window_size
            idx = np.where((times >= t) & (times < win_end))[0]
            if len(idx):
                frames.append(idx)
            t = t + window_size  # non-overlapping
        return frames

    if mode == 'hopping':
        start = times[0]
        end = times[-1]
        t = start
        while t <= end:
            win_end = t + window_size
            idx = np.where((times >= t) & (times < win_end))[0]
            if len(idx):
                frames.append(idx)
            t = t + (hop_size if hop_size > 0 else window_size)
        return frames

    raise ValueError(f"Unknown window mode: {mode}")


def compute_global_max(xs: np.ndarray, ys: np.ndarray, width: int, height: int,
                       bins: Tuple[int,int], sigma: float, frames: List[np.ndarray]) -> float:
    m = 0.0
    for idx in frames:
        if len(idx) == 0:
            continue
        H, _, _ = np.histogram2d(xs[idx], ys[idx], bins=bins, range=[[0,width],[0,height]])
        H = H.T
        if _HAS_SCIPY and sigma > 0:
            H = gaussian_filter(H, sigma=sigma)
        m = max(m, float(H.max()))
    return m if m > 0 else 1.0


def main():
    p = argparse.ArgumentParser(description="Incremental/windowed heatmaps with optional point/edge overlays.")
    p.add_argument('--csv', required=True, help='Path to fixation CSV.')
    p.add_argument('--out', required=True, help='Output directory (frames/ subfolder will be created).')

    # columns
    p.add_argument('--xcol', default=None)
    p.add_argument('--ycol', default=None)
    p.add_argument('--tcol', default=None, help='Timestamp column (seconds by default).')
    p.add_argument('--duration-col', default=None)
    p.add_argument('--normalized', action='store_true', help='If x/y are in [0,1], scale to width/height.')
    p.add_argument('--ts-ms', action='store_true', help='If set, treat timestamp as milliseconds and convert to seconds.')

    # viz params
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--bins', default='192x108')
    p.add_argument('--sigma', type=float, default=1.0)
    p.add_argument('--cmap', default='hot')
    p.add_argument('--alpha', type=float, default=0.6)
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--overlay', default=None)

    # windowing
    p.add_argument('--window-mode', choices=['cumulative','tumbling','hopping'], default='cumulative')
    p.add_argument('--window-size', type=float, default=60.0, help='Seconds for tumbling/hopping window.')
    p.add_argument('--hop-size', type=float, default=30.0, help='Seconds to advance per hop (hopping mode).')
    p.add_argument('--step', type=int, default=1, help='Render every Nth frame for cumulative mode.')
    p.add_argument('--max-frames', type=int, default=None, help='Optional cap on number of frames to render.')

    # overlays
    p.add_argument('--draw-points', action='store_true')
    p.add_argument('--draw-edges', action='store_true')

    # normalization
    p.add_argument('--global-norm', action='store_true', help='Normalize all frames by the same max value.')

    args = p.parse_args()

    # load data
    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError('CSV has no rows.')

    # auto detect columns
    xcol = args.xcol or autodetect(list(df.columns), X_CANDS)
    ycol = args.ycol or autodetect(list(df.columns), Y_CANDS)
    tcol = args.tcol or autodetect(list(df.columns), T_CANDS)
    dcol = args.duration_col or autodetect(list(df.columns), DUR_CANDS)

    if xcol is None or ycol is None:
        raise ValueError('Could not detect x/y columns. Use --xcol/--ycol.')
    if tcol is None:
        raise ValueError('Could not detect timestamp column. Use --tcol.')

    # clean
    df = df.dropna(subset=[xcol, ycol, tcol]).copy()
    # sort by time
    df.sort_values(tcol, inplace=True)

    # extract arrays
    xs = df[xcol].to_numpy(dtype=float)
    ys = df[ycol].to_numpy(dtype=float)
    ts = df[tcol].to_numpy(dtype=float)
    if args.ts_ms:
        ts = ts / 1000.0  # convert ms -> s

    if args.normalized:
        xs = xs * args.width
        ys = ys * args.height

    durations = None
    if dcol is not None and dcol in df.columns:
        durations = df[dcol].to_numpy(dtype=float)

    bins = parse_bins(args.bins)

    # build the frame index sets
    frame_indices = make_frames_indices(ts, mode=args.window_mode,
                                        window_size=args.window_size,
                                        hop_size=args.hop_size,
                                        step=args.step)

    if args.max_frames is not None:
        frame_indices = frame_indices[:args.max_frames]

    # compute global normalization if requested
    glob = None
    if args.global_norm:
        glob = compute_global_max(xs, ys, args.width, args.height, bins, args.sigma, frame_indices)

    # render frames
    out_frames_dir = os.path.join(args.out, 'frames')
    os.makedirs(out_frames_dir, exist_ok=True)

    for i, idx in enumerate(frame_indices):
        if len(idx) == 0:
            continue
        H = make_heatmap(xs[idx], ys[idx], width=args.width, height=args.height,
                          bins=bins, sigma=args.sigma, global_max=glob)
        # for overlays, we want the *current* subset in time order
        cur_x, cur_y = xs[idx], ys[idx]
        cur_d = durations[idx] if durations is not None else None

        out_path = os.path.join(out_frames_dir, f"frame_{i:05d}.png")
        render_frame(out_path, H, width=args.width, height=args.height,
                     overlay=args.overlay, alpha=args.alpha, cmap=args.cmap,
                     xs=cur_x, ys=cur_y, durations=cur_d,
                     draw_points_flag=args.draw_points, draw_edges_flag=args.draw_edges,
                     dpi=args.dpi)

    print(f"Wrote {len(frame_indices)} frame(s) to: {out_frames_dir}")


if __name__ == '__main__':
    main()
