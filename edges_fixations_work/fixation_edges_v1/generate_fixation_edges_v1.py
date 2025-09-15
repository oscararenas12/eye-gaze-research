# %% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import gc

# %% Settings
CSV_PATH    = '../../data/P23_T2(in)_valid_fixations.csv'
OUT_DIR     = '../fixation_edges_v1/generated_images_v1'
SCREEN_RES  = (1920, 1080)   # width, height in pixels
DPI         = 150            # figure DPI (used with SCREEN_RES to size the canvas)

# point styles (match your heatmap-style dots)
PREV_SIZE   = 10             # size for previous points
PREV_ALPHA  = 0.30           # alpha for previous points
CURR_SIZE   = 60             # size for current point
CURR_ALPHA  = 1.00           # alpha for current point

# edge styles
EDGE_COLOR  = 'gray'         # color for connecting lines
EDGE_ALPHA  = 0.50           # alpha for connecting lines
EDGE_WIDTH  = 1.0            # width for connecting lines

# performance settings
GC_INTERVAL = 100            # run garbage collection every N images

# %% Matplotlib optimizations
plt.ioff()  # Turn off interactive mode for faster rendering
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['agg.path.chunksize'] = 10000  # Handle large paths efficiently

# %% Load the eye-tracking data
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows of eye-tracking data")
print(df.head(3))  # quick peek

# Basic column checks
required_cols = {'FPOGX', 'FPOGY'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns {required_cols}")

# Optional: clip normalized coords to [0,1]
# (comment out if you prefer raw values)
df['FPOGX'] = df['FPOGX'].clip(0, 1)
df['FPOGY'] = df['FPOGY'].clip(0, 1)

# Precompute numpy arrays for faster access
x_coords_all = df['FPOGX'].values
y_coords_all = df['FPOGY'].values

# %% Create output directory for images
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Images will be saved in: {OUT_DIR}")

# Try to resolve optional metadata column names used in your text block
time_col = 'TIME(2025/05/28 13:20:00.683)'
if time_col not in df.columns:
    # fallback: use the first column that starts with 'TIME(' if present
    t_candidates = [c for c in df.columns if isinstance(c, str) and c.startswith('TIME(')]
    time_col = t_candidates[0] if t_candidates else None

fpoid_col = 'FPOGID' if 'FPOGID' in df.columns else None
fpodur_col = 'FPOGD' if 'FPOGD' in df.columns else None

# %% Optimized cumulative image function with vectorized edges
def create_cumulative_fixation_image_with_edges_v1(data_up_to_row, current_row, output_path,
                                                   x_coords, y_coords,
                                                   screen_res=SCREEN_RES, dpi=DPI,
                                                   prev_size=PREV_SIZE, prev_alpha=PREV_ALPHA,
                                                   curr_size=CURR_SIZE, curr_alpha=CURR_ALPHA,
                                                   edge_color=EDGE_COLOR, edge_alpha=EDGE_ALPHA,
                                                   edge_width=EDGE_WIDTH):
    """
    Optimized version: Create an image showing all fixation points up to and including the current row,
    with sequential edges connecting them using vectorized operations.

    Previous points: small, semi-transparent white.
    Current point: larger, solid white.
    Edges: connecting lines showing gaze path sequence (vectorized).
    """
    fig_w = screen_res[0] / dpi
    fig_h = screen_res[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='black')
    ax.set_facecolor('black')

    # normalized screen coordinates
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.axis('off')

    # Draw edges connecting sequential fixation points using vectorized LineCollection
    if len(x_coords) > 1:
        # Create line segments array for LineCollection (much faster than individual plots)
        segments = np.array([[[x_coords[i], y_coords[i]], [x_coords[i+1], y_coords[i+1]]]
                            for i in range(len(x_coords) - 1)])

        # Create and add LineCollection (vectorized operation)
        lc = LineCollection(segments, colors=edge_color, alpha=edge_alpha, linewidths=edge_width)
        ax.add_collection(lc)

    # Plot previous points (all but current) - vectorized scatter
    if len(x_coords) > 1:
        prev_x = x_coords[:-1]
        prev_y = y_coords[:-1]
        ax.scatter(prev_x, prev_y, s=prev_size, c='white', alpha=prev_alpha, edgecolors='none')

    # Plot the current fixation point
    ax.scatter([x_coords[-1]], [y_coords[-1]], s=curr_size, c='white', alpha=curr_alpha, edgecolors='none')

    # Metadata text (shown if columns exist)
    info_lines = []
    if time_col is not None and time_col in current_row:
        info_lines.append(f"Time: {current_row[time_col]}s")
    if fpoid_col is not None:
        info_lines.append(f"Fixation ID: {current_row[fpoid_col]}")
    if fpodur_col is not None:
        info_lines.append(f"Duration: {current_row[fpodur_col]}s")
    info_lines.append(f"Points shown: {len(x_coords)}")

    ax.text(0.02, 0.02, "\n".join(info_lines),
            color='white', fontsize=10, transform=ax.transAxes, va='bottom')

    # Optimized saving with tight bbox
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi,
                facecolor='black', edgecolor='none')
    plt.close(fig)

# %% Generate images with performance optimizations
N = len(df)
print(f"Generating {N} images with performance optimizations...")

for i in range(N):
    current_row = df.iloc[i]

    # Use precomputed numpy arrays (faster than pandas iloc operations)
    x_coords = x_coords_all[:i+1]
    y_coords = y_coords_all[:i+1]

    # Use CNT to name files if available; otherwise use index
    if 'CNT' in df.columns:
        name_id = int(current_row['CNT'])
        fname = f"fixation_edges_{name_id:05d}.png"
    else:
        fname = f"fixation_edges_{i+1:05d}.png"

    out_path = os.path.join(OUT_DIR, fname)
    create_cumulative_fixation_image_with_edges_v1(
        df.iloc[:i+1], current_row, out_path, x_coords, y_coords
    )

    # Progress reporting and memory management
    if (i + 1) % 100 == 0 or (i + 1) == N:
        print(f"Created image {i+1}/{N}: {out_path}")

    # Periodic garbage collection to prevent memory buildup
    if (i + 1) % GC_INTERVAL == 0:
        gc.collect()

print("Done with v1 optimizations.")

# %%