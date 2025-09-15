# %% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# %% Settings
CSV_PATH    = '../../data/P23_T2(in)_valid_fixations.csv'
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_duration_images_v1')
SCREEN_RES  = (1920, 1080)   # width, height in pixels
DPI         = 150            # figure DPI (used with SCREEN_RES to size the canvas)

# heat map settings
GRID_SIZE   = (192, 108)     # heat map grid resolution (1/10th of screen res for performance)
BLUR_SIGMA  = 2.0           # gaussian blur radius for heat map smoothing
MIN_ALPHA   = 0.3           # minimum alpha for heat map overlay
MAX_ALPHA   = 0.8           # maximum alpha for heat map overlay

# current point settings
CURR_SIZE   = 100           # size for current point marker
CURR_ALPHA  = 1.0           # alpha for current point

# %% Load the eye-tracking data
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows of eye-tracking data")

# Basic column checks
required_cols = {'FPOGX', 'FPOGY'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns {required_cols}")

# Optional: clip normalized coords to [0,1]
df['FPOGX'] = df['FPOGX'].clip(0, 1)
df['FPOGY'] = df['FPOGY'].clip(0, 1)

# %% Create output directory for images
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Images will be saved in: {OUT_DIR}")

# Try to resolve optional metadata column names
time_col = 'TIME(2025/05/28 13:20:00.683)'
if time_col not in df.columns:
    t_candidates = [c for c in df.columns if isinstance(c, str) and c.startswith('TIME(')]
    time_col = t_candidates[0] if t_candidates else None

fpoid_col = 'FPOGID' if 'FPOGID' in df.columns else None
fpodur_col = 'FPOGD' if 'FPOGD' in df.columns else None

# Check if duration column exists
if fpodur_col is None:
    raise ValueError("Duration column (FPOGD) not found. Duration-based heat mapping requires fixation duration data.")

# Get duration statistics for scaling
min_duration = df[fpodur_col].min()
max_duration = df[fpodur_col].max()
print(f"Duration range: {min_duration:.3f}s to {max_duration:.3f}s")

# %% Heat map generation function
def create_duration_heatmap_image(data_up_to_row, current_row, output_path,
                                  screen_res=SCREEN_RES, dpi=DPI, grid_size=GRID_SIZE,
                                  blur_sigma=BLUR_SIGMA, min_alpha=MIN_ALPHA, max_alpha=MAX_ALPHA,
                                  curr_size=CURR_SIZE, curr_alpha=CURR_ALPHA):
    """
    Create an image with a duration-based heat map of all fixations up to current row.
    Each fixation creates a heat blob proportional to its duration.
    Current point is highlighted with a bright marker.
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

    # Create heat map grid (height, width for array indexing)
    heat_grid = np.zeros((grid_size[1], grid_size[0]))

    # Convert fixation positions to grid coordinates
    x_positions = data_up_to_row['FPOGX'].values
    y_positions = data_up_to_row['FPOGY'].values
    durations = data_up_to_row[fpodur_col].values

    # Normalize durations to [0, 1] for heat intensity
    normalized_durations = (durations - min_duration) / (max_duration - min_duration)

    # Add each fixation to the heat map
    for x, y, intensity in zip(x_positions, y_positions, normalized_durations):
        # Convert normalized coords to grid indices
        # Note: grid_size is (width, height) but array indexing is [height, width]
        grid_x = int(x * (grid_size[0] - 1))
        grid_y = int(y * (grid_size[1] - 1))

        # Ensure indices are within bounds
        grid_x = max(0, min(grid_size[0] - 1, grid_x))
        grid_y = max(0, min(grid_size[1] - 1, grid_y))

        # Add intensity to the heat map (accumulative for overlapping fixations)
        # Array indexing: heat_grid[row, col] = heat_grid[y, x]
        heat_grid[grid_y, grid_x] += intensity

    # Apply gaussian blur for smooth heat map
    if blur_sigma > 0:
        heat_grid = gaussian_filter(heat_grid, sigma=blur_sigma)

    # Normalize heat map to [0, 1]
    if heat_grid.max() > 0:
        heat_grid = heat_grid / heat_grid.max()

    # Create heat map overlay with varying alpha
    if heat_grid.max() > 0:
        # Create RGBA heat map (red-yellow-white colormap)
        heat_rgba = plt.cm.hot(heat_grid)

        # Set alpha based on heat intensity
        alpha_map = min_alpha + heat_grid * (max_alpha - min_alpha)
        heat_rgba[..., 3] = alpha_map

        # Display heat map
        ax.imshow(heat_rgba, extent=[0, 1, 1, 0], aspect='auto', interpolation='bilinear')

    # Plot the current fixation point with bright marker
    ax.scatter([current_row['FPOGX']], [current_row['FPOGY']],
               s=curr_size, c='cyan', alpha=curr_alpha,
               edgecolors='white', linewidths=2, zorder=10)

    # Metadata text
    info_lines = []
    if time_col is not None and time_col in current_row:
        info_lines.append(f"Time: {current_row[time_col]}s")
    if fpoid_col is not None:
        info_lines.append(f"Fixation ID: {current_row[fpoid_col]}")
    if fpodur_col is not None:
        info_lines.append(f"Duration: {current_row[fpodur_col]}s")
    info_lines.append(f"Points shown: {len(data_up_to_row)}")

    ax.text(0.02, 0.02, "\n".join(info_lines),
            color='white', fontsize=10, transform=ax.transAxes, va='bottom')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)

# %% Generate images
N = len(df)
for i in range(N):
    current_row = df.iloc[i]
    data_up_to_row = df.iloc[:i+1]

    # Use CNT to name files if available; otherwise use index
    if 'CNT' in df.columns:
        name_id = int(current_row['CNT'])
        fname = f"fixation_heatmap_{name_id:05d}.png"
    else:
        fname = f"fixation_heatmap_{i+1:05d}.png"

    out_path = os.path.join(OUT_DIR, fname)
    create_duration_heatmap_image(data_up_to_row, current_row, out_path)

    if (i + 1) % 500 == 0 or (i + 1) == N:
        print(f"Created heat map image {i+1}/{N}: {out_path}")

print("Done generating duration-based heat map images.")

# %%