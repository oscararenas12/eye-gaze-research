# %% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Settings
CSV_PATH    = '../../data/P23_T2(in)_valid_fixations.csv'
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_duration_images_v0')
SCREEN_RES  = (1920, 1080)   # width, height in pixels
DPI         = 150            # figure DPI (used with SCREEN_RES to size the canvas)

# duration-based sizing settings
MIN_SIZE    = 20             # minimum point size
MAX_SIZE    = 150            # maximum point size
PREV_ALPHA  = 0.60           # alpha for previous points
CURR_ALPHA  = 1.00           # alpha for current point

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

# Check if duration column exists
if fpodur_col is None:
    raise ValueError("Duration column (FPOGD) not found. Duration-based sizing requires fixation duration data.")

# Get duration statistics for scaling
min_duration = df[fpodur_col].min()
max_duration = df[fpodur_col].max()
print(f"Duration range: {min_duration:.3f}s to {max_duration:.3f}s")

# %% Cumulative image function (updated point styling)
def create_duration_fixation_image(data_up_to_row, current_row, output_path,
                                   screen_res=SCREEN_RES, dpi=DPI,
                                   min_size=MIN_SIZE, max_size=MAX_SIZE,
                                   prev_alpha=PREV_ALPHA, curr_alpha=CURR_ALPHA):
    """
    Create an image showing all fixation points up to and including the current row.
    Point sizes are scaled based on fixation duration.

    Previous points: duration-scaled, semi-transparent orange.
    Current point: duration-scaled, bright yellow with red border.
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

    # Calculate duration-based sizes for all points
    durations = data_up_to_row[fpodur_col].values
    # Normalize durations to size range
    normalized_durations = (durations - min_duration) / (max_duration - min_duration)
    sizes = min_size + normalized_durations * (max_size - min_size)

    # Plot previous points (all but current) with duration-based sizing
    if len(data_up_to_row) > 1:
        prev = data_up_to_row.iloc[:-1]
        prev_sizes = sizes[:-1]
        ax.scatter(prev['FPOGX'], prev['FPOGY'],
                   s=prev_sizes, c='orange', alpha=prev_alpha,
                   edgecolors='white', linewidths=1.0, zorder=2)

    # Plot the current fixation point with duration-based sizing
    current_size = sizes[-1]
    ax.scatter([current_row['FPOGX']], [current_row['FPOGY']],
               s=current_size, c='yellow', alpha=curr_alpha,
               edgecolors='red', linewidths=2, zorder=3)

    # Metadata text (shown if columns exist)
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

# %% Generate images (you can change range to a subset if testing)
N = len(df)
for i in range(N):
    current_row = df.iloc[i]
    data_up_to_row = df.iloc[:i+1]

    # Use CNT to name files if available; otherwise use index
    if 'CNT' in df.columns:
        name_id = int(current_row['CNT'])
        fname = f"fixation_duration_{name_id:05d}.png"
    else:
        fname = f"fixation_duration_{i+1:05d}.png"

    out_path = os.path.join(OUT_DIR, fname)
    create_duration_fixation_image(data_up_to_row, current_row, out_path)

    if (i + 1) % 500 == 0 or (i + 1) == N:
        print(f"Created image {i+1}/{N}: {out_path}")

print("Done generating duration-based fixation images.")

# %%
