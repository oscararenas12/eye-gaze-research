# %% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Settings
CSV_PATH   = '../../data/P23_T2(in)_valid_fixations.csv'
OUT_DIR    = './generated_images_v1'
IMG_W, IMG_H = 1920, 1080
BINS       = 50
DPI        = 150
STEP       = 1          # generate every Nth frame (use 10/25 if you want fewer files)
CUMULATIVE = True       # True = up to row i, False = only row i

# %% Load data
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows of eye-tracking data")
if not {'FPOGX','FPOGY'}.issubset(df.columns):
    raise ValueError("CSV must contain 'FPOGX' and 'FPOGY' columns.")

# Optional: clip to [0,1] to avoid edge spill
df['FPOGX'] = df['FPOGX'].clip(0, 1)
df['FPOGY'] = df['FPOGY'].clip(0, 1)

# %% Prepare output dir
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Images will be saved in: {OUT_DIR}")

# %% Heatmap function
def create_fixation_heatmap(data, output_path, bins=BINS):
    fig, ax = plt.subplots(figsize=(IMG_W/120, IMG_H/120), facecolor='black')  # ~reasonable size
    ax.set_facecolor('black')

    # Limits and coordinate convention
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()  # screen-like coordinates

    x = data['FPOGX'].values
    y = data['FPOGY'].values

    # 2D histogram (density map)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Show heat layer (transpose to align orientation)
    plt.imshow(
        heatmap.T, extent=extent, origin='lower',
        cmap='hot', alpha=1.0
    )

    # Scatter all fixations as white points
    ax.scatter(x, y, color='white', s=10, alpha=0.3)

    # Title + clean frame
    ax.set_title("Fixation Heatmap", color='white')
    ax.axis('off')

    # Save and close
    plt.savefig(output_path, bbox_inches='tight', dpi=DPI)
    plt.close(fig)

# %% Generate one image per row
N = len(df)
count = 0
for i in range(1, N + 1, STEP):
    subset = df.iloc[:i] if CUMULATIVE else df.iloc[[i - 1]]
    out_path = os.path.join(OUT_DIR, f"heatmap_step_{i:04d}.png")
    create_fixation_heatmap(subset, out_path)
    count += 1
    if i % (500*STEP) == 0 or i == N:
        print(f"Saved {count} images (up to row {i}/{N})")

print("Done.")

# %%
