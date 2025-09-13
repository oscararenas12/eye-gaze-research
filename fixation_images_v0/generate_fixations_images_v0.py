#%% Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

#%% Load the eye-tracking data
# Read the CSV file
df = pd.read_csv('../data/P23_T2(in)_valid_fixations.csv')
print(f"Loaded {len(df)} rows of eye-tracking data")
print(df.head(3))  # Display first few rows to verify data


#%% Create output directory for images
output_dir = '/fixation_images_v0/generated_images_v0'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")
else:
    print(f"Directory already exists: {output_dir}")

#%% Define function to create cumulative fixation image
def create_cumulative_fixation_image(data_up_to_row, current_row, output_path, screen_res=(1920, 1080)):
    """
    Create an image showing all fixation points up to and including the current row
    
    Parameters:
    - data_up_to_row: DataFrame containing all rows up to current one
    - current_row: Current row being processed
    - output_path: Path to save the image
    - img_size: Size of the output image (width, height)
    """
    # Create a black background
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='black')
    ax.set_facecolor('black')
    
    # Set plot limits (assuming FPOGX and FPOGY are normalized between 0 and 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Invert Y-axis to match screen coordinates (0,0 at top-left)
    ax.invert_yaxis()
    
    # Plot all previous fixation points as smaller gray dots
    if len(data_up_to_row) > 1:  # If there are previous points
        previous_points = data_up_to_row.iloc[:-1]  # All but the current row
        ax.scatter(previous_points['FPOGX'], previous_points['FPOGY'], 
                  color='gray', s=30, alpha=0.5)
    
    # Plot the current fixation point as a larger white dot
    ax.scatter(current_row['FPOGX'], current_row['FPOGY'], color='white', s=30)
    
    # # Add a small circle to represent the current fixation area
    # circle = plt.Circle((current_row['FPOGX'], current_row['FPOGY']), 
    #                     0.02, color='white', fill=False, alpha=0.7)
    # # ax.add_patch(circle)
    
    # Add metadata as text
    info_text = f"Time: {current_row['TIME(2025/05/28 13:20:00.683)']}s\n" \
                f"Fixation ID: {current_row['FPOGID']}\n" \
                f"Duration: {current_row['FPOGD']}s\n" \
                f"Points shown: {len(data_up_to_row)}"
    ax.text(0.02, 0.02, info_text, color='white', fontsize=10, 
            transform=ax.transAxes, verticalalignment='bottom')
    
    # Remove axes
    ax.axis('off')
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

#%% Process a subset of rows to test (first 10 rows)
test_rows = min(10, len(df))
for i in range(test_rows):
    current_row = df.iloc[i]
    data_up_to_row = df.iloc[:i+1]  # All rows up to and including current
    
    output_path = os.path.join(output_dir, f"fixation_{int(current_row['CNT']):05d}.png")
    create_cumulative_fixation_image(data_up_to_row, current_row, output_path)
    print(f"Created image {i+1}/{test_rows}: {output_path}")

# #%% Process all rows (uncomment when ready)
# for i in range(len(df)):
#     current_row = df.iloc[i]
#     data_up_to_row = df.iloc[:i+1]  # All rows up to and including current
    
#     output_path = os.path.join(output_dir, f"fixation_{int(current_row['CNT']):05d}.png")
#     create_cumulative_fixation_image(data_up_to_row, current_row, output_path)
    
#     # Print progress every 100 images
#     if (i + 1) % 100 == 0:
#         print(f"Processed {i+1}/{len(df)} images")

#%% Create a heatmap of all fixation points
def create_fixation_heatmap(data, output_path, img_size=(1920, 1080)):
    """Create a heatmap of all fixation points"""
    # Create a black background
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='black')
    ax.set_facecolor('black')
    
    # Set plot limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Invert Y-axis to match screen coordinates
    ax.invert_yaxis()
    
    # Create heatmap using kernel density estimation
    x = data['FPOGX']
    y = data['FPOGY']
    

    # Create a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, range=[[0, 1], [0, 1]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
    # Display the heatmap
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', alpha=1.0)
    
    # Plot all fixation points as white dots
    ax.scatter(x, y, color='white', s=10, alpha=0.3)
    
    # Add title
    plt.title("Fixation Heatmap", color='white')
    
    # Remove axes
    ax.axis('off')
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

# Create a heatmap of all fixation points
create_fixation_heatmap(df, os.path.join(output_dir, 'fixation_heatmap.png'))
# %%
