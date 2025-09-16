
### Generating heatmap on fixations:

Problem: heatmap can exaggerate very few fixations points, if handful of points may get mapped to bright “hotspots,” making it look like there was strong visual attention when in reality it was just a few glances. This misleads interpretation, since heatmaps are typically read as density or intensity maps, and overemphasizing sparse data can distort conclusions about gaze behavior.

#### But What if the data im looking has be validated at good fixations points then would we care?

If your data has already been validated as good fixation points, then you don’t need to worry about whether they’re “real”, they are. The issue isn’t about validity, but about **interpretation**: even valid fixations can look misleading on a heatmap if there are only a few of them, because the color scale can exaggerate their weight. So you would still care if you’re using the visualization to communicate density or attention patterns, but less so if your main goal is simply to show where those valid fixations occurred.

### basic_fixation_work

`generate_fixations_images_v0.py` = ~15m
- My iteration of generating the image, its good. Just straight forward. Points were to dim so I want to make darker using heatmap. It have a function that creates but not for each image.


`generate_fixation_images_v1.py` = 27m 23.2s
- At the end it looks good and it even get outline in red but early frames looked almost blank/unhelpful.

`generate_fixation_images_v2.py` = 13m 7.4s
- Preferred approach for visualization: shows all prior fixations with the current fixation emphasized. If the current point looks too large, I can adjust it. I also added a final image of this approach.

### edges_fixation_work

`generate_fixation_edges_v0.py` = 1hr to generate 2500 out of 4730

Creates edge-connected fixation images by connecting gray lines between fixation points with white dots for previous fixations. Only down to this way is that its taking way too long because its independently generating each frame from scratch, redrawing all cumulative edges and points up to the current fixation and the large white dot for the current point becomes obscured as edge density increases over time. Also visually not appealing, very bland

- Gray lines connecting fixation points in sequence
- Small white dots for previous fixations
- Large white dot for current fixation
- Progressive edge buildup as more fixations are processed
- After while the large white dot for current fixation get hidden behind the points and edges
- It's NOT building incrementally - each image is generated independently by processing the cumulative dataset up to that frame.
- For each image, it:
  1. Creates a fresh black canvas
  2. Draws all edges from point `1→2`, `2→3`, `3→4`, ..., up to current point
  3. Draws all previous points as small white dots
  4. Draws current point as large white dot
  This approach ensures:
  - ✅ No dependency on previous image files
  - ❌ More computationally intensive (redraws everything each time)
  - ❌ Slower overall processing

`generate_fixation_edges_v1.py` = 23m 9.7s

 Optimized version of v0, uses batch operations to draw all connecting lines at once instead of one-by-one, processes data in efficient numpy arrays rather than slower pandas operations, and includes memory cleanup to prevent slowdowns. Achieves 5x speed improvement while maintaining identical image output. Same cons exist from v0, large white dot for the current point becomes obscured as edge density increases over time and not visually appealing, very bland.
 
  1. Vectorized edge plotting: Uses LineCollection instead of individual ax.plot() calls - should be 5-10x faster for edge drawing    
  2. Precomputed numpy arrays: Extract coordinates once, reuse with array slicing instead of repeated pandas operations
  3. Matplotlib optimizations: Disabled interactive mode, increased path chunk size, removed figure warnings
  4. Memory management: Periodic garbage collection every 100 images to prevent memory buildup
  5. Optimized saving: Added explicit face/edge color settings for faster rendering

  Expected performance gains:
  - 5-10x faster edge rendering (vectorized LineCollection vs loops)
  - Faster data access (numpy arrays vs pandas iloc)
  - Better memory usage (garbage collection)
  - Same visual output with absolute precision maintained

  The script maintains the exact same visual output as v0 but should run significantly faster, especially noticeable on the later images with hundreds of edges.

`generate_fixation_edges_v2.py` = 18m 31s

Visually more appealing + optimized version. 

- Improved color scheme: orange previous points, yellow current point with red border, cyan edges
- Added z-order control to prevent current point overlap by edges
- Increased sizes and alpha values for better visibility against black background
- Compare visual quality with v1 images - confirm better visibility of points and edges

### duration_fixation_work

`generate_fixation_duration_v0.py` = 23m 40.7s
Looks interesting enough, I think it might have more value than original iteration since we there emphasis on duration of outer fixations points

`generate_fixation_duration_v1.py` = 37m 21.9s
Since I got duration points from v0 I thought maybe instead of point it could be heatmap duration point so longer fixations create "hotter" spots on the grid but too much smoothing for the heatmap, the smoothing should be less aggressive because there so many datapoint if deviates from source of heat it wont leave any trace of it. It did help create better/ideal heatmap that I think make sense.
- Note About Smoothing - Smoothing: Applies Gaussian blur to create smooth heat gradients instead of discrete pixels, making the visualization more readable. 


`generate_fixation_duration_v2.py` = 49m 22.3s
My thought for this one was merge v0 + v1 because now the heatmap could a layer over duration fixation points and overall I thought this was pretty good. Initial images are concerning just i don't how CNN will react to them but final mid way to end looks good. Primary concern is GRID_SIZE for heatmap, since this impacts performance. The GRID_SIZE parameter controls the spatial resolution of the heatmap computation, representing a critical trade-off between visual quality and processing speed.

How GRID_SIZE Works:
  - Creates a computational grid that maps screen coordinates to discrete cells
  - Each fixation point gets mapped to a grid cell based on its (x,y) position
  - Duration intensity accumulates within each cell (multiple fixations stack)
  - Gaussian blur operates across this grid to create smooth heat gradients
  - Final result gets upscaled to full screen resolution for display

Resolution Options:
  - (192, 108) - 1/10th resolution: Fast processing (~18-37min), moderate detail
  - (768, 432) - 1/2.5th resolution: Balanced quality/speed (~1-2hrs)
  - (1920, 1080) - Full resolution: Maximum detail (~3-6hrs), pixel-perfect precision

Key Features:
- Layered Visualization: Duration-weighted heatmap (top layer, z=10) overlays persistent fixation points (bottom layer, z=2-3)
- Dual Duration Encoding: Point size scales with fixation duration (20-150px range), while heatmap intensity reflects cumulative duration density
- Progressive Accumulation: Each frame shows all previous fixations plus current, building comprehensive gaze pattern over time
- Enhanced Visibility: Increased heatmap alpha (0.4-0.7) ensures heat patterns remain visible even with hundreds of accumulated points

Technical Implementation:
- Creates low-resolution heat grid for performance optimization
- Applies Gaussian smoothing (σ=2.0) for gradient transitions
- Uses 'hot' colormap (black→red→yellow→white) with adaptive transparency
- Color scheme: orange previous points, cyan current point with white borders

Overall the final image look really helpful, i would say.

### edges_duration_fixation_work

`generate_edges_duration_fixations_v0.py` = 71m 5.7s
Merge of fixation_duration_v2.py + fixation_edges_v2.py. I think edges may cause too much noise but would love to hear your thoughts 

#### Notes:

My mentor Joel mentioned Segmentation