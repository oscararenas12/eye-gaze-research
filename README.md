
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

### edges_fixation_work

`generate_fixation_edges_v0.py` = 1hr to generate 2500 out of 4730

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
- Improved color scheme: orange previous points, yellow current point with red border, cyan edges
- Added z-order control to prevent current point overlap by edges
- Increased sizes and alpha values for better visibility against black background
- Compare visual quality with v1 images - confirm better visibility of points and edges
 
Russell have nice heat map, is there value in bringing them into one with fixation points.

#### Notes:

My mentor Joel mentioned Segmentation