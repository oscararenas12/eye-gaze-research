
### Generating heatmap on fixations:

Problem: heatmap can exaggerate very few fixations points, if handful of points may get mapped to bright “hotspots,” making it look like there was strong visual attention when in reality it was just a few glances. This misleads interpretation, since heatmaps are typically read as density or intensity maps, and overemphasizing sparse data can distort conclusions about gaze behavior.

#### But What if the data im looking has be validated at good fixations points then would we care?

If your data has already been validated as good fixation points, then you don’t need to worry about whether they’re “real”, they are. The issue isn’t about validity, but about **interpretation**: even valid fixations can look misleading on a heatmap if there are only a few of them, because the color scale can exaggerate their weight. So you would still care if you’re using the visualization to communicate density or attention patterns, but less so if your main goal is simply to show where those valid fixations occurred.

### Time to generate images

`generate_fixations_images_v0.py` = ~15m
- My iteration of generating the image, its good. Just straight forward. Points were to dim so I want to make darker using heatmap. It have a function that creates but not for each image.


`generate_fixation_images_v1.py` = 27m 23.2s
- At the end it looks good and it even get outline in red but early frames looked almost blank/unhelpful.

`generate_fixation_images_v2.py` = 13m 7.4s
- Preferred approach for visualization: shows all prior fixations with the current fixation emphasized. If the current point appears too large, adjust CURR_SIZE. 

`generate_fixation_images_v2.py` = TBD
- Testing a cockpit image as the background for the fixation overlay. I may need the original high-resolution image to evaluate properly.

Russell have nice heat map, is there value in bringing them into one with fixation points.

#### Notes:

My mentor Joel mentioned Segmentation