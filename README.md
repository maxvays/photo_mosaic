# Generating a photo mosaic
This project creates a photo mosaic: an arrangement of small images which form a larger main image.

## To use this code:

Have all of the potential small images in a folder "small_images".
Have the main image you want to form in the cwd. In params set "main_file" to the filename.

### Params:

adjustment_factor: how much to distort (color, brightness, saturation) the small images to better match the main image
up_scale_factor: by what factor to resize the main image (affects resolution of final mosaic)
height: the height of the small images (affects resolution of the small images in the mosaic)
blend: with what transparency to overlay the original image onto the mosaic to create the final image
usage_penalty: how much to penalize using the same image multiple times. Higher value leads to more balanced spread of which images are used.
