from PIL import Image, ImageOps, ImageEnhance
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import colorsys
import numpy as np

Image.MAX_IMAGE_PIXELS = 933120000

params = {
    "adjustment_factor":0.25,
    "main_file":"main3",
    "up_scale_factor":8,
    "height":400,
    "blend":.4,
    "usage_penalty":.01
}

def calculate_average_brightness(image):
    # Convert the image to grayscale and return the average pixel value
    grayscale_image = image.convert('L')
    return np.mean(grayscale_image)

def calculate_average_saturation(image):
    # Convert the image to the HSV color space and return the average saturation
    hsv_image = image.convert('HSV')
    return np.mean(hsv_image.getdata(1))

def calculate_average_color(image):
    # Return the average color of the image
    return np.mean(image, axis=(0, 1))

usage_penalty = params["usage_penalty"]

def adjust_image(image, target_image, adjustment_factor=params["adjustment_factor"]):
    # Convert the images to NumPy arrays
    image = np.array(image)
    target_image = np.array(target_image)

    # Calculate the average color of the image and the target image
    image_color = calculate_average_color(image)
    target_color = calculate_average_color(target_image)

    # Calculate the average brightness of the image and the target image
    image_brightness = calculate_average_brightness(Image.fromarray(image))
    target_brightness = calculate_average_brightness(Image.fromarray(target_image))

    # Calculate the average saturation of the image and the target image
    image_saturation = calculate_average_saturation(Image.fromarray(image))
    target_saturation = calculate_average_saturation(Image.fromarray(target_image))

    # Adjust the image's color
    for channel in range(3):
        image[:,:,channel] = np.clip(image[:,:,channel] + adjustment_factor * (target_color[channel] - image_color[channel]), 0, 255)

    # Convert the image back to the PIL format
    image = Image.fromarray(image.astype('uint8'))

    # Adjust the image's brightness
    brightness_converter = ImageEnhance.Brightness(image)
    image = brightness_converter.enhance(1 + adjustment_factor * (target_brightness / image_brightness - 1))

    # Adjust the image's saturation
    saturation_converter = ImageEnhance.Color(image)
    image = saturation_converter.enhance(1 + adjustment_factor * (target_saturation / image_saturation - 1))

    return image

def calculate_average_color(image):
    # Calculate the average color of an image
    r, g, b = np.mean(image, axis=(0, 1))
    return r, g, b

def calculate_correlation(main_image_section, small_image):
    # Flatten the arrays and calculate the cosine similarity
    return cosine_similarity(main_image_section.flatten().reshape(1, -1), small_image.flatten().reshape(1, -1))

def select_best_image(main_image_section, images, image_usage):
    # Calculate the correlation for each image and its mirror image and apply a penalty for usage
    scores = [(max(calculate_correlation(main_image_section, np.array(images[i])),
                   calculate_correlation(main_image_section, np.array(images[i].transpose(Image.Transpose.FLIP_LEFT_RIGHT))))
               - usage_penalty * image_usage[i]) for i in range(len(images))]
    
    # Return the index of the image with the highest score
    return np.argmax(scores)


def create_mosaic(main_image, portrait_images, landscape_images):
    # Make sure the main image size is a multiple of the small image size
    portrait_width, portrait_height = portrait_images[0].size
    landscape_width, landscape_height = landscape_images[0].size
    main_width = (main_image.width // max(portrait_width, landscape_width)) * max(portrait_width, landscape_width)
    main_height = (main_image.height // portrait_height) * portrait_height
    main_image = main_image.resize((main_width, main_height))

    # Create separate usage dictionaries for landscape and portrait images
    landscape_image_usage = defaultdict(int)
    portrait_image_usage = defaultdict(int)

    # Create a new image to hold the mosaic
    mosaic = Image.new('RGB', main_image.size)

    for i in range(0, main_height, portrait_height):
        print(i)
        j = 0
        while j < main_width:
            if j + landscape_width <= main_width:
                # Try both portrait and landscape images and pick the best one
                main_image_section = np.array(main_image.crop((j, i, j + landscape_width, i + portrait_height)))
                portrait_image_section = np.array(main_image.crop((j, i, j + portrait_width, i + portrait_height)))
                landscape_image_index = select_best_image(main_image_section, landscape_images, landscape_image_usage)
                landscape_image = landscape_images[landscape_image_index]
                portrait_image_index = select_best_image(portrait_image_section, portrait_images, portrait_image_usage)
                portrait_image = portrait_images[portrait_image_index]
                
                if calculate_correlation(main_image_section, np.array(landscape_image)) > calculate_correlation(portrait_image_section, np.array(portrait_image)):
                    # Adjust the landscape image and paste it into the mosaic
                    if calculate_correlation(main_image_section, np.array(landscape_image)) > calculate_correlation(main_image_section, np.array(landscape_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))):
                        landscape_image = adjust_image(landscape_image, Image.fromarray(main_image_section))
                        mosaic.paste(landscape_image, (j, i))
                    else:
                        landscape_image = adjust_image(landscape_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), Image.fromarray(main_image_section))
                        mosaic.paste(landscape_image, (j, i))
                    landscape_image_usage[landscape_image_index] += 1
                    j += landscape_width
                else:
                    # Adjust the portrait image and paste it into the mosaic
                    if calculate_correlation(portrait_image_section, np.array(portrait_image)) > calculate_correlation(portrait_image_section, np.array(portrait_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))):
                        portrait_image = adjust_image(portrait_image, Image.fromarray(portrait_image_section))
                        mosaic.paste(portrait_image, (j, i))
                    else:
                        portrait_image = adjust_image(portrait_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), Image.fromarray(portrait_image_section))
                        mosaic.paste(portrait_image, (j, i))
                    portrait_image_usage[portrait_image_index] += 1
                    j += portrait_width
            else:
                # Only enough space for a portrait image
                main_image_section = np.array(main_image.crop((j, i, j + portrait_width, i + portrait_height)))
                portrait_image_index = select_best_image(main_image_section, portrait_images, portrait_image_usage)
                portrait_image = portrait_images[portrait_image_index]
                # Adjust the portrait image and paste it into the mosaic
                if calculate_correlation(main_image_section, np.array(portrait_image)) > calculate_correlation(main_image_section, np.array(portrait_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))):
                    portrait_image = adjust_image(portrait_image, Image.fromarray(main_image_section))
                    mosaic.paste(portrait_image, (j, i))
                else:
                    portrait_image = adjust_image(portrait_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), Image.fromarray(main_image_section))
                    mosaic.paste(portrait_image, (j, i))
                portrait_image_usage[portrait_image_index] += 1
                j += portrait_width
    return mosaic

main_file = params["main_file"]

# Load the main image
main_image = Image.open(f'{main_file}.jpg')
main_image = ImageOps.exif_transpose(main_image)
up_scale_factor = params["up_scale_factor"]
main_image = main_image.resize((main_image.width * up_scale_factor, main_image.height * up_scale_factor))

# Load the small images
# Load the small images
image_files = [ImageOps.exif_transpose(Image.open(os.path.join('small_images', filename))) for filename in os.listdir('small_images')]

# Separate the images into portrait and landscape
portrait_images = [image for image in image_files if image.width < image.height]
landscape_images = [image for image in image_files if image.width >= image.height]

# Find the smallest width and height for each category
portrait_width = min(image.width for image in portrait_images)
portrait_height = min(image.height for image in portrait_images)
landscape_width = min(image.width for image in landscape_images)
landscape_height = min(image.height for image in landscape_images)

# Resize all images to the smallest dimensions
portrait_images = [image.resize((portrait_width, portrait_height)) for image in portrait_images]
landscape_images = [image.resize((landscape_width, landscape_height)) for image in landscape_images]


# Resize the portrait and landscape images to have the same height
height = params["height"]
portrait_height = height
landscape_height = height

# Resize the images while maintaining aspect ratio
portrait_images = [image.resize((int(image.width * portrait_height / image.height), portrait_height)) for image in portrait_images]
landscape_images = [image.resize((int(image.width * landscape_height / image.height), landscape_height)) for image in landscape_images]

# Create the mosaic
mosaic = create_mosaic(main_image, portrait_images, landscape_images)

# Save the mosaic
mosaic.save(f'temp_{main_file}_mosaic.jpg')

# Re-open the saved mosaic (this is necessary because PIL needs to re-load the image data)
mosaic = Image.open(f'temp_{main_file}_mosaic.jpg')

# Re-size the main image to match the mosaic size
main_image = main_image.resize(mosaic.size)

# Blend the mosaic with the main image
blend_image = Image.blend(mosaic, main_image, params["blend"])

# Save the blend image
blend_image.save(f'{main_file}_adj{params["adjustment_factor"]}_up{params["up_scale_factor"]}_h{params["height"]}_b{params["blend"]}_mosaic.jpg')