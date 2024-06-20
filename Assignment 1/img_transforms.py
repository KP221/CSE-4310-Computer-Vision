
import numpy as np
from skimage import color, transform

def hsv_to_rgb(hsv):
    #Convert an HSV image to RGB using the skimage library.
    return color.hsv2rgb(hsv)  # Utilize skimage's conversion function for accurate color translation

def random_crop(img, size):
    #Generate a random crop from an image.
    if size > min(img.shape[:2]):
        raise ValueError("Crop size too large.")
    start_x = np.random.randint(0, img.shape[1] - size)
    start_y = np.random.randint(0, img.shape[0] - size)
    return img[start_y:start_y + size, start_x:start_x + size]

def resize_image(image, scale_factor):
    #Resize an image by a scale factor using skimage.
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    return transform.resize(image, (new_height, new_width), anti_aliasing=True, mode='reflect')

def create_image_pyramid(image, levels):
    #Create a series of scaled images forming a pyramid.
    pyramids = [transform.rescale(image, scale=1/(2**i), multichannel=True) for i in range(levels)]
    return pyramids

# Example usage
if __name__ == "__main__":
    img = np.random.rand(256, 256, 3)
    resized_img = resize_image(img, 0.5)
    pyramid = create_image_pyramid(img, 3)

