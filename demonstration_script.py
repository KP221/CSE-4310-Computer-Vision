
import numpy as np
import skimage.io as io
import cv2
from color_space_test import normalize_and_split_channels, calculate_hsv, convert_hsv_to_rgb, adjust_hsv_values
from create_img_pyramid import load_image, create_image_pyramid, save_pyramid
from img_transforms import hsv_to_rgb, random_crop, resize_image, create_image_pyramid as transform_image_pyramid

def main():
    # Example image loading (You will need to replace 'example.jpg' with an actual image file path)
    image_path = 'example.jpg'
    image = load_image(image_path)
    
    # Demonstrating color space normalization and splitting
    r, g, b = normalize_and_split_channels(image)
    print("RGB Channels split and normalized.")

    # Calculating HSV from RGB and converting back
    hsv = calculate_hsv(r, g, b)
    rgb_converted = convert_hsv_to_rgb(hsv)
    print("Converted RGB to HSV and back to RGB.")

    # Adjusting HSV values
    adjusted_hsv = adjust_hsv_values(hsv, 10, 0.1, 0.1)
    print("Adjusted HSV values.")
    
    # Creating and saving an image pyramid
    pyramid = create_image_pyramid(image, 3)
    save_pyramid(pyramid, 'pyramid_example')
    print("Created and saved image pyramid.")
    
    # Random crop and resizing example
    cropped_image = random_crop(image, 100)
    resized_image = resize_image(image, 0.5)
    print("Performed random crop and resized image.")

    # Displaying or saving images can be added here for visual confirmation.

if __name__ == "__main__":
    main()
