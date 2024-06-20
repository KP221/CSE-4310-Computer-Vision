import cv2
import sys
import numpy as np

def load_image(filename):
    #Load an image from file using OpenCV.
    image = cv2.imread(filename)
    if image is None:
        raise FileNotFoundError(f"No file found at {filename}")
    return image

def create_image_pyramid(image, levels):
    #Create an image pyramid using OpenCV's pyrDown function.
    pyramid = [image]
    for i in range(1, levels):
        image = cv2.pyrDown(image)  # Reduce size using Gaussian pyramid
        pyramid.append(image)
    return pyramid

def save_pyramid(pyramid, base_filename):
    #Save each level of the pyramid as a separate file.
    for i, img in enumerate(pyramid):
        filename = f"{base_filename}_{2**(i+1)}x.png"
        cv2.imwrite(filename, img)
        print(f"Saved {filename}")

def main(filename, levels):
    try:
        image = load_image(filename)
        pyramid = create_image_pyramid(image, int(levels))
        base_filename = filename.split('.')[0]  # Assuming filename has a simple '.ext' extension
        save_pyramid(pyramid, base_filename)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_img_pyramid.py <filename> <levels>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
