import numpy as np
import skimage.io as io
import sys

def normalize_and_split_channels(rgb):
    #Normalize RGB values and split into separate channels.
    normalized_rgb = rgb.astype('float32') / 255.0
    return normalized_rgb[..., 0], normalized_rgb[..., 1], normalized_rgb[..., 2]

def calculate_hsv(r, g, b):
    #Calculate the HSV from the RGB components."""
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    chroma = max_val - min_val
    
    # Safe divide for saturation
    saturation = np.where(max_val == 0, 0, chroma / max_val)

    # Initialize hue
    hue = np.zeros_like(r)
    
    # Mask where chroma is non-zero to avoid division by zero
    mask = chroma > 0
    # RGB -> GBR -> BRG to loop over different conditions
    for i, (comp1, comp2) in enumerate([(g, b), (b, r), (r, g)]):
        comp_mask = ((r == max_val) & (i == 0)) | ((g == max_val) & (i == 1)) | ((b == max_val) & (i == 2))
        valid_mask = mask & comp_mask
        hue[valid_mask] = ((comp1 - comp2) / chroma)[valid_mask] + i * 2

    hue = (hue * 60) % 360
    hue[chroma == 0] = 0  # Set hue to 0 where there is no color

    value = max_val

    return np.stack([hue, saturation, value], axis=-1)

def convert_hsv_to_rgb(hsv):
    #Convert HSV values back to RGB using vectorization."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h = h / 60
    c = v * s
    x = c * (1 - np.abs(h % 2 - 1))
    m = v - c

    rgb = np.zeros_like(hsv)
    z = np.zeros_like(h)

    f = h.astype(int) % 6
    rgb[..., 0] = np.choose(f, [c, x, z, z, x, c], mode='clip') + m
    rgb[..., 1] = np.choose(f, [x, c, c, x, z, z], mode='clip') + m
    rgb[..., 2] = np.choose(f, [z, z, x, c, c, x], mode='clip') + m

    return (rgb * 255).astype(np.uint8)

def adjust_hsv_values(hsv, hue, saturation, value):
    """Adjust HSV values based on input parameters."""
    hsv[..., 0] = (hsv[..., 0] + hue) % 360
    hsv[..., 1] = np.clip(hsv[..., 1] + saturation, 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] + value, 0, 1)
    return hsv

def process_image(filename, hue, saturation, value):
    #Process the image by adjusting its HSV values and saving the RGB result.
    if not (0 <= hue <= 360 and 0 <= saturation <= 1 and 0 <= value <= 1):
        raise ValueError("Hue, saturation, or value out of accepted range.")
    
    img = io.imread(filename)
    r, g, b = normalize_and_split_channels(img)
    hsv = calculate_hsv(r, g, b)
    hsv_adjusted = adjust_hsv_values(hsv, hue, saturation, value)
    rgb = convert_hsv_to_rgb(hsv_adjusted)
    
    output_filename = f"adjusted_{filename}"
    io.imsave(output_filename, rgb)
    print(f"Image saved as {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <filename> <hue> <saturation> <value>")
        sys.exit(1)
    
    process_image(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))


