import numpy as np
from PIL import Image

def replace_black_pixels_vectorized(image_path, height, replacement_color):
    img = Image.open(image_path)
    img_array = np.array(img)

    # Extract R, G, B channels
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    # Create a mask for black pixels below the specified height
    black_pixels = (r == 0) & (g == 0) & (b == 0) & (np.arange(img_array.shape[0])[:, None] > height)

    # Apply replacement color to the selected pixels
    img_array[black_pixels] = replacement_color

    modified_img = Image.fromarray(np.uint8(img_array))
    modified_img.save('modified_image.png')

# Example usage
image_path = '/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/logs/images/map_sdv15_ctv11_city2real_v4_g50_c2k_zurich_seed30/ctnet_i2i_debug_e2000/src_lbl_50.png'
# image_path = 'example.jpg'  # Replace with the path to your image
height = 400  # Set the height below which pixels will be replaced
replacement_color = (0, 0, 142)  # Set the RGB tuple for replacement color

replace_black_pixels_vectorized(image_path, height, replacement_color)
