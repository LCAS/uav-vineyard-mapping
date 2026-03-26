import numpy as np
from PIL import Image

def threshold_image_at_percentile(image_path: str, percentile: float, output_path: str):
    """
    Loads an image, finds the pixel intensity threshold at a given percentile,
    and creates a new binary image based on that threshold.

    Args:
        image_path (str): The path to the input image file.
        percentile (float): The percentile (0-100) to use for thresholding.
        output_path (str): The path to save the output binary image.
    """
    try:
        # 1. Open the image and convert it to grayscale.
        # This is necessary because intensity is a single value,
        # which is easier to work with than RGB channels.
        original_image = Image.open(image_path).convert('L')
        print(f"Image '{image_path}' loaded and converted to grayscale.")

        # 2. Convert the grayscale image to a NumPy array for efficient calculation.
        pixel_array = np.array(original_image)

        # 3. Calculate the threshold value using numpy.percentile.
        # The 95th percentile value means that 95% of the pixels have an
        # intensity at or below this value.
        threshold_value = np.percentile(pixel_array, percentile)
        print(f"The threshold value for the {percentile}th percentile is: {threshold_value}")

        # 4. Create a new binary array.
        # Pixels with an intensity >= the threshold are set to 255 (white).
        # All other pixels are set to 0 (black).
        binary_array = np.where(pixel_array >= threshold_value, 255, 0).astype(np.uint8)
        
        # 5. Convert the NumPy array back to a PIL Image object.
        thresholded_image = Image.fromarray(binary_array)

        # 6. Save the new binary image.
        thresholded_image.save(output_path)
        print(f"Thresholded image saved to '{output_path}' successfully.")

    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define the input image path and desired percentile.
    input_file = "inference_outputs/post_probability_heatmap.png"
    threshold_percentile = 99.0
    output_file = "inference_outputs/post_probability_heatmap_thresholded.png"

    # Run the function with the specified parameters.
    threshold_image_at_percentile(input_file, threshold_percentile, output_file)
