import rasterio

def get_geotiff_transform(image_file):
    """
    Retrieves and prints the transform of a GeoTIFF image.

    Args:
        image_file (str): Path to the GeoTIFF image.
    """
    try:
        with rasterio.open(image_file) as src:
            transform = src.transform
            print("Transform:")
            print(transform)
            print("\nComponents:")
            print(f"  Origin (top-left corner): {transform.c}, {transform.f}")
            print(f"  Pixel width (x-resolution): {transform.a}")
            print(f"  Pixel height (y-resolution): {transform.e}")
            print(f"  Rotation (x-axis): {transform.b}")
            print(f"  Rotation (y-axis): {transform.d}")

            width = src.width
            height = src.height
            print("Image Dimensions (Pixels):")
            print(f"  Width: {width} px")
            print(f"  Height: {height} px\n")
            
            transform = src.transform
            print("Transform:")
            print(transform)

            crs = src.crs
            print("CRS:")
            print(crs)
            print("\nEPSG Code:")
            if crs.is_epsg_code:
                print(crs.to_epsg())
            else:
                print("CRS does not have an EPSG code.")
    except rasterio.RasterioIOError:
        print(f"Error: Could not open GeoTIFF file: {image_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    image_file = "../images/orthophoto/39_feet/odm_orthophoto.tif"  # Replace with your GeoTIFF file path
    get_geotiff_transform(image_file)