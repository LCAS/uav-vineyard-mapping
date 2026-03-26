'''
This script processes an image by extracting metadata, flight and gimbal degrees, GPS coordinates, and altitude from the EXIF data. 
It then calculates the ground sample distance (GSD) and creates a GeoTIFF file with georeferenced information.

extract_flight_gimbal_degrees: Extracts flight yaw, gimbal yaw, GPS coordinates (latitude, longitude, and altitude) from the image's EXIF metadata.
dms_to_decimal: Converts DMS (Degrees, Minutes, Seconds) coordinates to decimal degrees.
parse_dms: Parses a GPS DMS string into degrees, minutes, seconds, and direction.
create_geotiff: Creates a GeoTIFF image by using the calculated ground sample distance (GSD) and the image's EXIF data (GPS coordinates, altitude, flight degree, and gimbal degree). The image is georeferenced using the WGS 84 coordinate system (EPSG:4326).
process_image: Main function that extracts the relevant data, calculates GPS coordinates in decimal degrees, and generates the GeoTIFF if valid data is available.
'''

import rasterio
from rasterio.transform import from_origin
import numpy as np
import subprocess
import json
from PIL import Image, ImageDraw

# Function to extract flight data (e.g., FlightYawDegree, GimbalYawDegree) and GPS coordinates from the image EXIF
def extract_flight_gimbal_degrees(image_path):
    try:
        # Run ExifTool with -json option
        result = subprocess.run(['exiftool', '-json', image_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None, None, None, None, None

        # Parse the JSON output
        metadata = json.loads(result.stdout)
        if metadata:
            metadata_dict = metadata[0]
            flight_degree = metadata_dict.get('FlightYawDegree')
            gimbal_degree = metadata_dict.get('GimbalYawDegree')
            gps_latitude = metadata_dict.get('GPSLatitude')
            gps_longitude = metadata_dict.get('GPSLongitude')
            gps_altitude = metadata_dict.get('GPSAltitude')

            # Extract numeric part from altitude if it contains "m Above Sea Level"
            if gps_altitude and isinstance(gps_altitude, str):
                altitude_value = gps_altitude.split()[0]  # Get the numeric part (before "m")
                gps_altitude = float(altitude_value)  # Convert to float

            return flight_degree, gimbal_degree, gps_latitude, gps_longitude, gps_altitude
        else:
            print("No metadata found.")
            return None, None, None, None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None


# Function to convert DMS (Degrees, Minutes, Seconds) to decimal degrees
def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

# Function to parse GPS DMS string into degrees, minutes, seconds, and direction
def parse_dms(dms_str):
    # Extract components from DMS string, which includes deg, ', ", and N/S/E/W
    dms_parts = dms_str.replace("deg", "").replace("'", "").replace('"', "").split()
    degrees = float(dms_parts[0])
    minutes = float(dms_parts[1])
    seconds = float(dms_parts[2])
    direction = dms_parts[3]
    
    return degrees, minutes, seconds, direction

# Function to create a GeoTIFF
def create_geotiff(image_path, gps_latitude, gps_longitude, gps_altitude, flight_degree, gimbal_degree):
    
    # Open the image
    img = Image.open(image_path)
    image_data = np.array(img)
    
    # Get the image width and height directly
    image_width_pixels, image_height_pixels = img.size
    
    # Camera specifications (updated values)
    focal_length_mm = 4.5  # in mm (focal length of the camera)
    sensor_width_mm = 6.3  # in mm (sensor width for example)
    
    # Calculate the pixel size (size of one pixel on the sensor)
    # The field of view and sensor width can be used to calculate the pixel size
    # Use the FoV and sensor width to calculate the sensor resolution (in pixels)
    
    # The sensor resolution in pixels horizontally can be derived from the FOV and sensor width
    sensor_resolution_x = image_width_pixels
    pixel_size = sensor_width_mm / sensor_resolution_x  # in mm
    
    # Assuming the altitude is in meters
    altitude = gps_altitude if gps_altitude else 100  # Default altitude 100m if not provided
    
    # Calculate the ground sample distance (GSD) in meters per pixel
    gsd = (altitude * pixel_size) / focal_length_mm  # GSD in meters per pixel
    
    print(f"Ground sample distance (GSD): {gsd} meters per pixel")
    
    # Ensure gps_latitude and gps_longitude are floats
    gps_latitude = float(gps_latitude)
    gps_longitude = float(gps_longitude)

    # Adjust pixel size for the GeoTIFF
    # Assuming each pixel represents 'gsd' meters on the ground
    transform = from_origin(gps_longitude, gps_latitude, gsd, gsd)
    
    # Define the CRS (coordinate reference system) - WGS 84 for GPS coordinates
    crs = 'EPSG:4326'
    
    # Create and write the GeoTIFF
    geo_tiff_path = "../images/output_image.tif"
    with rasterio.open(geo_tiff_path, 'w', driver='GTiff', height=image_height_pixels,
                        width=image_width_pixels, count=3, dtype='uint8', crs=crs, transform=transform) as dst:
        dst.write(image_data[:, :, 0], 1)  # Red band
        dst.write(image_data[:, :, 1], 2)  # Green band
        dst.write(image_data[:, :, 2], 3)  # Blue band
        
    print(f"GeoTIFF created at {geo_tiff_path}")

# Main function to process the image
def process_image(image_path):
    flight_degree, gimbal_degree, gps_latitude, gps_longitude, gps_altitude = extract_flight_gimbal_degrees(image_path)

    flight_degree = float(flight_degree) if flight_degree else 0.0
    gimbal_degree = float(gimbal_degree) if gimbal_degree else 0.0

    # Ensure that the GPS values are in a format that can be parsed (e.g., "53 deg 16' 05.37\" N")
    if gps_latitude and gps_longitude:
        gps_lat_deg, gps_lat_min, gps_lat_sec, lat_ref = parse_dms(gps_latitude)
        gps_lat_decimal = dms_to_decimal(gps_lat_deg, gps_lat_min, gps_lat_sec, lat_ref)

        gps_lon_deg, gps_lon_min, gps_lon_sec, lon_ref = parse_dms(gps_longitude)
        gps_lon_decimal = dms_to_decimal(gps_lon_deg, gps_lon_min, gps_lon_sec, lon_ref)
        
        print(f"GPS Coordinates: Latitude = {gps_lat_decimal}, Longitude = {gps_lon_decimal}")
    else:
        print("Invalid GPS coordinates provided.")
        gps_lat_decimal = gps_lon_decimal = None

    if gps_altitude:
        print(f"GPS Altitude: {gps_altitude}")        
    if flight_degree:
        print(f"Flight Degree: {flight_degree}")
    if gimbal_degree:
        print(f"Gimbal Degree: {gimbal_degree}")

    # Only create GeoTIFF if valid GPS coordinates are present
    if gps_lat_decimal and gps_lon_decimal:
        create_geotiff(image_path, gps_lat_decimal, gps_lon_decimal, gps_altitude, flight_degree, gimbal_degree)
    else:
        print("GeoTIFF creation skipped due to invalid GPS coordinates.")

if __name__ == "__main__":
    image_path = "../images/riseholme/august_2024/39_feet/DJI_20240802142835_0003_W.JPG"
    process_image(image_path)
