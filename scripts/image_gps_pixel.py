'''
The code extracts flight and gimbal orientation data, GPS coordinates, and altitude from an image's EXIF metadata, 
and then uses the pixel coordinates to calculate real-world distances and update GPS coordinates based on 
flight and gimbal angles. It also converts GPS coordinates from DMS (Degrees, Minutes, Seconds) format to decimal degrees.
'''

import subprocess
import json
from PIL import Image
import math
import piexif

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

# Function to extract GPS data from EXIF
def extract_gps(exif_data):
    gps_info = exif_data.get('GPS', {})
    if gps_info:
        latitude = gps_info.get(piexif.GPSIFD.GPSLatitude)
        latitude_ref = gps_info.get(piexif.GPSIFD.GPSLatitudeRef)
        longitude = gps_info.get(piexif.GPSIFD.GPSLongitude)
        longitude_ref = gps_info.get(piexif.GPSIFD.GPSLongitudeRef)

        if latitude and longitude and latitude_ref and longitude_ref:
            # Convert GPS coordinates from DMS to decimal degrees
            lat_deg, lat_min, lat_sec = latitude
            lon_deg, lon_min, lon_sec = longitude
            lat = dms_to_decimal(lat_deg[0], lat_min[0], lat_sec[0], latitude_ref.decode())
            lon = dms_to_decimal(lon_deg[0], lon_min[0], lon_sec[0], longitude_ref.decode())

            return lat, lon
    return None, None

# Function to extract the image's EXIF data
def extract_exif(image_path):
    img = Image.open(image_path)
    exif_data = piexif.load(img.info['exif'])  # Extract EXIF data from image
    return exif_data

# Function to calculate real-world distances based on pixel distances
def calculate_real_world_distance(pixel_distance, focal_length, fov_deg, image_width, sensor_width_mm=None):
    if sensor_width_mm is None:
        sensor_width_mm = 11.04 # 36  # Assuming full-frame sensor (36mm)
    
    fov_rad = math.radians(fov_deg)
    image_width_mm = sensor_width_mm * (image_width / 4056)  # Scale sensor width to match image width
    real_world_distance_per_pixel = (math.tan(fov_rad / 2) * 2 * image_width_mm) / image_width
    
    real_world_distance = pixel_distance * real_world_distance_per_pixel
    return real_world_distance

# Function to get new latitude and longitude for a pixel
def get_gps_from_pixel(x, y, flight_degree, gimbal_degree, image_width, image_height, real_world_distance_per_pixel, latitude, longitude):
    # Calculate pixel displacement from the center of the image
    pixel_x = x - (image_width / 2)
    pixel_y = y - (image_height / 2)

    # Correct the pixel distances based on flight and gimbal orientation
    corrected_pixel_x = pixel_x * math.cos(math.radians(flight_degree))  # Adjusting by flight yaw
    corrected_pixel_y = pixel_y * math.cos(math.radians(gimbal_degree))  # Adjusting by gimbal pitch

    # Calculate the real-world distance
    lat_change = corrected_pixel_y * real_world_distance_per_pixel
    lon_change = corrected_pixel_x * real_world_distance_per_pixel

    # Convert real-world changes to geographic changes in lat/lon
    new_latitude = latitude + (lat_change / 111320)  # Approximate conversion for latitude
    new_longitude = longitude + (lon_change / (40008000 * math.cos(math.radians(latitude))))  # Approx for longitude

    return new_latitude, new_longitude

# Function to parse GPS DMS string into degrees, minutes, seconds, and direction
def parse_dms(dms_str):
    dms_parts = dms_str.replace("deg", "").replace("'", "").replace('"', "").split()
    degrees = float(dms_parts[0])
    minutes = float(dms_parts[1])
    seconds = float(dms_parts[2])
    direction = dms_parts[3]
    
    return degrees, minutes, seconds, direction

# Example of using pixel_x and pixel_y in the main function
def process_image(image_path):
    flight_degree, gimbal_degree, gps_latitude, gps_longitude, gps_altitude = extract_flight_gimbal_degrees(image_path)    

    flight_degree = float(flight_degree) if flight_degree else 0.0
    gimbal_degree = float(gimbal_degree) if gimbal_degree else 0.0
    
    if flight_degree:
        print(f"Flight Degree: {flight_degree}")
    if gimbal_degree:
        print(f"Gimbal Degree: {gimbal_degree}")
    if gps_latitude and gps_longitude:
        print(f"GPS Coordinates: Latitude = {gps_latitude}, Longitude = {gps_longitude}")
    if gps_altitude:
        print(f"GPS Altitude: {gps_altitude}")

    if gps_latitude and gps_longitude:
        gps_lat_deg, gps_lat_min, gps_lat_sec, lat_ref = parse_dms(gps_latitude)
        gps_lat_decimal = dms_to_decimal(gps_lat_deg, gps_lat_min, gps_lat_sec, lat_ref)

        gps_lon_deg, gps_lon_min, gps_lon_sec, lon_ref = parse_dms(gps_longitude)
        gps_lon_decimal = dms_to_decimal(gps_lon_deg, gps_lon_min, gps_lon_sec, lon_ref)
        
        print(f"Converted GPS Coordinates: Latitude = {gps_lat_decimal}, Longitude = {gps_lon_decimal}")
        
        img = Image.open(image_path)
        image_width, image_height = img.size
        print(f"Image Dimensions: {image_width} x {image_height}")
        
        # From camera sepc
        focal_length_mm = 4.5
        fov_deg = 73.7
        fov_rad = math.radians(fov_deg)
        sensor_width_mm = 6.3
        print(f"Sensor Width (mm): {sensor_width_mm}")
        
        pixel_x = 1500
        pixel_y = 2300
        print(f"Pixel Coordinates: ({pixel_x}, {pixel_y})")

        corrected_pixel_distance = pixel_x * math.cos(math.radians(flight_degree))
        corrected_pixel_distance = corrected_pixel_distance * math.cos(math.radians(gimbal_degree))

        print(f"Corrected Pixel Distance: {corrected_pixel_distance}")

        real_world_distance = calculate_real_world_distance(corrected_pixel_distance, focal_length_mm, fov_deg, image_width, sensor_width_mm)
        print(f"Estimated Real-World Distance: {real_world_distance} meters")

        # Calculate real-world distance per pixel
        real_world_distance_per_pixel = (math.tan(fov_rad / 2) * 2 * sensor_width_mm) / image_width
        new_latitude, new_longitude = get_gps_from_pixel(pixel_x, pixel_y, flight_degree, gimbal_degree, image_width, image_height, real_world_distance_per_pixel, gps_lat_decimal, gps_lon_decimal)
        print(f"New GPS Coordinates: Latitude = {new_latitude}, Longitude = {new_longitude}")

    else:
        print("No valid GPS data found.")

if __name__ == "__main__":
    image_path = "../images/riseholme/august_2024/39_feet/DJI_20240802142835_0003_W.JPG"  # Replace with your image path
    process_image(image_path)
