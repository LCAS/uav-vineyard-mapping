import subprocess
import json
import re
from PIL import Image, ImageDraw
import math
import pandas as pd
import os

def extract_exif(image_path):
    try:
        result = subprocess.run(['exiftool', '-json', image_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None, None, None, None, None

        metadata = json.loads(result.stdout)
        if metadata:
            metadata_dict = metadata[0]
            
            flight_yaw_degree = metadata_dict.get('FlightYawDegree', None)
            flight_pitch_degree = metadata_dict.get('FlightPitchDegree', None)
            flight_roll_degree = metadata_dict.get('FlightRollDegree', None)
            gimbal_yaw_degree = metadata_dict.get('GimbalYawDegree', 0)
            gimbal_pitch_degree = metadata_dict.get('GimbalPitchDegree', 0)
            gimbal_roll_degree = metadata_dict.get('GimbalRollDegree', 0)
            gps_latitude_dms = metadata_dict.get('GPSLatitude', None)
            gps_longitude_dms = metadata_dict.get('GPSLongitude', None)
            gps_altitude = metadata_dict.get('RelativeAltitude', None)
            fov_degrees = metadata_dict.get('FOV', None)
            focal_length_mm = metadata_dict.get('FocalLength', None)
            image_height = metadata_dict.get('ImageHeight', None)
            image_width = metadata_dict.get('ImageWidth', None)

            if gps_latitude_dms and gps_longitude_dms:
                gps_latitude = dms_to_decimal(gps_latitude_dms)
                gps_longitude = dms_to_decimal(gps_longitude_dms)
            else:
                gps_latitude = gps_longitude = None

            return flight_yaw_degree, flight_pitch_degree, flight_roll_degree, gimbal_yaw_degree, gimbal_pitch_degree, gimbal_roll_degree, gps_latitude, gps_longitude, gps_altitude, fov_degrees, focal_length_mm, image_height, image_width
        else:
            print("No metadata found.")
            return None, None, None, None, None, None, None, None, None, None, None, None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None, None, None, None, None, None, None, None, None

def dms_to_decimal(dms_str):
    parts = dms_str.split()
    degrees = float(parts[0])
    minutes = float(parts[2].replace("'", ""))
    seconds = float(parts[3].replace('"', ""))
    direction = parts[4]

    decimal = degrees + (minutes / 60) + (seconds / 3600)

    if direction in ['S', 'W']:
        decimal = -decimal

    return decimal

def extract_number(input_string):
    if input_string is None:
        return None
    input_string = str(input_string)

    match = re.search(r"[-+]?\d*\.\d+|\d+", input_string)
    if match:
        return float(match.group())
    return None

def get_gps_from_pixel(pixel_x, pixel_y, image_width, image_height, flight_degree, gimbal_degree, gps_lat_decimal, gps_lon_decimal, altitude_meters, focal_length_mm, sensor_width_mm, sensor_height_mm):
    fov_rad_h = 2 * math.atan((sensor_width_mm / (2 * focal_length_mm)))
    fov_rad_v = 2 * math.atan((sensor_height_mm / (2 * focal_length_mm)))

    ground_width_meters = 2 * altitude_meters * math.tan(fov_rad_h / 2)
    ground_height_meters = 2 * altitude_meters * math.tan(fov_rad_v / 2)

    gsd_meters_per_pixel_x = ground_width_meters / image_width
    gsd_meters_per_pixel_y = ground_height_meters / image_height

    corrected_pixel_x = pixel_x - (image_width / 2)
    corrected_pixel_y = (image_height / 2) - pixel_y

    corrected_lon_change = corrected_pixel_x * gsd_meters_per_pixel_x
    corrected_lat_change = corrected_pixel_y * gsd_meters_per_pixel_y

    gimbal_radians = math.radians(float(gimbal_degree))

    lon_change = corrected_lon_change * math.cos(gimbal_radians) + corrected_lat_change * math.sin(gimbal_radians)
    lat_change = -corrected_lon_change * math.sin(gimbal_radians) + corrected_lat_change * math.cos(gimbal_radians)

    latitude = gps_lat_decimal + (lat_change / 111320)
    longitude = gps_lon_decimal + (lon_change / (40008000 * math.cos(math.radians(latitude)) / 360))

    return latitude, longitude

def get_gps_from_pixel(pixel_x, pixel_y, image_width, image_height, flight_degree, gimbal_degree, gps_lat_decimal, gps_lon_decimal, altitude_meters, focal_length_mm, sensor_width_mm, sensor_height_mm):
    fov_rad_h = 2 * math.atan((sensor_width_mm / (2 * focal_length_mm)))
    fov_rad_v = 2 * math.atan((sensor_height_mm / (2 * focal_length_mm)))

    ground_width_meters = 2 * altitude_meters * math.tan(fov_rad_h / 2)
    ground_height_meters = 2 * altitude_meters * math.tan(fov_rad_v / 2)

    gsd_meters_per_pixel_x = ground_width_meters / image_width
    gsd_meters_per_pixel_y = ground_height_meters / image_height

    corrected_pixel_x = pixel_x - (image_width / 2)
    corrected_pixel_y = (image_height / 2) - pixel_y

    corrected_lon_change = corrected_pixel_x * gsd_meters_per_pixel_x
    corrected_lat_change = corrected_pixel_y * gsd_meters_per_pixel_y

    gimbal_radians = math.radians(float(gimbal_degree))

    lon_change = corrected_lon_change * math.cos(gimbal_radians) + corrected_lat_change * math.sin(gimbal_radians)
    lat_change = -corrected_lon_change * math.sin(gimbal_radians) + corrected_lat_change * math.cos(gimbal_radians)

    latitude = gps_lat_decimal + (lat_change / 111320)
    longitude = gps_lon_decimal + (lon_change / (40008000 * math.cos(math.radians(latitude)) / 360))

    return latitude, longitude

def get_pixel_from_gps(latitude, longitude, flight_degree, gimbal_degree, image_width, image_height, gsd_x, gsd_y, gps_lat_decimal, gps_lon_decimal):
    lat_change = (latitude - gps_lat_decimal) * 111320
    lon_change = (longitude - gps_lon_decimal) * (40008000 * math.cos(math.radians(latitude)) / 360)

    gimbal_radians = math.radians(float(gimbal_degree))

    corrected_lon_change = lon_change * math.cos(gimbal_radians) - lat_change * math.sin(gimbal_radians)
    corrected_lat_change = lon_change * math.sin(gimbal_radians) + lat_change * math.cos(gimbal_radians)

    corrected_pixel_x = corrected_lon_change / gsd_x
    corrected_pixel_y = corrected_lat_change / gsd_y

    pixel_x = (image_width / 2) + corrected_pixel_x
    pixel_y = (image_height / 2) - corrected_pixel_y

    return int(pixel_x), int(pixel_y)

def process_image(image_path, df, output_folder):
    img = Image.open(image_path)
    image_width, image_height = img.size

    focal_length_mm = 4.5
    fov_deg = 73.7
    sensor_width_mm = 6.17
    sensor_height_mm = 4.55

    flight_yaw_degree, flight_pitch_degree, flight_roll_degree, gimbal_yaw_degree, gimbal_pitch_degree, gimbal_roll_degree, gps_latitude, gps_longitude, gps_altitude, fov_degrees, focal_length_mm_exif, image_height_exif, image_width_exif = extract_exif(image_path)

    flight_yaw_num = extract_number(flight_yaw_degree)
    flight_pitch_num = extract_number(flight_pitch_degree)
    flight_roll_num = extract_number(flight_roll_degree)
    gimbal_yaw_num = extract_number(gimbal_yaw_degree)
    gimbal_pitch_num = extract_number(gimbal_pitch_degree)
    gimbal_roll_num = extract_number(gimbal_roll_degree)
    gps_altitude_num = extract_number(gps_altitude)
    fov_degrees_num = extract_number(fov_degrees)
    focal_length_mm_exif = extract_number(focal_length_mm_exif)

    fov_rad_h = 2 * math.atan((sensor_width_mm / (2 * focal_length_mm)))
    fov_rad_v = 2 * math.atan((sensor_height_mm / (2 * focal_length_mm)))

    ground_width_meters = 2 * gps_altitude_num * math.tan(fov_rad_h / 2)
    ground_height_meters = 2 * gps_altitude_num * math.tan(fov_rad_v / 2)

    gsd_meters_per_pixel_x = ground_width_meters / image_width
    gsd_meters_per_pixel_y = ground_height_meters / image_height

    for index, row in df.iterrows():
        vine_lat, vine_lon = row['Latitude'], row['Longitude']
        vine_id = row['Vine ID']

        pixel_x, pixel_y = get_pixel_from_gps(vine_lat, vine_lon, flight_yaw_num, gimbal_yaw_num, image_width, image_height, gsd_meters_per_pixel_x, gsd_meters_per_pixel_y, gps_latitude, gps_longitude)

        # one_meter_pixels_x = XX / gsd_meters_per_pixel_x ' change XX to desired sirwe of cropeed image
        one_meter_pixels_x = 1 / gsd_meters_per_pixel_x # X m squared area in pixels
        one_meter_pixels_y = 1 / gsd_meters_per_pixel_y # X m squared area in pixels

        crop_left = int(pixel_x - (one_meter_pixels_x / 2))
        crop_top = int(pixel_y - (one_meter_pixels_y / 2))
        crop_right = int(pixel_x + (one_meter_pixels_x / 2))
        crop_bottom = int(pixel_y + (one_meter_pixels_y / 2))

        # Check if entire crop area is within image bounds
        if crop_left < 0 or crop_top < 0 or crop_right > image_width or crop_bottom > image_height:
            print(f"Skipping crop for {vine_id} in {os.path.basename(image_path)}: Crop area outside image bounds.")
            continue  # Skip to the next vine

        cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))

        output_crop_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_{vine_id}.png")
        cropped_img.save(output_crop_path, "PNG")
        print(f"Cropped and saved: {output_crop_path}")

if __name__ == "__main__":
    image_folder = "../images/riseholme/august_2024/100_feet/"
    csv_file = "../data/riseholme_vineyard_lat_long_vines.csv"

    output_folder = "../images/output/vines_from_tile_images/1m_square_images/august_2024_100_feet"
    os.makedirs(output_folder, exist_ok=True)

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        exit()

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            process_image(image_path, df, output_folder)
            print(f"Processed image: {filename}")

    print("Image processing complete.")