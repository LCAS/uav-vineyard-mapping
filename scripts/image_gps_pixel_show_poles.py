'''
This script extracts metadata from an image, GPS coordinates, altitude, and camera orientation, and then uses this data to calculate pixel locations for geographical points and draw circles on the image corresponding to these locations.

Key components:
extract_exif(image_path): This function extracts metadata from an image file using ExifTool. It retrieves information such as GPS coordinates (latitude, longitude), altitude, field of view, and camera angles (yaw, pitch, roll) for both the flight and the gimbal.

dms_to_decimal(dms_str): Converts GPS coordinates in Degrees, Minutes, and Seconds (DMS) format to decimal degrees.

extract_number(input_string): This utility function extracts the first numeric value from a string using regular expressions.

get_gps_from_pixel(...): Converts pixel coordinates from an image into GPS coordinates based on the image's field of view, altitude, and orientation of the camera.

get_pixel_from_gps(...): Converts GPS coordinates back into pixel coordinates on the image. This also adjusts for the gimbal's orientation and the field of view.

draw_circles_on_image(...): This function draws circles on the image based on a list of GPS coordinates. It calculates pixel positions for these coordinates and draws red circles at those positions.

process_image(image_path, gps_points): The main function that uses the above utilities to process an image. It extracts metadata, computes pixel coordinates for given GPS points, and draws circles at those positions on the image. Saves locations in a GeoJSON file.
'''

import subprocess
import json
import re
from PIL import Image, ImageDraw
import math
import piexif

def extract_exif(image_path):
    try:
        # Run ExifTool with -json option to extract metadata
        result = subprocess.run(['exiftool', '-json', image_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None, None, None, None, None

        # Parse the JSON output
        metadata = json.loads(result.stdout)
        if metadata:
            metadata_dict = metadata[0]  # Assuming the first item contains the metadata
            
            # Extract required fields
            flight_yaw_degree = metadata_dict.get('FlightYawDegree', None) # DO NOT USE FlightYawDegree use GimbalYawDegree, GimbalYawDegree is the compass heading of the camera and image in degrees.
            flight_pitch_degree = metadata_dict.get('FlightPitchDegree', None)
            flight_roll_degree = metadata_dict.get('FlightRollDegree', None)
            gimbal_yaw_degree = metadata_dict.get('GimbalYawDegree', 0) # GimbalYawDegree is the compass heading of the camera and image in degrees.
            gimbal_pitch_degree = metadata_dict.get('GimbalPitchDegree', 0)
            gimbal_roll_degree = metadata_dict.get('GimbalRollDegree', 0)
            gps_latitude_dms = metadata_dict.get('GPSLatitude', None)
            gps_longitude_dms = metadata_dict.get('GPSLongitude', None)
            gps_altitude = metadata_dict.get('RelativeAltitude', None)
            fov_degrees = metadata_dict.get('FOV', None)
            focal_length_mm = metadata_dict.get('FocalLength', None)
            image_height = metadata_dict.get('ImageHeight', None)
            image_width = metadata_dict.get('ImageWidth', None)

            # print(f"Flight Yaw Degree: {flight_yaw_degree}")
            # print(f"Flight Pitch Degree: {flight_pitch_degree}")
            # print(f"Flight Roll Degree: {flight_roll_degree}")
            # print(f"Gimbal Yaw Degree: {gimbal_yaw_degree}")
            # print(f"Gimbal Pitch Degree: {gimbal_pitch_degree}")
            # print(f"Gimbal Roll Degree: {gimbal_roll_degree}")
            # print(f"GPS Latitude (DMS): {gps_latitude_dms}")
            # print(f"GPS Longitude (DMS): {gps_longitude_dms}")
            # print(f"GPS Altitude: {gps_altitude}")
            # print(f"Field of View: {fov_degrees}")
            # print(f"Image Height: {image_height}")
            # print(f"Image Width: {image_width}")
            
            # Convert DMS to Decimal Degrees
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
    """
    Convert a DMS (degree, minute, second) string to decimal degrees.
    
    :param dms_str: DMS string (e.g., '53 deg 16\' 5.36" N')
    :return: Decimal degrees (e.g., 53.2681556)
    """
    # Split the string into components
    parts = dms_str.split()
    degrees = float(parts[0])
    minutes = float(parts[2].replace("'", ""))
    seconds = float(parts[3].replace('"', ""))
    direction = parts[4]

    # Convert to decimal degrees
    decimal = degrees + (minutes / 60) + (seconds / 3600)

    # Apply negative sign for south or west
    if direction in ['S', 'W']:
        decimal = -decimal

    return decimal

# Function to extract numeric values
def extract_number(input_string):
    # Ensure input_string is a string before processing
    if input_string is None:
        return None
    input_string = str(input_string)

    # Use regular expression to extract the first number in the string
    match = re.search(r"[-+]?\d*\.\d+|\d+", input_string)
    if match:
        return float(match.group())
    return None

# Function to get latitude and longitude for a pixel
def get_gps_from_pixel(pixel_x, pixel_y, image_width, image_height, 
                       flight_degree, gimbal_degree, gps_lat_decimal, gps_lon_decimal, 
                       altitude_meters, focal_length_mm, sensor_width_mm, sensor_height_mm):
    """
    Convert pixel coordinates back to GPS latitude and longitude.
    
    Args:
    - pixel_x, pixel_y: The pixel coordinates in the image.
    - image_width, image_height: The dimensions of the image in pixels.
    - flight_degree: The flight yaw orientation in degrees.
    - gimbal_degree: The gimbal yaw orientation in degrees.
    - gps_lat_decimal, gps_lon_decimal: GPS coordinates (latitude and longitude) of the image center.
    - altitude_meters: The altitude of the drone in meters.
    - focal_length_mm: Camera focal length in millimeters.
    - sensor_width_mm, sensor_height_mm: Camera sensor size in millimeters.
    
    Returns:
    - (latitude, longitude): The GPS coordinates corresponding to the pixel location.
    """

    # **Calculate Horizontal and Vertical FOV using Focal Length**
    fov_rad_h = 2 * math.atan((sensor_width_mm / (2 * focal_length_mm)))  
    fov_rad_v = 2 * math.atan((sensor_height_mm / (2 * focal_length_mm)))

    # **Calculate Ground Coverage**
    ground_width_meters = 2 * altitude_meters * math.tan(fov_rad_h / 2)
    ground_height_meters = 2 * altitude_meters * math.tan(fov_rad_v / 2)

    # **Calculate Ground Sample Distance (GSD)**
    gsd_meters_per_pixel_x = ground_width_meters / image_width
    gsd_meters_per_pixel_y = ground_height_meters / image_height

    # Offset the pixel coordinates relative to the center of the image
    corrected_pixel_x = pixel_x - (image_width / 2)
    corrected_pixel_y = (image_height / 2) - pixel_y  # Invert Y-axis for image coordinates

    # Convert pixel offsets to real-world distances in meters
    corrected_lon_change = corrected_pixel_x * gsd_meters_per_pixel_x
    corrected_lat_change = corrected_pixel_y * gsd_meters_per_pixel_y

    # Convert gimbal orientation to radians
    gimbal_radians = math.radians(float(gimbal_degree))

    # Reverse the displacement adjustments for gimbal orientation
    lon_change = corrected_lon_change * math.cos(gimbal_radians) + corrected_lat_change * math.sin(gimbal_radians)
    lat_change = -corrected_lon_change * math.sin(gimbal_radians) + corrected_lat_change * math.cos(gimbal_radians)

    # Convert the real-world displacements back to degrees
    latitude = gps_lat_decimal + (lat_change / 111320)  # Convert meters to degrees for latitude
    longitude = gps_lon_decimal + (lon_change / (40008000 * math.cos(math.radians(latitude)) / 360))  # Convert meters to degrees for longitude

    return latitude, longitude


def get_pixel_from_gps(latitude, longitude, flight_degree, gimbal_degree, 
                       image_width, image_height, 
                       gsd_x, gsd_y,  # Separate GSD values for x and y
                       gps_lat_decimal, gps_lon_decimal):
    
    # Calculate the real-world displacement in meters from the given lat/lon to the center point
    lat_change = (latitude - gps_lat_decimal) * 111320  # Approximate conversion to meters for latitude
    lon_change = (longitude - gps_lon_decimal) * (40008000 * math.cos(math.radians(latitude)) / 360)  # Conversion to meters for longitude

    # Convert gimbal orientation to radians
    gimbal_radians = math.radians(float(gimbal_degree))

    # Adjust the displacement for gimbal orientation
    corrected_lon_change = lon_change * math.cos(gimbal_radians) - lat_change * math.sin(gimbal_radians)
    corrected_lat_change = lon_change * math.sin(gimbal_radians) + lat_change * math.cos(gimbal_radians)

    # Adjust pixel distance using separate real-world distance per pixel values (GSD)
    corrected_pixel_x = corrected_lon_change / gsd_x
    corrected_pixel_y = corrected_lat_change / gsd_y

    # Convert to pixel coordinates (adjusting for the center of the image)
    pixel_x = (image_width / 2) + corrected_pixel_x
    pixel_y = (image_height / 2) - corrected_pixel_y  # Invert Y-axis for typical image coordinates

    return int(pixel_x), int(pixel_y)

# Function to draw circles on the image based on lat/lon coordinates
def draw_circles_on_image(image_path, gps_points, flight_degree, gimbal_degree, 
                          gps_latitude, gps_longitude, altitude_meters, 
                          fov_deg, focal_length_mm, sensor_width_mm, sensor_height_mm):

    # Open the image
    img = Image.open(image_path)
    image_width, image_height = img.size
    draw = ImageDraw.Draw(img)

    flight_degree = float(flight_degree) if flight_degree else 0.0
    gimbal_degree = float(gimbal_degree) if gimbal_degree else 0.0

    # **Calculate Horizontal and Vertical FOV using Focal Length**
    fov_rad_h = 2 * math.atan((sensor_width_mm / (2 * focal_length_mm)))  
    fov_rad_v = 2 * math.atan((sensor_height_mm / (2 * focal_length_mm)))

    # **Calculate Ground Coverage using FOV and Altitude**
    ground_width_meters = 2 * altitude_meters * math.tan(fov_rad_h / 2)
    ground_height_meters = 2 * altitude_meters * math.tan(fov_rad_v / 2)

    # **Calculate Ground Sample Distance (GSD)**
    gsd_meters_per_pixel_x = ground_width_meters / image_width
    gsd_meters_per_pixel_y = ground_height_meters / image_height

    pixels = []

    for lat, lon in gps_points:
        # Convert lat/lon to pixel coordinates using accurate scaling
        pixel_x, pixel_y = get_pixel_from_gps(lat, lon, flight_degree, gimbal_degree, 
                                              image_width, image_height, 
                                              gsd_meters_per_pixel_x, gsd_meters_per_pixel_y,
                                              gps_latitude, gps_longitude)

        # Draw a circle at the calculated pixel coordinates
        radius = 20  
        draw.ellipse([pixel_x - radius, pixel_y - radius, pixel_x + radius, pixel_y + radius], 
                     outline="red", width=10)
        print("Circle drawn at:", pixel_x, pixel_y)

        pixels.append({"x": pixel_x, "y": pixel_y})

    # Save the modified image with circles
    output_image_path = "../images/output_image_with_circles.jpg"
    img.save(output_image_path)

    return pixels

# Example of using pixel_x and pixel_y in the main function
def process_image(image_path, gps_points):
    # Open the image
    img = Image.open(image_path)
    image_width, image_height = img.size

    # # Camera specifications Riseholme # https://enterprise.dji.com/zenmuse-h20-series/specs
    # focal_length_mm = 4.5
    # fov_deg = 73.7
    # sensor_width_mm =  6.17
    # sensor_height_mm = 4.55

    # Camera specifications Outfields drone DJI Mavic 3 Multispectral (M3M) 1/2.3 inch wide sensor
    # focal_length_mm = 12.3
    # fov_deg = 73.7
    # sensor_width_mm = 17.4
    # sensor_height_mm = 13.0

    # Camera specifications Agri tech centre drone P1 camera # https://enterprise.dji.com/zenmuse-p1
    focal_length_mm = 35.0 # * 0.12
    fov_deg = 63.5
    sensor_width_mm = 35.9
    sensor_height_mm = 24.0
    
    flight_yaw_degree, flight_pitch_degree, flight_roll_degree, gimbal_yaw_degree, gimbal_pitch_degree, gimbal_roll_degree, gps_latitude, gps_longitude, gps_altitude, fov_degrees, focal_length_mm, image_height, image_width = extract_exif(image_path)

    flight_yaw_num = extract_number(flight_yaw_degree)
    flight_pitch_num = extract_number(flight_pitch_degree)
    flight_roll_num = extract_number(flight_roll_degree)
    gimbal_yaw_num = extract_number(gimbal_yaw_degree)
    gimbal_pitch_num = extract_number(gimbal_pitch_degree)
    gimbal_roll_num = extract_number(gimbal_roll_degree)
    gps_altitude_num = extract_number(gps_altitude)
    fov_degrees_num = extract_number(fov_degrees)
    focal_length_mm = extract_number(focal_length_mm)

    # print(f"Flight Degree: {flight_yaw_num}")
    # print(f"Flight Pitch Degree: {flight_pitch_num}")
    # print(f"Flight Roll Degree: {flight_roll_num}")    
    # print(f"Gimbal Degree: {gimbal_yaw_num}")
    # print(f"Gimbal Pitch Degree: {gimbal_pitch_num}")
    # print(f"Gimbal Roll Degree: {gimbal_roll_num}")
    # print(f"GPS Coordinates: Latitude = {gps_latitude}, Longitude = {gps_longitude}")
    # print(f"GPS Altitude: {gps_altitude_num}")
    # print(f"Field of View: {fov_degrees_num}")

    # Draw cirels on image where the poles are
    pole_pixels = draw_circles_on_image(image_path, gps_points, flight_yaw_num, gimbal_yaw_num, gps_latitude, gps_longitude, gps_altitude_num, fov_degrees_num, focal_length_mm, sensor_width_mm, sensor_height_mm)
    # TODO fix gps to pixel in some rotations.

    # Initialize GeoJSON FeatureCollection structure
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }

    # pixels = [
    #         {'x': 0, 'y': 0}, 
    #         {'x': 4056, 'y': 3040}, 
    #         {'x': 2028, 'y': 1520}, 
    #         {'x': 100, 'y': 100}, 
    #         {'x': 950, 'y': 950}, 
    #         {'x': 2000, 'y': 2000}]

    pixels = pole_pixels

    for pixel in pixels:
        latitude, longitude = get_gps_from_pixel(pixel["x"], pixel["y"], image_width, image_height, flight_yaw_num, gimbal_yaw_num, gps_latitude, gps_longitude, gps_altitude_num, focal_length_mm, sensor_width_mm, sensor_height_mm)

        # Create a feature for each pole with lat/long coordinates
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [longitude, latitude]  # GeoJSON uses [longitude, latitude]
            },
            "properties": {
                "type": "pole"  # You can add more properties if needed
            }
        }
        
        # Append the feature to the features list
        geojson_data["features"].append(feature)

        print(f"Pixel ({pixel['x']}, {pixel['y']}) -> Latitude: {latitude}, Longitude: {longitude}")

    # Save the GeoJSON data to a file
    output_geojson_file = "../data/detected_pole_coordinates.geojson"
    with open(output_geojson_file, "w") as json_file:
        json.dump(geojson_data, json_file, indent=4)

if __name__ == "__main__":
    # image_path = "../images/riseholme/august_2024/39_feet/DJI_20240802143112_0076_W.JPG"
    # image_path = "../images/agri_tech_centre/jojo/DJI_20250331152541_0196.JPG"
    image_path = "../images/agri_tech_centre/jojo/DJI_20250331160730_0594.JPG"

    # Load the uploaded GeoJSON file
    geojson_path = "../data/Detected_PostPositions_UTM_shifted.geojson"
    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)

    # Extract coordinates from the GeoJSON features
    gps_points = []
    for feature in geojson_data["features"]:
        coords = feature["geometry"]["coordinates"]
        # Assuming coordinates are in (lon, lat) and we want (lat, lon)
        gps_points.append((coords[1], coords[0]))

    # gps_points = [
    #         (53.26818842,-0.52427737),
    #         (53.26813837,-0.52426541),
    #         (53.26808856,-0.52425335),
    #         (53.26803849,-0.52424047),
    #         (53.26818522,-0.52431449),
    #         (53.26813509,-0.52430208),
    #         (53.26808532,-0.52428968),
    #         (53.26803515,-0.52427742),
    #         (53.26818187,-0.52435181),
    #         (53.26813158,-0.52433952),
    #         (53.26808211,-0.52432693),
    #         (53.26803182,-0.52431475),
    #         (53.26817882,-0.52438866),
    #         (53.26812848,-0.52437636),
    #         (53.26807903,-0.52436409),
    #         (53.26802873,-0.52435185),
    #         (53.26817541,-0.52442589),
    #         (53.26812517,-0.5244135),
    #         (53.26807555,-0.5244011),
    #         (53.2680255,-0.52438878),
    #         (53.26817238,-0.52446323),
    #         (53.26812194,-0.52445077),
    #         (53.26807253,-0.52443819),
    #         (53.26802228,-0.52442619),
    #         (53.26816928,-0.52449965),
    #         (53.26811864,-0.52448766),
    #         (53.26806932,-0.52447579),
    #         (53.26801926,-0.52446331),
    #         (53.26816599,-0.52453691),
    #         (53.26811528,-0.5245244),
    #         (53.26806603,-0.52451219),
    #         (53.26801591,-0.52449999),
    #         (53.26816264,-0.52457417),
    #         (53.2681122,-0.52456217),
    #         (53.26806294,-0.52454963),
    #         (53.26801275,-0.52453719),
    #         (53.26815947,-0.52461139),
    #         (53.26810906,-0.52459885),
    #         (53.26805976,-0.52458653),
    #         (53.26800959,-0.52457471)#,

    #         #(53.26815, -0.524575), # centre of image to check if it is correct
    #         #(53.268175184088804, -0.5245453595031128) # random point
    # ]
        
    # image_path = "../images/outfields/jojo/topdown/DJI_20240618142122_0258_D.JPG"

    # with open("../data/jojo_row_posts_10_rows.geojson", 'r') as f:
    #     geojson_data = json.load(f)

    # gps_points = []
    # if geojson_data['type'] == 'FeatureCollection':
    #     for feature in geojson_data['features']:
    #         if feature['geometry']['type'] == 'Point':
    #             coordinates = feature['geometry']['coordinates']
    #             # GeoJSON stores coordinates as [longitude, latitude], so we reverse them.
    #             latitude, longitude = coordinates[1], coordinates[0]
    #             gps_points.append((latitude, longitude))
    # else:
    #     print("Error: GeoJSON file is not a FeatureCollection.")

    process_image(image_path, gps_points)