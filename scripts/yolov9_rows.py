import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import json
import image_gps_pixel_show_poles

# Load YOLO model
model = YOLO('../weights/vine_row_segmentation/best.pt')

# Load image and predict
input_image = '../images/riseholme/august_2024/39_feet/DJI_20240802142942_0034_W.JPG'
results = model.predict(input_image, conf=0.2)

# Open original image
img = Image.open(input_image).convert("RGB")
width, height = img.size

# Create an empty mask (black image)
mask_image = np.zeros((height, width), dtype=np.uint8)

# Process results
for result in results:
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # Convert masks to numpy array
        for mask in masks:
            mask = (mask * 255).astype(np.uint8)  # Scale mask to 0-255
            mask_resized = cv2.resize(mask, (width, height))  # Resize to match input image size
            
            # Add mask to the global mask for merging
            mask_image = np.maximum(mask_image, mask_resized)

# Find merged contours (polygons)
contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
merged_polygons = [contour.reshape(-1, 2).tolist() for contour in contours]

# Convert mask to RGB red overlay
mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)  # Black image
mask_rgb[..., 0] = mask_image  # Red channel

# Convert to PIL Image
mask_overlay = Image.fromarray(mask_rgb)

# Blend images (alpha = 0.5 for transparency)
blended = Image.blend(img, mask_overlay, alpha=0.5)

# Save the final overlay
output_path = "../images/segmentation_overlay.png"
blended.save(output_path)

print(f"Saved segmentation overlay to {output_path}")

# Print merged polygons
print("Merged Polygons:", merged_polygons)
print("Total Merged Polygons:", len(merged_polygons))

flight_yaw_degree, flight_pitch_degree, flight_roll_degree, gimbal_yaw_degree, gimbal_pitch_degree, gimbal_roll_degree, gps_latitude, gps_longitude, gps_altitude, fov_degrees, image_height, image_width = image_gps_pixel_show_poles.extract_exif(input_image)

if flight_yaw_degree is not None:
    # Extracting numeric values
    flight_yaw_num = image_gps_pixel_show_poles.extract_number(flight_yaw_degree)
    flight_pitch_num = image_gps_pixel_show_poles.extract_number(flight_pitch_degree)
    flight_roll_num = image_gps_pixel_show_poles.extract_number(flight_roll_degree)
    gimbal_yaw_num = image_gps_pixel_show_poles.extract_number(gimbal_yaw_degree)
    gimbal_pitch_num = image_gps_pixel_show_poles.extract_number(gimbal_pitch_degree)
    gimbal_roll_num = image_gps_pixel_show_poles.extract_number(gimbal_roll_degree)
    gps_altitude_num = image_gps_pixel_show_poles.extract_number(gps_altitude)
    fov_degrees_num = image_gps_pixel_show_poles.extract_number(fov_degrees)

    print(f"Flight Yaw Degree: {flight_yaw_num}")
    print(f"Flight Pitch Degree: {flight_pitch_num}")
    print(f"Flight Roll Degree: {flight_roll_num}")
    print(f"Gimbal Yaw Degree: {gimbal_yaw_num}")
    print(f"Gimbal Pitch Degree: {gimbal_pitch_num}")
    print(f"Gimbal Roll Degree: {gimbal_roll_num}")
    print(f"GPS Latitude (Decimal): {gps_latitude}")
    print(f"GPS Longitude (Decimal): {gps_longitude}")
    print(f"GPS Altitude: {gps_altitude_num}")
    print(f"Field of View: {fov_degrees_num}")
    
    # Open the image
    img = Image.open(input_image)
    image_width, image_height = img.size

    # Camera specifications
    focal_length_mm = 4.5
    fov_deg = 73.7
    sensor_width_mm = 11.04 # 6.3

    # Initialize GeoJSON FeatureCollection structure for merged polygons
    polygon_geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }

    for polygon in merged_polygons:
        converted_polygon = []
        for point in polygon:
            x_pixel, y_pixel = point
            latitude, longitude = image_gps_pixel_show_poles.get_gps_from_pixel(
                int(x_pixel), int(y_pixel), 
                image_width, image_height, 
                flight_yaw_num, gimbal_yaw_num, 
                gps_latitude, gps_longitude, gps_altitude_num, 
                fov_degrees_num, sensor_width_mm
            )
            converted_polygon.append([longitude, latitude])  # GeoJSON format [lon, lat]

        # Ensure the polygon is closed (first and last coordinates must be the same)
        if converted_polygon[0] != converted_polygon[-1]:
            converted_polygon.append(converted_polygon[0])  # Close the polygon

        # Add the converted polygon to GeoJSON
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [converted_polygon]  # List of lists for GeoJSON polygons
            },
            "properties": {
                "type": "vine_row"  # You can add more metadata here
            }
        }
        polygon_geojson_data["features"].append(feature)

# Save the GeoJSON data to a file
output_polygon_geojson_file = "../data/detected_merged_vine_rows.geojson"
with open(output_polygon_geojson_file, "w") as json_file:
    json.dump(polygon_geojson_data, json_file, indent=4)

print(f"Merged polygon lat/long coordinates saved to: {output_polygon_geojson_file}")

