import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import json
import image_gps_pixel_show_poles
import os
from shapely.geometry import shape
from shapely.ops import unary_union

# Load YOLO model
model = YOLO('../weights/vineyard_segmentation_v7_weights.pt')

# Define the folder containing images for inference
image_folder = "../images/riseholme/august_2024/39_feet/"

# Get a list of all image files
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
total_images = len(image_files)  # Get total number of images

# Initialize GeoJSON FeatureCollection structure
polygon_geojson_data = {
    "type": "FeatureCollection",
    "features": []
}

# Initialize GeoJSON FeatureCollection structure
pole_geojson_data = {
    "type": "FeatureCollection",
    "features": []
}

# Camera specifications
focal_length_mm = 4.5
sensor_width_mm = 11.04  # 6.3
fov_degrees = 73.7

poles_coodinates = []

# Process each image in the folder
for i, image_file in enumerate(image_files, start=1):
    image_path = os.path.join(image_folder, image_file)
    
    print(f"Processing image {i} of {total_images}: {image_file}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    results = model.predict(image_path, conf=0.6, iou=0.8, device='0')

    # Open original image
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # Create an empty mask (black image)
    mask_image = np.zeros((height, width), dtype=np.uint8)

    # Process results
    for result in results:
        if result.masks is not None or result.boxes is not None:
            # Ensure masks is not None before proceeding
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()  # Convert masks to numpy array
            else:
                masks = None  # Set masks to None if not available
            
            for i, class_id in enumerate(result.boxes.cls):  # Iterate through detected class IDs
                class_name = result.names[int(class_id)]  # Get class name

                if class_name == "pole":  # Filter only "pole" class
                    pole_x1, pole_y1, pole_x2, pole_y2 = map(int, result.boxes.xyxy[i])  # Convert to int
                    
                    # Calculate the center of the box
                    center_x = (pole_x1 + pole_x2) / 2
                    center_y = (pole_y1 + pole_y2) / 2
                    poles_coodinates.append({"center_x": center_x, "center_y": center_y})

                elif class_name == 'vine_row' and masks is not None:  # Only process vine rows if masks are available
                    mask = masks[i]  # Get the mask for this 'vine_row' detection
                    mask = (mask * 255).astype(np.uint8)  # Scale mask to 0-255
                    mask_resized = cv2.resize(mask, (width, height))  # Resize to match input image size
                    
                    # Add the 'vine_row' mask to the global mask image
                    mask_image = (mask_image + mask_resized) / 2

    # Scale the image to 0-255 and convert to uint8 (after all masks are added)
    mask_image = (mask_image * 255).astype(np.uint8)

    # Find merged contours (polygons)
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged_polygons = [contour.reshape(-1, 2).tolist() for contour in contours]

    # # Convert mask to RGB red overlay
    # mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)  # Black image
    # mask_rgb[..., 0] = mask_image  # Red channel

    # # Convert to PIL Image
    # mask_overlay = Image.fromarray(mask_rgb)

    # # Blend images (alpha = 0.5 for transparency)
    # blended = Image.blend(img, mask_overlay, alpha=0.5)

    # # Save the final overlay
    # output_path = "../images/segmentation_overlay.png"
    # blended.save(output_path)

    # print(f"Saved segmentation overlay to {output_path}")

    # # Print merged polygons
    # print("Merged Polygons:", merged_polygons)
    # print("Total Merged Polygons:", len(merged_polygons))

    flight_yaw_degree, flight_pitch_degree, flight_roll_degree, gimbal_yaw_degree, gimbal_pitch_degree, gimbal_roll_degree, gps_latitude, gps_longitude, gps_altitude, fov_degrees, image_height, image_width = image_gps_pixel_show_poles.extract_exif(image_path)

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

        # print(f"Flight Yaw Degree: {flight_yaw_num}")
        # print(f"Flight Pitch Degree: {flight_pitch_num}")
        # print(f"Flight Roll Degree: {flight_roll_num}")
        # print(f"Gimbal Yaw Degree: {gimbal_yaw_num}")
        # print(f"Gimbal Pitch Degree: {gimbal_pitch_num}")
        # print(f"Gimbal Roll Degree: {gimbal_roll_num}")
        # print(f"GPS Latitude (Decimal): {gps_latitude}")
        # print(f"GPS Longitude (Decimal): {gps_longitude}")
        # print(f"GPS Altitude: {gps_altitude_num}")
        # print(f"Field of View: {fov_degrees_num}")
        
        # Open the image
        img = Image.open(image_path)
        image_width, image_height = img.size

        for pole in poles_coodinates:
            latitude, longitude = image_gps_pixel_show_poles.get_gps_from_pixel(
                int(pole["center_x"]), int(pole["center_y"]), 
                image_width, image_height, 
                flight_yaw_num, gimbal_yaw_num, 
                gps_latitude, gps_longitude, gps_altitude_num, 
                fov_degrees_num, sensor_width_mm
            )

            # Append to GeoJSON
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [longitude, latitude]
                },
                "properties": {
                    "type": "pole"
                }
            }
            pole_geojson_data["features"].append(feature)

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

# In geojson data merge overlapping polygons
# Convert GeoJSON features into Shapely geometries
polygons = []
for feature in polygon_geojson_data['features']:
    geometry = feature['geometry']
    if geometry['type'] == 'Polygon':
        shapely_polygon = shape(geometry)
        polygons.append(shapely_polygon)

# Merge overlapping polygons
merged_polygon = unary_union(polygons) if polygons else None

# Create a new GeoJSON feature collection for the merged polygons
merged_geojson_data = {
    "type": "FeatureCollection",
    "features": []
}

# Handle the case where the result is either a Polygon or MultiPolygon
filtered_polygons = []
if merged_polygon:
    if merged_polygon.geom_type == 'MultiPolygon':
        filtered_polygons = list(merged_polygon.geoms)  # Extract individual polygons
    elif merged_polygon.geom_type == 'Polygon':
        filtered_polygons = [merged_polygon]  # Single polygon case

# Calculate the average area
if filtered_polygons:
    average_area = sum(poly.area for poly in filtered_polygons) / len(filtered_polygons)
    min_area_threshold = 0.5 * average_area  # Set threshold as half of the average area

    # Filter polygons based on area threshold
    filtered_polygons = [poly for poly in filtered_polygons if poly.area >= min_area_threshold]

    # Convert filtered polygons back to GeoJSON format
    for poly in filtered_polygons:
        merged_geojson_data["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(poly.exterior.coords)]  # Extract exterior coordinates
            },
            "properties": {
                "type": "merged_vine_row"
            }
        })

# Save the filtered GeoJSON data to a file
output_polygon_geojson_file = "../data/detected_merged_vine_rows.geojson"
with open(output_polygon_geojson_file, "w") as json_file:
    json.dump(merged_geojson_data, json_file, indent=4)

print(f"Merged & filtered polygon data saved to: {output_polygon_geojson_file}")

# Save the GeoJSON data to a file
# output_geojson_file = os.path.join(output_folder, "detected_pole_coordinates.geojson")
output_geojson_file = "../data/detected_pole_coordinates.geojson"
with open(output_geojson_file, "w") as json_file:
    json.dump(pole_geojson_data, json_file, indent=4)

print(f"Poles GeoJSON file saved to: {output_geojson_file}")