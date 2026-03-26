'''
The code performs pole detection (for one folder of images) using a pre-trained YOLOv8 model, extracts GPS coordinates for detected poles, 
converts pixel locations to geographic coordinates (latitude/longitude), and saves the results in a GeoJSON file. 
It also generates an annotated image showing the detected poles.
'''

import os
from inference import get_model
import supervision as sv
import cv2
import json
from PIL import Image
import image_gps_pixel_show_poles
import numpy as np

# Define the folder containing images for inference
image_folder = "../../images/riseholme/august_2024/39_feet/"
output_folder = "../../images/output/"
os.makedirs(output_folder, exist_ok=True)

# Load the API key
with open('../../config/api_key.json', 'r') as file:
    config = json.load(file)
ROBOFLOW_API_KEY = config.get("ROBOFLOW_API_KEY")

# Load a pre-trained YOLOv8n model
model = get_model(model_id="vineyard_test/8", api_key=ROBOFLOW_API_KEY)

# Initialize GeoJSON FeatureCollection structure
geojson_data = {
    "type": "FeatureCollection",
    "features": []
}

vine_row_geojson_data = {
    "type": "FeatureCollection",
    "features": []
}

# Camera specifications
focal_length_mm = 4.5
sensor_width_mm = 11.04  # 6.3
fov_degrees = 73.7

# Process each image in the folder
for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image files
        continue
    
    print(f"Processing image: {image_file}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Run inference on the image
    results = model.infer(image)[0]

    # Extract pole coordinates
    coordinates = [
        {"x": prediction.x, "y": prediction.y}
        for prediction in results.predictions
        if prediction.class_name == "pole"
    ]

    # Extract vine row coordinates
    row_coordinates = [
        {"x": prediction.x, "y": prediction.y}
        for prediction in results.predictions
        if prediction.class_name == "vine_row"
    ]

    print(f"Pole Coordinates for {image_file}: {coordinates}")

    # Extract metadata from the image
    flight_yaw_degree, flight_pitch_degree, flight_roll_degree, \
    gimbal_yaw_degree, gimbal_pitch_degree, gimbal_roll_degree, \
    gps_latitude, gps_longitude, gps_altitude, fov_degrees, \
    image_height, image_width = image_gps_pixel_show_poles.extract_exif(image_path)

    if flight_yaw_degree is not None:
        flight_yaw_num = image_gps_pixel_show_poles.extract_number(flight_yaw_degree)
        gimbal_yaw_num = image_gps_pixel_show_poles.extract_number(gimbal_yaw_degree)
        gps_altitude_num = image_gps_pixel_show_poles.extract_number(gps_altitude)
        fov_degrees_num = image_gps_pixel_show_poles.extract_number(fov_degrees)

        for prediction in results.predictions:
            if prediction.class_name != "pole":
                latitude, longitude = image_gps_pixel_show_poles.get_gps_from_pixel(
                    int(prediction.x), int(prediction.y), 
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
                geojson_data["features"].append(feature)

            elif prediction.class_name == "vine_row" and prediction.mask is not None:
                mask = prediction.mask.astype(np.uint8) * 255  # Convert mask to binary image
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                print("contours: ", contours)

                converted_polygon = []
                for contour in contours:
                    for point in contour:
                        x, y = point[0]  # Extract pixel coordinates
                        lat, lon = image_gps_pixel_show_poles.get_gps_from_pixel(
                            int(x), int(y),
                            image_width, image_height,
                            flight_yaw_num, gimbal_yaw_num,
                            gps_latitude, gps_longitude, gps_altitude_num,
                            fov_degrees_num, sensor_width_mm
                        )
                        converted_polygon.append([lon, lat])  # GeoJSON expects [lon, lat]

                if converted_polygon:
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [converted_polygon]  # List of lists for GeoJSON polygons
                        },
                        "properties": {
                            "type": "vine_row"
                        }
                    }
                    vine_row_geojson_data["features"].append(feature)

    # Annotate the image
    detections = sv.Detections.from_inference(results)
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Save the annotated image
    annotated_image_path = os.path.join(output_folder, f"annotated_{image_file}")
    cv2.imwrite(annotated_image_path, annotated_image)
    print(f"Annotated image saved to: {annotated_image_path}")

# Save the GeoJSON data to a file
# output_geojson_file = os.path.join(output_folder, "detected_pole_coordinates.geojson")
output_geojson_file = "../../data/detected_pole_coordinates.geojson"
with open(output_geojson_file, "w") as json_file:
    json.dump(geojson_data, json_file, indent=4)

# Save the vine row GeoJSON data to a file
output_geojson_file_vine_row = "../../data/detected_vine_row_coordinates.geojson"
with open(output_geojson_file_vine_row, "w") as json_file:
    json.dump(vine_row_geojson_data, json_file, indent=4)

print(f"GeoJSON file saved to: {output_geojson_file}")
print(f"GeoJSON file saved to: {output_geojson_file_vine_row}")