'''
The code performs pole detection (for one image) using a pre-trained YOLOv8 model, extracts GPS coordinates for detected poles, 
converts pixel locations to geographic coordinates (latitude/longitude), and saves the results in a GeoJSON file. 
It also generates an annotated image showing the detected poles.
'''

from inference import get_model
import supervision as sv
import cv2
import json
from PIL import Image, ImageDraw
import image_gps_pixel_show_poles

# Define the image URL to use for inference
image_file = "../../images/riseholme/august_2024/39_feet/DJI_20240802142844_0007_W.JPG"
# image_file = "../../images/jojo/agri_tech_centre/RX1RII/DSC00610.JPG"
image = cv2.imread(image_file)

# Load the API key
with open('../../config/api_key.json', 'r') as file:
    config = json.load(file)
ROBOFLOW_API_KEY = config.get("ROBOFLOW_API_KEY")

# Load a pre-trained YOLOv model
model = get_model(model_id="vineyard_test/4", api_key=ROBOFLOW_API_KEY)

# Run inference on our chosen image, image can be a URL, a NumPy array, a PIL image, etc.
results = model.infer(image)[0] # confidence=0.75, iou_threshold=0.5

print("Results:", results)

# Extract x and y coordinates of each 'pole'
coordinates = [
    {"x": prediction.x, "y": prediction.y}
    for prediction in results.predictions
    if prediction.class_name == "pole"
]

print("Pole Coordinates:", coordinates)

# Extract and display the center pixel for each pole
center_pixels = [
    {"center_x": prediction.x, "center_y": prediction.y}
    for prediction in results.predictions
    if prediction.class_name == "pole"
]

print("Center Pixels:", center_pixels)

flight_yaw_degree, flight_pitch_degree, flight_roll_degree, gimbal_yaw_degree, gimbal_pitch_degree, gimbal_roll_degree, gps_latitude, gps_longitude, gps_altitude, fov_degrees, image_height, image_width = image_gps_pixel_show_poles.extract_exif(image_file)

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
    img = Image.open(image_file)
    image_width, image_height = img.size

    # Camera specifications
    focal_length_mm = 4.5
    fov_deg = 73.7
    sensor_width_mm = 11.04 # 6.3

    # Initialize GeoJSON FeatureCollection structure
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }

    for center_pixel in center_pixels:        
        latitude, longitude = image_gps_pixel_show_poles.get_gps_from_pixel(int(center_pixel["center_x"]), int(center_pixel["center_y"]), 
                                                                            image_width, image_height, 
                                                                            flight_yaw_num, gimbal_yaw_num, 
                                                                            gps_latitude, gps_longitude, gps_altitude_num, 
                                                                            fov_degrees_num, sensor_width_mm)
        
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
        
        print(f"Latitude: {latitude}, Longitude: {longitude}")
    
    # Save the GeoJSON data to a file
    output_geojson_file = "../../data/detected_pole_coordinates.geojson"
    with open(output_geojson_file, "w") as json_file:
        json.dump(geojson_data, json_file, indent=4)

    print(f"Pole lat/long coordinates saved to: {output_geojson_file}")
        

# Load the results into the supervision Detections API
detections = sv.Detections.from_inference(results)

# Create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# Save the annotated image instead of displaying it
output_file = "../../images/annotated_image.jpg"
cv2.imwrite(output_file, annotated_image)
print(f"Annotated image saved to: {output_file}")
