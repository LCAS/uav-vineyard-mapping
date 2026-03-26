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

def detect_poles_and_vine_rows(image_file, model, sensor_width_mm, output_folder, geojson_data_poles, geojson_data_vine_rows):

    # Run inference on the chosen image
    image = cv2.imread(image_file)
    results = model.infer(image, confidence=0.2)[0]  # confidence=0.75, iou_threshold=0.5
    # print("Results:", results)

    # Extract center pixel coordinates for detected poles and vine rows
    center_pixels = [
        {"center_x": prediction.x, "center_y": prediction.y}
        for prediction in results.predictions
        if prediction.class_name == "pole"
    ]

    vine_rows_points = [
        {"vine_row": prediction.class_name, "points": [(point.x, point.y) for point in prediction.points]}
        for prediction in results.predictions
        if prediction.class_name == "vine_row"
    ]

    # Extract EXIF data for image and GPS info
    flight_yaw_degree, flight_pitch_degree, flight_roll_degree, gimbal_yaw_degree, gimbal_pitch_degree, gimbal_roll_degree, gps_latitude, gps_longitude, gps_altitude, fov_degrees, image_height, image_width = image_gps_pixel_show_poles.extract_exif(image_file)

    # Ensure valid EXIF data
    if flight_yaw_degree is not None:
        flight_yaw_num = image_gps_pixel_show_poles.extract_number(flight_yaw_degree)
        flight_pitch_num = image_gps_pixel_show_poles.extract_number(flight_pitch_degree)
        flight_roll_num = image_gps_pixel_show_poles.extract_number(flight_roll_degree)
        gimbal_yaw_num = image_gps_pixel_show_poles.extract_number(gimbal_yaw_degree)
        gimbal_pitch_num = image_gps_pixel_show_poles.extract_number(gimbal_pitch_degree)
        gimbal_roll_num = image_gps_pixel_show_poles.extract_number(gimbal_roll_degree)
        gps_altitude_num = image_gps_pixel_show_poles.extract_number(gps_altitude)
        fov_degrees_num = image_gps_pixel_show_poles.extract_number(fov_degrees)

        if gimbal_yaw_num == 0.0:
            gimbal_yaw_num = flight_yaw_degree

        if gimbal_pitch_num == 0.0:
            gimbal_pitch_num = flight_pitch_degree

        if gimbal_roll_num == 0.0:
            gimbal_roll_num = flight_roll_degree

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

        # Open the image for size info
        img = Image.open(image_file)
        image_width, image_height = img.size

        # Process detected poles
        for center_pixel in center_pixels:
            latitude, longitude = image_gps_pixel_show_poles.get_gps_from_pixel(
                int(center_pixel["center_x"]), int(center_pixel["center_y"]), 
                image_width, image_height, flight_yaw_num, gimbal_yaw_num, 
                gps_latitude, gps_longitude, gps_altitude_num, fov_degrees_num, 
                sensor_width_mm
            )

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [longitude, latitude]
                },
                "properties": {"type": "pole"}
            }
            geojson_data_poles["features"].append(feature)

        # Process detected vine rows
        for vine_row in vine_rows_points:
            vine_row_coordinates = []
            for point in vine_row["points"]:
                vine_row_point_latitude, vine_row_point_longitude = image_gps_pixel_show_poles.get_gps_from_pixel(
                    int(point[0]), int(point[1]), image_width, image_height, 
                    flight_yaw_num, gimbal_yaw_num, gps_latitude, gps_longitude, 
                    gps_altitude_num, fov_degrees_num, sensor_width_mm=11.04
                )
                vine_row_coordinates.append([vine_row_point_longitude, vine_row_point_latitude])

            # Ensure polygon is closed
            if vine_row_coordinates[0] != vine_row_coordinates[-1]:
                vine_row_coordinates.append(vine_row_coordinates[0])

            feature_vine_row = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [vine_row_coordinates]
                },
                "properties": {"type": "vine_row"}
            }
            geojson_data_vine_rows["features"].append(feature_vine_row)

        # # Load detections and annotate the image
        # detections = sv.Detections.from_inference(results)
        # mask_annotator = sv.MaskAnnotator()
        # label_annotator = sv.LabelAnnotator()

        # annotated_image = mask_annotator.annotate(scene=image, detections=detections)
        # annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # # Save the annotated image
        # cv2.imwrite(f"{output_folder}annotated_image.jpg", annotated_image)
        # print(f"Annotated image saved to: {output_folder}annotated_image.jpg")

        return geojson_data_poles, geojson_data_vine_rows

if __name__ == "__main__":
    image_file = "../../images/39_feet/DJI_20240802142844_0007_W.JPG" # Riseholme
    # image_file = "../../images/jojo/agri_tech_centre/RX1RII/DSC00610.JPG"
    # image_file = "../../images/outfields/wraxall/topdown/rgb/DJI_20241004151205_0014_D.JPG"

    output_folder = "../../data/"
    
    api_key_file = '../../config/api_key.json'
    model_id="vineyard_segmentation/7"

    # Camera specifications Riseholme drone
    focal_length_mm = 4.5
    fov_deg = 73.7
    sensor_width_mm = 11.04 # 6.3

    # Camera specifications Agri tech centre jojo drone
    # focal_length_mm = 35.0 # * 0.12
    # fov_deg = 54.4
    # sensor_width_mm = 35.9

    # Camera specifications Outfields Wraxall drone
    # focal_length_mm = 12.3
    # fov_deg = 73.7
    # sensor_width_mm = 3.9
    
    # Load the API key
    with open(api_key_file, 'r') as file:
        config = json.load(file)
    ROBOFLOW_API_KEY = config.get("ROBOFLOW_API_KEY")

    # Load a pre-trained YOLOv model
    model = get_model(model_id=model_id, api_key=ROBOFLOW_API_KEY)
   
    # Initialize GeoJSON structures
    geojson_data_poles = {"type": "FeatureCollection", "features": []}
    geojson_data_vine_rows = {"type": "FeatureCollection", "features": []}
    
    # Call the detection function
    geojson_data_poles, geojson_data_vine_rows = detect_poles_and_vine_rows(image_file, model, sensor_width_mm, output_folder, geojson_data_poles, geojson_data_vine_rows)   

    # Save GeoJSON data
    with open(f"{output_folder}detected_pole_coordinates.geojson", "w") as json_file:
        json.dump(geojson_data_poles, json_file, indent=4)

    with open(f"{output_folder}detected_vine_row_coordinates.geojson", "w") as json_file:
        json.dump(geojson_data_vine_rows, json_file, indent=4)

    print(f"GeoJSON files saved to: {output_folder}")
