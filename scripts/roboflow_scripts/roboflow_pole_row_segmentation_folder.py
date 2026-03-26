'''
The code performs pole detection (for one folder of images) using a pre-trained YOLOv8 model, extracts GPS coordinates for detected poles, 
converts pixel locations to geographic coordinates (latitude/longitude), and saves the results in a GeoJSON file. 
It also generates an annotated image showing the detected poles.
'''

from inference import get_model
import supervision as sv
import cv2
import json
from PIL import Image, ImageDraw
import image_gps_pixel_show_poles
import os
from shapely.geometry import Polygon, MultiPolygon, LineString, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
import numpy as np
import shapely.geometry
import math
import pole_clustering
import time

def erode_polygon(polygon, erosion_distance):
    """Erodes a polygon by a given distance."""
    return polygon.buffer(-erosion_distance)

def merge_cluster(polygon, polygons, visited):
    """Recursively merges overlapping polygons."""
    cluster = [polygon]
    to_visit = [polygon]

    while to_visit:
        current_poly = to_visit.pop()
        for other_poly in polygons:
            if other_poly not in visited and current_poly.intersects(other_poly):
                cluster.append(other_poly)
                to_visit.append(other_poly)
                visited.add(other_poly)

    if cluster:
        valid_cluster = [p for p in cluster if p.is_valid and not p.is_empty]
        if valid_cluster:
            merged_cluster = unary_union(valid_cluster)
            if isinstance(merged_cluster, MultiPolygon):
                return list(merged_cluster.geoms) # return list of polygons.
            else:
                return [merged_cluster]
    return []

def polygon_iou(polygon1, polygon2):
    """Calculates the IoU of two shapely polygons."""
    try:
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        if union == 0:
            return 0.0
        return intersection / union
    except shapely.errors.GEOSException as e:
        print(f"Error calculating IoU: {e}")
        return 0.0

def validate_polygon(polygon_coordinates):
    """Validates and fixes a polygon."""
    if len(polygon_coordinates) < 4:
        print(f"Invalid polygon (less than 4 points): {polygon_coordinates}")
        return None  # Return None or an alternative placeholder
    
    polygon = shapely.geometry.Polygon(polygon_coordinates)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # Attempt to fix invalid geometries
    return polygon

def non_maximum_suppression_polygons(detections, iou_threshold):
    """Performs NMS for polygon masks."""
    sorted_detections = sorted(detections, key=lambda x: x["properties"]["confidence"], reverse=True)
    keep = []
    
    while sorted_detections:
        best_detection = sorted_detections.pop(0)
        best_polygon = validate_polygon(best_detection["geometry"]["coordinates"][0])

        if best_polygon is None:
            continue  # Skip invalid polygons

        keep.append(best_detection)
        sorted_detections_temp = []

        for d in sorted_detections:
            current_polygon = validate_polygon(d["geometry"]["coordinates"][0])

            if current_polygon is None:
                continue  # Skip invalid polygons

            if polygon_iou(best_polygon, current_polygon) < iou_threshold:
                sorted_detections_temp.append(d)

        sorted_detections = sorted_detections_temp

    return keep

def polygon_to_line(polygon_coords):
    # Convert to Shapely Polygon
    # polygon = Polygon(polygon_coords)
    
    # Get boundary coordinates (excluding duplicate last point)
    boundary_coords = list(polygon.exterior.coords[:-1])
    
    # Find the longest edge
    longest_edge = None
    max_length = 0
    for i in range(len(boundary_coords) - 1):
        p1, p2 = boundary_coords[i], boundary_coords[i + 1]
        length = np.linalg.norm(np.array(p2) - np.array(p1))
        if length > max_length:
            max_length = length
            longest_edge = (p1, p2)

    # Convert longest edge to LineString
    return LineString(longest_edge) if longest_edge else None

def detect_poles_and_vine_rows(image_file, model, confidence, sensor_width_mm, sensor_height_mm, output_folder):
    # Run inference on the chosen image
    image = cv2.imread(image_file)
    results = model.infer(image, confidence=confidence)[0]  # confidence=0.75, iou_threshold=0.5
    # print("Results:", results)

    # Extract center pixel coordinates for detected poles and vine rows
    center_pixels = [
        {"center_x": prediction.x, "center_y": prediction.y, "confidence": prediction.confidence}
        for prediction in results.predictions
        if prediction.class_name == "pole"
    ]

    vine_rows_points = [
        {"vine_row": prediction.class_name, "points": [(point.x, point.y) for point in prediction.points], "confidence": prediction.confidence}
        for prediction in results.predictions
        if prediction.class_name == "vine_row"
    ]

    # Extract EXIF data for image and GPS info
    flight_yaw_degree, flight_pitch_degree, flight_roll_degree, gimbal_yaw_degree, gimbal_pitch_degree, gimbal_roll_degree, gps_latitude, gps_longitude, gps_altitude, fov_degrees, focal_length_mm, image_height, image_width = image_gps_pixel_show_poles.extract_exif(image_path)

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
        focal_length_mm = image_gps_pixel_show_poles.extract_number(focal_length_mm)

        if gimbal_yaw_num == 0.0:
            gimbal_yaw_num = flight_yaw_degree

        if gimbal_pitch_num == 0.0:
            gimbal_pitch_num = flight_pitch_degree

        if gimbal_roll_num == 0.0:
            gimbal_roll_num = flight_roll_degree

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
        # print(f"Focal Length: {focal_length_mm}")

        # Open the image for size info
        img = Image.open(image_file)
        image_width, image_height = img.size

        all_pole_coordinates = []
        all_vine_row_coordinates = []

        # only use poles and  vine rows detected around the center of the image as the lat long calulation is not accurate for the edges of the image
        image_center_x = image_width / 2
        image_center_y = image_height / 2

        # Calculate rectangle boundaries
        rect_half_width = image_width / 4
        rect_half_height = image_height / 4

        rect_left = image_center_x - rect_half_width
        rect_right = image_center_x + rect_half_width
        rect_top = image_center_y - rect_half_height
        rect_bottom = image_center_y + rect_half_height

        # # Process detected poles without using the rectangle
        for center_pixel in center_pixels:
            latitude, longitude = image_gps_pixel_show_poles.get_gps_from_pixel(
                int(center_pixel["center_x"]), int(center_pixel["center_y"]), 
                image_width, image_height, flight_yaw_num, gimbal_yaw_num, 
                gps_latitude, gps_longitude, gps_altitude_num, 
                focal_length_mm, sensor_width_mm, sensor_height_mm
            )

            all_pole_coordinates.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [longitude, latitude]}, "properties": {"type": "pole", "confidence": center_pixel["confidence"]}})

        # # Process detected poles within the rectangle
        # filtered_center_pixels = []
        # for center_pixel in center_pixels:
        #     # Check if the pole is within the rectangle.
        #     if rect_left <= center_pixel["center_x"] <= rect_right and rect_top <= center_pixel["center_y"] <= rect_bottom:
        #         filtered_center_pixels.append(center_pixel)

        # for center_pixel in filtered_center_pixels:
        #     latitude, longitude = image_gps_pixel_show_poles.get_gps_from_pixel(
        #         int(center_pixel["center_x"]), int(center_pixel["center_y"]), 
        #         image_width, image_height, flight_yaw_num, gimbal_yaw_num, 
        #         gps_latitude, gps_longitude, gps_altitude_num, 
        #         focal_length_mm, sensor_width_mm, sensor_height_mm
        #     )

        #     all_pole_coordinates.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [longitude, latitude]}, "properties": {"type": "pole", "confidence": center_pixel["confidence"]}})

        # # Process detected vine rows an dont clip them to the rectangle
        # filtered_vine_rows_points = []
        # for vine_row in vine_rows_points:
        #     points = vine_row["points"]

        #     # Check if points list is empty
        #     if not points:
        #         print(f"Warning: Empty points list for vine_row: {vine_row}")
        #         continue  # Skip to the next vine_row

        #     try:
        #         # Calculate the center of the mask.
        #         center_x = sum(p[0] for p in points) / len(points)
        #         center_y = sum(p[1] for p in points) / len(points)
        #         detection_center = (center_x, center_y)

        #         # Calculate rectangle boundary centered on image center
        #         rect_half_width = image_width / 2
        #         rect_half_height = image_height / 2

        #         rect_left = image_center_x - rect_half_width
        #         rect_right = image_center_x + rect_half_width
        #         rect_top = image_center_y - rect_half_height
        #         rect_bottom = image_center_y + rect_half_height

        #         # Check if the detection is within the rectangle.
        #         if rect_left <= detection_center[0] <= rect_right and rect_top <= detection_center[1] <= rect_bottom:
        #             filtered_vine_rows_points.append(vine_row)

        #     except ZeroDivisionError:
        #         print(f"Error: ZeroDivisionError for vine_row: {vine_row}")
        #         continue  # Skip to the next vine_row

        # # Process detected vine rows and clip them to the rectangle
        # filtered_vine_rows_points = []
        # for vine_row in vine_rows_points:
        #     points = vine_row["points"]

        #     # Check if points list is empty
        #     if not points:
        #         print(f"Warning: Empty points list for vine_row: {vine_row}")
        #         continue  # Skip to the next vine_row

        #     try:
        #         # Create a Shapely Polygon from the points
        #         polygon = Polygon(points)

        #         # Calculate rectangle boundary centered on image center
        #         rect_half_width = image_width / 2
        #         rect_half_height = image_height / 2

        #         rect_left = image_center_x - rect_half_width
        #         rect_right = image_center_x + rect_half_width
        #         rect_top = image_center_y - rect_half_height
        #         rect_bottom = image_center_y + rect_half_height

        #         # Create a Shapely Polygon for the rectangle
        #         rectangle = box(rect_left, rect_top, rect_right, rect_bottom)

        #         # Calculate the intersection of the polygon and the rectangle
        #         intersection = polygon.intersection(rectangle)

        #         # Check if the intersection is not empty (meaning there's overlap)
        #         if not intersection.is_empty:
        #             # If there's an intersection, get the coordinates of the clipped polygon
        #             if intersection.geom_type == 'Polygon':
        #                 clipped_points = list(intersection.exterior.coords)
        #                 # Remove the last coordinate as it duplicates the first for a closed polygon
        #                 clipped_points = clipped_points[:-1]
        #                 filtered_vine_rows_points.append({"points": clipped_points, "confidence": vine_row["confidence"]})
        #             elif intersection.geom_type == 'MultiPolygon':
        #                 # Handle cases where the intersection results in multiple polygons
        #                 for geom in intersection.geoms:
        #                     clipped_points = list(geom.exterior.coords)
        #                     clipped_points = clipped_points[:-1]
        #                     filtered_vine_rows_points.append({"points": clipped_points, "confidence": vine_row["confidence"]})
        #             else:
        #                 # Handle other intersection types if needed (e.g., LineString, Point)
        #                 print(f"Warning: Intersection resulted in a {intersection.geom_type}, skipping.")

        #     except ZeroDivisionError:
        #         print(f"Error: ZeroDivisionError for vine_row: {vine_row}")
        #         continue  # Skip to the next vine_row
        #     except Exception as e:
        #         print(f"Error processing polygon: {e}")
        #         continue

        # # Now, use filtered_vine_rows_points for coordinate conversion
        # all_vine_row_coordinates = []
        # for vine_row in filtered_vine_rows_points:
        #     vine_row_coordinates = []
        #     for point in vine_row["points"]:
        #         vine_row_point_latitude, vine_row_point_longitude = image_gps_pixel_show_poles.get_gps_from_pixel(
        #             int(point[0]), int(point[1]), image_width, image_height,
        #             flight_yaw_num, gimbal_yaw_num, gps_latitude, gps_longitude,
        #             gps_altitude_num, focal_length_mm, sensor_width_mm, sensor_height_mm
        #         )

        #         vine_row_coordinates.append([vine_row_point_longitude, vine_row_point_latitude])

        #     all_vine_row_coordinates.append({
        #         "type": "Feature",
        #         "geometry": {"type": "Polygon", "coordinates": [vine_row_coordinates]},
        #         "properties": {"type": "vine_row", "confidence": vine_row["confidence"]}
        #     })

        # # Load detections and annotate the image
        # detections = sv.Detections.from_inference(results)
        # if len(detections) > 0:
        #     mask_annotator = sv.MaskAnnotator()
        #     label_annotator = sv.LabelAnnotator()

        #     annotated_image = mask_annotator.annotate(scene=image, detections=detections)
        #     annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        #     # Save the annotated image
        #     filename = os.path.basename(image_file)
        #     cv2.imwrite(f"../../images/output/row_detection/annotated_{filename}", annotated_image)
        #     print(f"Annotated image saved to: ../../images/output/row_detection/annotated_{filename}")

        return all_pole_coordinates, all_vine_row_coordinates

if __name__ == "__main__":
    image_folder="../../images/riseholme/august_2024/39_feet/" # Riseholme
    # image_folder="../../images/riseholme/august_2024/65_feet/" # Riseholme
    # image_folder="../../images/riseholme/august_2024/100_feet/" # Riseholme
    # image_folder="../../images/riseholme/march_2025/39_feet/" # Riseholme
    # image_folder="../../images/riseholme/march_2025/65_feet/" # Riseholme
    # image_folder="../../images/riseholme/march_2025/100_feet/" # Riseholme

    # image_folder="../../images/jojo/agri_tech_centre/RX1RII/"

    # image_folder="../../images/outfields/wraxall/topdown/rgb/"
    # image_folder="../../images/outfields/jojo/topdown/"
    # image_folder="../../images/outfields/jojo/high_altitude_oblique/"
    # image_folder="../../images/outfields/jojo/low_altitude_side_view/"

    # image_folder="../../images/agri_tech_centre/jojo/"
    # image_folder="../../images/agri_tech_centre/arun_1/"
    # image_folder="../../images/agri_tech_centre/arun_2/"
    # image_folder="../../images/agri_tech_centre/coolhurst/"

    output_folder = "../../data/"
    
    api_key_file = '../../config/api_key.json'
    model_id="vineyard_segmentation/11"
    confidence=0.4

    # Camera specifications Riseholme # https://enterprise.dji.com/zenmuse-h20-series/specs
    focal_length_mm = 4.5
    fov_deg = 73.7
    sensor_width_mm =  6.17
    sensor_height_mm = 4.55

    # # Camera specifications Agri tech centre jojo drone # https://www.sony.co.uk/electronics/cyber-shot-compact-cameras/dsc-rx1rm2
    # focal_length_mm = 35.0 # * 0.12
    # fov_deg = 54.4
    # sensor_width_mm = 35.9
    # sensor_height_mm = 24.0

    # # Camera specifications Agri tech centre drone P1 camera # https://enterprise.dji.com/zenmuse-p1
    # focal_length_mm = 35.0 # * 0.12
    # fov_deg = 63.5
    # sensor_width_mm = 35.9
    # sensor_height_mm = 24.0
    
    # Camera specifications Outfields drone DJI Mavic 3 Multispectral (M3M) 1/2.3 inch wide sensor
    # focal_length_mm = 12.3
    # fov_deg = 73.7
    # sensor_width_mm = 17.4
    # sensor_height_mm = 13.0

    # Initialize GeoJSON structures
    geojson_data_poles = {"type": "FeatureCollection", "features": []}
    geojson_data_vine_rows = {"type": "FeatureCollection", "features": []}

    all_pole_coordinates = []
    all_vine_row_coordinates = []

    # Load the API key
    with open(api_key_file, 'r') as file:
        config = json.load(file)
    ROBOFLOW_API_KEY = config.get("ROBOFLOW_API_KEY")

    # Load a pre-trained YOLOv model
    model = get_model(model_id=model_id, api_key=ROBOFLOW_API_KEY)

    # Get a list of all image files
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)

    start_time = time.time()
        
    for i, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing image {i} of {total_images}: {image_file}")

        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image files
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # Call the detection function
        pole_coordinates, vine_row_coordinates = detect_poles_and_vine_rows(image_path, model, confidence, sensor_width_mm, sensor_height_mm, output_folder)
        all_pole_coordinates.extend(pole_coordinates)
        all_vine_row_coordinates.extend(vine_row_coordinates)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken {elapsed_time:.6f} seconds")

    for pole in all_pole_coordinates:
        geojson_data_poles["features"].append(pole)

    for vine_row in all_vine_row_coordinates:
        geojson_data_vine_rows["features"].append(vine_row)

    # Cluster poles
    geojson_data_poles_clustered = pole_clustering.cluster_poles(geojson_data_poles, eps=0.0000002, min_samples=2, metric="chebyshev") # eps=0.0000002
   
    # iou_threshold = 0.3
    # nms_results = non_maximum_suppression_polygons(all_vine_row_coordinates, iou_threshold)

    # # Convert GeoJSON coordinates to shapely polygons
    # all_polygons = []
    # for feature in nms_results: # nms_results or all_vine_row_coordinates
    #     if feature["geometry"]["type"] == "Polygon":
    #         coordinates = feature["geometry"]["coordinates"][0]
    #         if len(coordinates) >= 4:  # Check for at least 4 coordinates
    #             polygon = Polygon(feature["geometry"]["coordinates"][0])

    #             # Fix invalid polygons
    #             if not polygon.is_valid or polygon.is_empty:
    #                 polygon = polygon.buffer(0)

    #             if polygon.is_valid and not polygon.is_empty:
    #                 all_polygons.append(polygon)
    #     elif feature["geometry"]["type"] == "MultiPolygon":
    #         for poly_coords in feature["geometry"]["coordinates"]:
    #             polygon = Polygon(poly_coords[0])
    #             if not polygon.is_valid or polygon.is_empty:
    #                 polygon = polygon.buffer(0)
    #             if polygon.is_valid and not polygon.is_empty:
    #                 all_polygons.append(polygon)

    # # # **Step 1: Filter out small polygons**
    # # if all_polygons:
    # #     average_area = sum(p.area for p in all_polygons) / len(all_polygons)
    # #     min_area_threshold = 0.5 * average_area  # Half the average area
    # #     filtered_polygons = [p for p in all_polygons if p.area >= min_area_threshold]
    # # else:
    # #     filtered_polygons = []

    # filtered_polygons = all_polygons

    # # # Step 1.2: Erode polygons
    # # erosion_distance = 0.0000001
    # # eroded_polygons = []
    # # for poly in filtered_polygons:
    # #     eroded_poly = erode_polygon(poly, erosion_distance)
    # #     if eroded_poly.is_valid and not eroded_poly.is_empty:
    # #         eroded_polygons.append(eroded_poly)

    # # Step 2: Cluster overlapping polygons together
    # merged_vine_rows = []
    # if filtered_polygons:
    #     str_tree = STRtree(filtered_polygons)
    #     visited = set()

    #     for poly in filtered_polygons:
    #         if poly in visited:
    #             continue

    #         merged_result = merge_cluster(poly, filtered_polygons, visited)
    #         merged_vine_rows.extend(merged_result)

    # # Step 3: Convert merged polygons back into GeoJSON format
    # merged_geojson_list = []
    # for merged_polygon in merged_vine_rows:
    #     if isinstance(merged_polygon, Polygon) and not merged_polygon.is_empty:
    #         merged_geojson_list.append({
    #             "type": "Feature",
    #             "geometry": {
    #                 "type": "Polygon",
    #                 "coordinates": [list(merged_polygon.exterior.coords)]
    #             },
    #             "properties": {"type": "merged_vine_row"}
    #         })

    # merged_geojson = {"type": "FeatureCollection", "features": merged_geojson_list}
    
    # for vine_row in all_vine_row_coordinates:     
    #     geojson_data_vine_rows["features"].append(vine_row)

    # Save GeoJSON data
    with open(f"{output_folder}detected_pole_coordinates.geojson", "w") as json_file:
        json.dump(geojson_data_poles, json_file, indent=4)

    with open(f"{output_folder}detected_clustered_pole_coordinates.geojson", "w") as json_file:
        json.dump(geojson_data_poles_clustered, json_file, indent=4)

    with open(f"{output_folder}detected_vine_row_coordinates.geojson", "w") as json_file:
        json.dump(geojson_data_vine_rows, json_file, indent=4)

    # with open(f"{output_folder}detected_merged_vine_rows.geojson", "w") as json_file:
    #     json.dump(merged_geojson, json_file, indent=4)

    print(f"GeoJSON files saved to: {output_folder}")