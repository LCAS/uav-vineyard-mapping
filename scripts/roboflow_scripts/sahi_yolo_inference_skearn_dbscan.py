import cv2
import json
import os
import time
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from PIL import Image

# --- New Imports for Slicing ---
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- NEW: Clustering Imports ---
from sklearn.cluster import DBSCAN

# Assuming your local module exists
import image_gps_pixel_show_poles
# import pole_clustering  <-- We are replacing this logic locally below

# --- Helper Functions ---

def cluster_poles_dbscan(geojson_data, distance_threshold_meters=1.5, min_samples=2):
    """
    Clusters GPS points using DBSCAN with Haversine metric.
    
    Args:
        geojson_data: The input FeatureCollection dictionary.
        distance_threshold_meters: The maximum distance (radius) to consider points as the same post.
        min_samples: Minimum detections required to confirm a post.
    """
    features = geojson_data['features']
    if not features:
        return geojson_data

    # 1. Extract coordinates (lon, lat) from GeoJSON
    coords = []
    for f in features:
        lon, lat = f['geometry']['coordinates']
        # sklearn haversine metric requires [latitude, longitude] order
        coords.append([lat, lon]) 
    
    # Convert to radians for Haversine
    coords_rad = np.radians(coords)

    # 2. Run DBSCAN
    # Earth radius in meters ~ 6371000. 
    # We divide our meter threshold by this to get radians.
    kms_per_radian = 6371.0088
    epsilon = (distance_threshold_meters / 1000.0) / kms_per_radian

    print(f"Running DBSCAN with eps={distance_threshold_meters}m...")
    db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='haversine', algorithm='ball_tree').fit(coords_rad)
    
    cluster_labels = db.labels_
    
    # 3. Merge Clusters into Centroids
    clustered_features = []
    unique_labels = set(cluster_labels)
    
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"  -> Found {num_clusters} unique posts from {len(features)} raw detections.")

    for label in unique_labels:
        if label == -1:
            # -1 represents "noise" (points that didn't have enough neighbors)
            # You can choose to keep them or discard them. Here we discard noise.
            continue

        # Get mask for all points in this cluster
        mask = (cluster_labels == label)
        points_in_cluster = np.array(coords)[mask]
        
        # Calculate Centroid (average lat/lon)
        centroid = points_in_cluster.mean(axis=0)
        mean_lat, mean_lon = centroid[0], centroid[1]
        
        # Calculate average confidence
        confidences = [features[i]['properties']['confidence'] for i in range(len(features)) if mask[i]]
        avg_conf = float(np.mean(confidences))
        detection_count = int(np.sum(mask))

        clustered_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [mean_lon, mean_lat] # GeoJSON standard is [lon, lat]
            },
            "properties": {
                "type": "pole",
                "confidence": avg_conf,
                "detection_count": detection_count
            }
        })

    return {"type": "FeatureCollection", "features": clustered_features}

def erode_polygon(polygon, erosion_distance):
    return polygon.buffer(-erosion_distance)

def merge_cluster(polygon, polygons, visited):
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
            return list(merged_cluster.geoms) if isinstance(merged_cluster, MultiPolygon) else [merged_cluster]
    return []

def mask_to_coordinates(prediction):
    """
    Converts a SAHI prediction mask (boolean) into a list of (x,y) polygon coordinates
    mapped to the original image space.
    """
    if not prediction.mask:
        return []

    bool_mask = prediction.mask.bool_mask
    mask_uint8 = (bool_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    largest_contour = max(contours, key=cv2.contourArea)
    bbox = prediction.bbox
    offset_x = bbox.minx
    offset_y = bbox.miny

    poly_coords = []
    for point in largest_contour:
        x, y = point[0]
        poly_coords.append((int(x + offset_x), int(y + offset_y)))

    return poly_coords

def annotate_and_save_image(image_path, object_prediction_list, output_dir):
    """
    Draws bounding boxes and segmentation masks on the full-res image and saves it.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image for annotation: {image_path}")
        return

    overlay = image.copy()
    COLOR_POLE = (0, 0, 255)      # Red for Poles
    COLOR_VINE = (0, 255, 0)      # Green for Vine Rows
    alpha = 0.4

    for prediction in object_prediction_list:
        if prediction.category.name == "pole":
            bbox = prediction.bbox
            x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
            cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_POLE, 3)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_POLE, -1)

        elif prediction.category.name == "vine_row" and prediction.mask:
            poly_coords = mask_to_coordinates(prediction)
            if len(poly_coords) > 2:
                pts = np.array(poly_coords, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], COLOR_VINE)
                cv2.polylines(image, [pts], True, COLOR_VINE, 2)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"annotated_{filename}")
    cv2.imwrite(save_path, image)
    print(f"Annotated image saved to: {save_path}")


# --- Main Detection Logic ---

def detect_poles_and_vine_rows(image_file, detection_model, confidence, sensor_width_mm, sensor_height_mm, focal_length_mm, slice_height=640, slice_width=640, annotated_images_output_dir=None):
    # 1. Run Sliced Prediction with Error Handling
    try:
        result = get_sliced_prediction(
            image_file,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0,
            # --- NEW SETTINGS TO FIX CRASH ---
            postprocess_type="NMS",        # Use NMS instead of merging to avoid invalid masks
            postprocess_match_metric="IOS", # Intersection Over Smaller is better for thin objects
            postprocess_match_threshold=0.5
        )
    except ValueError as e:
        print(f"Skipping {image_file} due to SAHI processing error: {e}")
        return [], []
    except Exception as e:
        print(f"Skipping {image_file} due to unexpected error: {e}")
        return [], []

    object_prediction_list = result.object_prediction_list

    if annotated_images_output_dir:
        try:
            annotate_and_save_image(image_file, object_prediction_list, annotated_images_output_dir)
        except Exception as e:
            print(f"Warning: Failed to save annotated image for {image_file}: {e}")

    # 2. Parse Results
    center_pixels = []
    vine_rows_points = []

    for prediction in object_prediction_list:
        class_name = prediction.category.name
        score = prediction.score.value

        if score < confidence:
            continue

        if class_name == "pole":
            bbox = prediction.bbox
            cx = bbox.minx + ((bbox.maxx - bbox.minx) / 2)
            cy = bbox.miny + ((bbox.maxy - bbox.miny) / 2)
            center_pixels.append({"center_x": cx, "center_y": cy, "confidence": score})

        elif class_name == "vine_row":
            if prediction.mask:
                poly_coords = mask_to_coordinates(prediction)
                if len(poly_coords) > 2:
                    vine_rows_points.append({"vine_row": class_name, "points": poly_coords, "confidence": score})

    # 3. Extract EXIF
    flight_yaw_degree, flight_pitch_degree, flight_roll_degree, gimbal_yaw_degree, gimbal_pitch_degree, gimbal_roll_degree, gps_latitude, gps_longitude, gps_altitude, fov_degrees, exif_focal_length, image_height, image_width = image_gps_pixel_show_poles.extract_exif(image_file)

    if flight_yaw_degree is None:
        print(f"Skipping {image_file} - No valid EXIF")
        return [], []

    flight_yaw_num = image_gps_pixel_show_poles.extract_number(flight_yaw_degree)
    gimbal_yaw_num = image_gps_pixel_show_poles.extract_number(gimbal_yaw_degree)
    if gimbal_yaw_num == 0.0: gimbal_yaw_num = flight_yaw_degree
    
    gps_altitude_num = image_gps_pixel_show_poles.extract_number(gps_altitude)
    
    if gps_altitude_num is None:
        gps_altitude_num = 0.0

    # Image size check
    img = Image.open(image_file)
    image_width, image_height = img.size

    all_pole_coordinates = []
    all_vine_row_coordinates = []

    # Process Poles
    for center_pixel in center_pixels:
        latitude, longitude = image_gps_pixel_show_poles.get_gps_from_pixel(
            int(center_pixel["center_x"]), int(center_pixel["center_y"]),
            image_width, image_height, flight_yaw_num, gimbal_yaw_num,
            gps_latitude, gps_longitude, gps_altitude_num,
            focal_length_mm, sensor_width_mm, sensor_height_mm
        )
        all_pole_coordinates.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
            "properties": {"type": "pole", "confidence": center_pixel["confidence"]}
        })

    # Process Vine Rows
    for vine_row in vine_rows_points:
        points = vine_row["points"]
        vine_row_coordinates_poly = []

        for point in points:
             lat, lon = image_gps_pixel_show_poles.get_gps_from_pixel(
                int(point[0]), int(point[1]), image_width, image_height,
                flight_yaw_num, gimbal_yaw_num, gps_latitude, gps_longitude,
                gps_altitude_num, focal_length_mm, sensor_width_mm, sensor_height_mm
            )
             vine_row_coordinates_poly.append([lon, lat])

        all_vine_row_coordinates.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [vine_row_coordinates_poly]},
            "properties": {"type": "vine_row", "confidence": vine_row["confidence"]}
        })

    return all_pole_coordinates, all_vine_row_coordinates

# --- Main Execution ---

if __name__ == "__main__":
    # image_folder = "../../images/agri_tech_centre/jojo/"
    # output_folder = "../../data/jojo/vineyard_segmentation-21/train3_inference_results/"
    image_folder = "../../images/riseholme/july_2025/39_feet/"
    output_folder = "../../data/riseholme/vineyard_segmentation-22/train5_inference_results/july_2025/39_feet"
    # annotated_images_output_dir = os.path.join(output_folder, "annotated_images")
    annotated_images_output_dir = None # dont save annotated images

    # Create output directories
    os.makedirs(output_folder, exist_ok=True)
    
    # Only attempt to create the debug folder if the variable is NOT None
    if annotated_images_output_dir:
        os.makedirs(annotated_images_output_dir, exist_ok=True)

    model_path = "../../data/datasets/trained/vineyard_segmentation-21/train3/weights/best.pt"
    confidence = 0.4
    slice_height = 760 # 1520 # 760 # 640
    slice_width = 1014 # 2028 # 1014 # 640

    # # --- Camera specifications Riseholme (Zenmuse H20) --- 
    focal_length_mm = 4.5
    fov_deg = 73.7
    sensor_width_mm = 6.07 # UPDATED to match 4:3 image ratio
    sensor_height_mm = 4.55

    # # Camera specifications Agri tech centre drone P1 camera # https://enterprise.dji.com/zenmuse-p1
    # focal_length_mm = 35.0 # * 0.12
    # fov_deg = 63.5
    # sensor_width_mm = 35.9
    # sensor_height_mm = 24.0

    print(f"Loading model from {model_path}...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', 
        model_path=model_path,
        confidence_threshold=confidence,
        device="cuda:0" 
    )

    geojson_data_poles = {"type": "FeatureCollection", "features": []}
    geojson_data_vine_rows = {"type": "FeatureCollection", "features": []}
    all_pole_features = []
    all_vine_features = []

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    start_time = time.time()

    for i, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing {i}/{len(image_files)}: {image_file}")

        try:
            # Passed focal_length_mm explicitly here
            poles, rows = detect_poles_and_vine_rows(
                image_path, detection_model, confidence, sensor_width_mm, sensor_height_mm, focal_length_mm,
                slice_height=slice_height, slice_width=slice_width,
                annotated_images_output_dir=annotated_images_output_dir 
            )
            all_pole_features.extend(poles)
            all_vine_features.extend(rows)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            import traceback
            traceback.print_exc() 
            continue

    print(f"Inference completed in {time.time() - start_time:.2f} seconds")

    geojson_data_poles["features"] = all_pole_features
    geojson_data_vine_rows["features"] = all_vine_features

    # --- UPDATED CLUSTERING LOGIC ---
    print("Clustering poles using DBSCAN (sklearn)...")
    try:
        # Distance is in meters. Adjust 1.5 if you need looser/tighter clusters.
        geojson_data_poles_clustered = cluster_poles_dbscan(
            geojson_data_poles, distance_threshold_meters=1.5, min_samples=2
        )
    except Exception as e:
        print(f"Clustering failed: {e}, saving unclustered only.")
        geojson_data_poles_clustered = geojson_data_poles
    # --------------------------------

    with open(os.path.join(output_folder, "detected_pole_coordinates.geojson"), "w") as f:
        json.dump(geojson_data_poles, f, indent=4)

    with open(os.path.join(output_folder, "detected_clustered_pole_coordinates.geojson"), "w") as f:
        json.dump(geojson_data_poles_clustered, f, indent=4)

    with open(os.path.join(output_folder, "detected_vine_row_coordinates.geojson"), "w") as f:
        json.dump(geojson_data_vine_rows, f, indent=4)

    print(f"GeoJSON files and annotated images saved to: {output_folder}")
    print("Done.")