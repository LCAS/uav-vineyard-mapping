import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import json
import image_gps_pixel_show_poles
import os
from shapely.geometry import shape
from shapely.ops import unary_union

def process_vine_row_segmentation(image_folder, output_folder, model_path, sensor_width_mm):
    # Load YOLO model
    model = YOLO(model_path)
    os.makedirs(output_folder, exist_ok=True)
    
    # Get a list of all image files
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    
    # Initialize GeoJSON FeatureCollection structure
    polygon_geojson_data = {"type": "FeatureCollection", "features": []}
    
    for i, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing image {i} of {total_images}: {image_file}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        results = model.predict(image_path, conf=0.4)
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        mask_image = np.zeros((height, width), dtype=np.uint8)        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                for i, class_id in enumerate(result.boxes.cls):
                    if result.names[int(class_id)] == 'vine_row':
                        mask = (masks[i] * 255).astype(np.uint8)
                        mask_resized = cv2.resize(mask, (width, height))
                        mask_image = np.maximum(mask_image, mask_resized)
        
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        merged_polygons = [contour.reshape(-1, 2).tolist() for contour in contours]
        
        flight_yaw_degree, flight_pitch_degree, flight_roll_degree, gimbal_yaw_degree, gimbal_pitch_degree, gimbal_roll_degree, gps_latitude, gps_longitude, gps_altitude, fov_degrees, image_height, image_width = image_gps_pixel_show_poles.extract_exif(image_path)        
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
                    converted_polygon.append([longitude, latitude])
                
                if converted_polygon[0] != converted_polygon[-1]:
                    converted_polygon.append(converted_polygon[0])
                
                feature = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [converted_polygon]}, "properties": {"type": "vine_row"}}
                polygon_geojson_data["features"].append(feature)
    
    polygons = [shape(feature['geometry']) for feature in polygon_geojson_data['features'] if feature['geometry']['type'] == 'Polygon']
    merged_polygon = unary_union(polygons) if polygons else None
    
    merged_geojson_data = {"type": "FeatureCollection", "features": []}
    if merged_polygon:
        filtered_polygons = list(merged_polygon.geoms) if merged_polygon.geom_type == 'MultiPolygon' else [merged_polygon]
        if filtered_polygons:
            avg_area = sum(poly.area for poly in filtered_polygons) / len(filtered_polygons)
            min_area_threshold = 0.5 * avg_area
            filtered_polygons = [poly for poly in filtered_polygons if poly.area >= min_area_threshold]
            for poly in filtered_polygons:
                merged_geojson_data["features"].append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [list(poly.exterior.coords)]}, "properties": {"type": "merged_vine_row"}})
    
    return merged_geojson_data

if __name__ == "__main__":
    image_folder = "../images/riseholme/august_2024/39_feet/"
    output_folder = "../images/output/row_detection/"
    model_path = "../weights/vine_row_segmentation/best.pt"
    # model_path = "../weights/vineyard_segmentation_v7_weights.pt"
    
    # Camera specifications
    focal_length_mm = 4.5
    sensor_width_mm = 11.04  # 6.3
    fov_degrees = 73.7
    
    merged_geojson_data = process_vine_row_segmentation(image_folder, output_folder, model_path, sensor_width_mm)
    output_polygon_geojson_file = "../data/detected_merged_vine_rows.geojson"
    
    with open(output_polygon_geojson_file, "w") as json_file:
        json.dump(merged_geojson_data, json_file, indent=4)
    
    print(f"Merged & filtered polygon data saved to: {output_polygon_geojson_file}")
