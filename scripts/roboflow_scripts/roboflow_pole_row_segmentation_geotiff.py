from inference import get_model
import supervision as sv
import cv2
import json
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform
from rasterio.plot import reshape_as_image
import numpy as np
import pole_clustering

def detect_poles_and_vine_rows(image_file, model, geojson_data_poles, geojson_data_vine_rows):
    with rasterio.open(image_file) as src:
        image = src.read([3, 2, 1])
        image = reshape_as_image(image)
        transform_src = src.transform
        src_crs = src.crs

    results = model.infer(image, confidence=0.2)[0]

    detections = sv.Detections.from_inference(results)
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image.copy(), detections=detections)

    # cv2.imwrite(f"{output_folder}annotated_image.jpg", annotated_image)
    print(f"Annotated image saved to: {output_folder}annotated_image.jpg")

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

    # Correct placement for poles
    for center_pixel in center_pixels:
        x, y = xy(transform_src, center_pixel["center_y"], center_pixel["center_x"])
        lon, lat = transform(src_crs, 'EPSG:4326', [x], [y])
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon[0], lat[0]]
            },
            "properties": {"type": "pole"}
        }
        geojson_data_poles["features"].append(feature)

    # Cluster poles
    geojson_data_poles_clustered = pole_clustering.cluster_poles(geojson_data_poles, eps=0.0000002, min_samples=3)

    # Correct placement for vine rows
    for vine_row in vine_rows_points:
        vine_row_coordinates = []
        if not vine_row["points"]:  # Check if points list is empty
            continue  # Skip this vine row
        for point in vine_row["points"]:
            x, y = xy(transform_src, point[1], point[0])
            lon, lat = transform(src_crs, 'EPSG:4326', [x], [y])
            vine_row_coordinates.append([lon[0], lat[0]])

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

    return geojson_data_poles, geojson_data_poles_clustered, geojson_data_vine_rows

if __name__ == "__main__":
    image_file = "../../images/orthophoto/39_feet/odm_orthophoto.tif"
    # image_file = "../../images/orthophoto/65_feet/odm_orthophoto.tif"
    # image_file = "../../images/orthophoto/100_feet/odm_orthophoto.tif"
    # image_file = "../../images/orthophoto/jojo/agri_tech_centre/winter_2022/Vineyard_RGB_transparent_mosaic_group1.tif"
    # image_file = "../../images/orthophoto/outfields/wraxall/topdown/odm_orthophoto.tif"
    # image_file = "../../images/orthophoto/outfields/jojo/topdown/odm_orthophoto.tif"

    output_folder = "../../data/riseholme/orthophoto/39_feet/"
    api_key_file = '../../config/api_key.json'
    model_id="vineyard_segmentation/11"
    
    with open(api_key_file, 'r') as file:
        config = json.load(file)
    ROBOFLOW_API_KEY = config.get("ROBOFLOW_API_KEY")
    
    model = get_model(model_id=model_id, api_key=ROBOFLOW_API_KEY)
    
    geojson_data_poles = {"type": "FeatureCollection", "features": []}
    geojson_data_vine_rows = {"type": "FeatureCollection", "features": []}
    
    geojson_data_poles, geojson_data_poles_clustered, geojson_data_vine_rows = detect_poles_and_vine_rows(image_file, model, geojson_data_poles, geojson_data_vine_rows)

    # Reverse the polygon coordinates here
    if geojson_data_vine_rows["features"]: #check if the geojson has features.
        geojson_data_vine_rows["features"][0]["geometry"]["coordinates"][0].reverse()

    
    with open(f"{output_folder}detected_pole_coordinates.geojson", "w") as json_file:
        json.dump(geojson_data_poles, json_file, indent=4)

    with open(f"{output_folder}detected_clustered_pole_coordinates.geojson", "w") as json_file:
        json.dump(geojson_data_poles_clustered, json_file, indent=4)

    with open(f"{output_folder}detected_vine_row_coordinates.geojson", "w") as json_file:
        json.dump(geojson_data_vine_rows, json_file, indent=4)

    print(f"GeoJSON files saved to: {output_folder}")