import json
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
from shapely.strtree import STRtree
import shapely.geometry

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
    """Calculates the IoU of two shapely polygons with validation."""
    if not polygon1.is_valid or not polygon2.is_valid:
        print("Warning: Invalid polygon encountered.")
        return 0.0

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
    polygon = shapely.geometry.Polygon(polygon_coordinates)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon

def non_maximum_suppression_polygons(detections, iou_threshold):
    """Performs NMS for polygon masks."""
    sorted_detections = sorted(detections, key=lambda x: x["properties"]["confidence"], reverse=True)
    keep = []
    while sorted_detections:
        best_detection = sorted_detections.pop(0)
        keep.append(best_detection)
        best_polygon = validate_polygon(best_detection["geometry"]["coordinates"][0])
        sorted_detections_temp = []
        for d in sorted_detections:
          current_polygon = validate_polygon(d["geometry"]["coordinates"][0])
          if polygon_iou(best_polygon, current_polygon) < iou_threshold:
            sorted_detections_temp.append(d)
        sorted_detections = sorted_detections_temp
    return keep

def load_geojson(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    return geojson_data.get('features', [])

# Example usage
file_path = "../../data/detected_vine_row_coordinates.geojson"
all_vine_row_coordinates = load_geojson(file_path)

iou_threshold = 0.1
nms_results = non_maximum_suppression_polygons(all_vine_row_coordinates, iou_threshold)

# Convert GeoJSON coordinates to shapely polygons
all_polygons = []
for feature in nms_results: # nms_results or all_vine_row_coordinates
    if feature["geometry"]["type"] == "Polygon":
        coordinates = feature["geometry"]["coordinates"][0]
        if len(coordinates) >= 4:  # Check for at least 4 coordinates
            polygon = Polygon(feature["geometry"]["coordinates"][0])

            # Fix invalid polygons
            if not polygon.is_valid or polygon.is_empty:
                polygon = polygon.buffer(0)

            if polygon.is_valid and not polygon.is_empty:
                all_polygons.append(polygon)
    elif feature["geometry"]["type"] == "MultiPolygon":
        for poly_coords in feature["geometry"]["coordinates"]:
            polygon = Polygon(poly_coords[0])
            if not polygon.is_valid or polygon.is_empty:
                polygon = polygon.buffer(0)
            if polygon.is_valid and not polygon.is_empty:
                all_polygons.append(polygon)

# # **Step 1: Filter out small polygons**
# if all_polygons:
#     average_area = sum(p.area for p in all_polygons) / len(all_polygons)
#     min_area_threshold = 0.5 * average_area  # Half the average area
#     filtered_polygons = [p for p in all_polygons if p.area >= min_area_threshold]
# else:
#     filtered_polygons = []

filtered_polygons = all_polygons

# Step 1.2: Erode polygons
erosion_distance = 0.000135
eroded_polygons = []
for poly in filtered_polygons:
    eroded_poly = erode_polygon(poly, erosion_distance)
    if eroded_poly.is_valid and not eroded_poly.is_empty:
        eroded_polygons.append(eroded_poly)

# Step 2: Cluster overlapping polygons together
merged_vine_rows = []
if filtered_polygons:
    str_tree = STRtree(filtered_polygons)
    visited = set()

    for poly in filtered_polygons:
        if poly in visited:
            continue

        merged_result = merge_cluster(poly, filtered_polygons, visited)
        merged_vine_rows.extend(merged_result)

# Step 3: Convert merged polygons back into GeoJSON format
merged_geojson_list = []
for merged_polygon in merged_vine_rows:
    if isinstance(merged_polygon, Polygon) and not merged_polygon.is_empty:
        merged_geojson_list.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(merged_polygon.exterior.coords)]
            },
            "properties": {"type": "merged_vine_row"}
        })

merged_geojson = {"type": "FeatureCollection", "features": merged_geojson_list}

# for vine_row in all_vine_row_coordinates:     
#     geojson_data_vine_rows["features"].append(vine_row)

# Save GeoJSON data

with open("../../data/detected_merged_vine_rows.geojson", "w") as json_file:
    json.dump(merged_geojson, json_file, indent=4)

print("GeoJSON files saved to: ../../data/detected_merged_vine_rows.geojson")