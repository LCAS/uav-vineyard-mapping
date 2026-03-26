import numpy as np
from sklearn.cluster import DBSCAN
import json
import os

def cluster_poles_dbscan(geojson_data, distance_threshold_meters=1.5, min_samples=2, metric='chebyshev', algorithm='ball_tree'):
    """
    Clusters GPS points using DBSCAN with Haversine metric.
    
    Args:
        geojson_data: The input FeatureCollection dictionary.
        distance_threshold_meters: The maximum distance between two points to be considered the same post.
                                   1.5 meters is a good starting point for non-RTK drones.
        min_samples: Minimum detections required to confirm a post (removes noise).
    """
    features = geojson_data['features']
    if not features:
        return geojson_data

    # 1. Extract coordinates (lon, lat)
    coords = []
    for f in features:
        lon, lat = f['geometry']['coordinates']
        coords.append([lat, lon]) # sklearn requires [lat, lon] for haversine
    
    coords_rad = np.radians(coords)

    # 2. Run DBSCAN
    # Earth radius in meters ~ 6371000
    kms_per_radian = 6371.0088
    epsilon = (distance_threshold_meters / 1000.0) / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=min_samples, metric=metric, algorithm=algorithm).fit(coords_rad)
    
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Found {num_clusters} unique posts from {len(features)} detections.")

    # 3. Merge Clusters into Centroids
    clustered_features = []
    
    # Process unique clusters (label -1 is noise/outliers)
    unique_labels = set(cluster_labels)
    
    for label in unique_labels:
        if label == -1:
            # OPTIONAL: Keep noise points as unconfirmed posts? 
            # Usually better to discard them if min_samples > 1
            continue

        # Get all points belonging to this cluster
        mask = (cluster_labels == label)
        points_in_cluster = np.array(coords)[mask]
        
        # Calculate Centroid (average lat/lon)
        centroid = points_in_cluster.mean(axis=0)
        mean_lat, mean_lon = centroid[0], centroid[1]
        
        # Calculate average confidence if available
        confidences = [features[i]['properties']['confidence'] for i in range(len(features)) if mask[i]]
        avg_conf = float(np.mean(confidences))

        clustered_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [mean_lon, mean_lat] # GeoJSON is [lon, lat]
            },
            "properties": {
                "type": "pole",
                "confidence": avg_conf,
                "detection_count": int(np.sum(mask)) # How many times this post was seen
            }
        })

    return {"type": "FeatureCollection", "features": clustered_features}

if __name__ == "__main__":
    # Example usage
    input_geojson_path = "resnet_inference/vineyard_segmentation_paper_1/full_images_2_filtered/train_resnet101_20260203_135036/inference_results_full/65_feet/poles_raw.geojson"
    output_geojson_path = "resnet_inference/vineyard_segmentation_paper_1/full_images_2_filtered/train_resnet101_20260203_135036/inference_results_full/65_feet/sklearn_dbscan_detected_clustered_pole_coordinates.geojson"
    
    # input_geojson_path = "../../data/riseholme/vineyard_segmentation-22/train5_inference_results/august_2024/39_feet/detected_pole_coordinates.geojson"
    # output_geojson_path = "../../data/riseholme/vineyard_segmentation-22/train5_inference_results/august_2024/39_feet/sklearn_dbscan/sklearn_dbscan_detected_clustered_pole_coordinates.geojson"
    
    # input_geojson_path = "../../data/jojo/vineyard_segmentation-22/train2_inference_results/detected_pole_coordinates.geojson"
    # output_geojson_path = "../../data/jojo/vineyard_segmentation-22/train2_inference_results/sklearn_dbscan/sklearn_dbscan_detected_clustered_pole_coordinates.geojson"

    with open(input_geojson_path, 'r') as f:
        geojson_data = json.load(f)

    clustered_geojson = cluster_poles_dbscan(geojson_data, distance_threshold_meters=0.9, min_samples=2, metric='chebyshev', algorithm='ball_tree')

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_geojson_path), exist_ok=True)

    with open(output_geojson_path, 'w') as f:
        json.dump(clustered_geojson, f, indent=2)

    print(f"Clustered posts saved to {output_geojson_path}")