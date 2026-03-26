import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from shapely.geometry import Point, MultiPoint
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from collections import Counter
import logging
import os
from datetime import datetime
import pyproj

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("clustering.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger("pole_clustering")

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def find_optimal_eps(coords, min_samples_range=[2, 3, 4, 5, 6, 7], eps_range=np.linspace(0.1, 2.0, 20), target_crs="EPSG:27700", expected_poles=40):
    """Find optimal eps parameter using silhouette and Davies-Bouldin scores for projected coordinates, factoring in expected poles."""
    best_combined_score = float('-inf')  # Initialize to negative infinity
    best_params = (0, 0)
    results = []

    source_crs = "EPSG:4326"
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    projected_coords = np.array(list(transformer.transform(coords[:, 0], coords[:, 1]))).T

    for min_samples in min_samples_range:
        for eps in eps_range:
            print(f"Trying eps={eps:.2f}, min_samples={min_samples}")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='chebyshev')
            labels = dbscan.fit_predict(projected_coords)

            if len(set(labels)) <= 1 or -1 not in set(labels):
                continue

            non_noise_indices = labels != -1
            if sum(non_noise_indices) < 2:
                continue

            try:
                silhouette = silhouette_score(projected_coords[non_noise_indices], labels[non_noise_indices])
                davies_bouldin = davies_bouldin_score(projected_coords[non_noise_indices], labels[non_noise_indices])
                num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                cluster_diff = abs(num_clusters - expected_poles)
                penalty = cluster_diff * 0.1  # Adjust penalty factor as needed
                combined_score = silhouette - penalty  # Or some other combination of scores

                results.append((eps, min_samples, silhouette, davies_bouldin, num_clusters, Counter(labels)[-1], len(set(labels)) - 1))

                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_params = (eps, min_samples)

            except:
                continue

    if results:
        return best_params
    else:
        return (2, 2)  # Default values in meters.

def cluster_poles(geojson_data, eps=None, min_samples=None, metric="haversine", auto_tune=True, min_samples_range=[2, 3, 4, 5, 6, 7], eps_range=np.linspace(0.1, 2.0, 20),target_crs="EPSG:27700", expected_poles=40):
    """
    Clusters pole coordinates from a GeoJSON feature collection using DBSCAN and returns the clustered centroids as GeoJSON.
    
    :param geojson_data: A GeoJSON feature collection containing pole coordinates.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :param metric: Distance metric to use for clustering.
    :param auto_tune: Whether to automatically find optimal parameters.
    :return: A GeoJSON feature collection containing the clustered centroids.
    """
    # Extract coordinates and feature properties from the GeoJSON data
    coordinates = []
    properties_list = []
    
    for feature in geojson_data["features"]:
        if feature["geometry"]["type"] == "Point":
            coordinates.append(feature["geometry"]["coordinates"])
            properties_list.append(feature.get("properties", {}))
    
    if not coordinates:
        # logger.warning("No valid point features found in the input GeoJSON.")
        return {"type": "FeatureCollection", "features": []}
    
    # Convert to NumPy array
    coords = np.array(coordinates)
    
    if auto_tune and eps is None:
        eps, min_samples = find_optimal_eps(coords, min_samples_range=min_samples_range, eps_range=eps_range, target_crs=target_crs, expected_poles=expected_poles) # Auto-tune in meters
    else:
        eps = eps or 2 # Default in meters
        min_samples = min_samples or 2
    
    print(f"Clustering with eps={eps:.2f}, min_samples={min_samples}, metric={metric}")
    
    source_crs = "EPSG:4326"
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    projected_coords = np.array(list(transformer.transform(coords[:, 0], coords[:, 1]))).T
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric) # Use chebyshev
    labels = dbscan.fit_predict(projected_coords)
    
    # Count clusters and noise points
    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    num_noise = list(labels).count(-1)
    
    # logger.info(f"Clustering results: {num_clusters} clusters, {num_noise} noise points out of {len(coords)} total points")
    
    # Create a new GeoJSON FeatureCollection for the clustered centroids
    clustered_geojson = {
        "type": "FeatureCollection", 
        "features": [],
        "metadata": {
            "clustering_info": {
                "algorithm": "DBSCAN",
                "eps": eps,
                "min_samples": min_samples,
                "metric": metric,
                "num_clusters": num_clusters,
                "num_noise_points": num_noise,
                "total_points": len(coords)
            }
        }
    }
    
    # Process each cluster
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_points = coords[mask]
        cluster_props = [props for i, props in enumerate(properties_list) if mask[i]]
        
        if cluster_id == -1:
            # Add noise points individually
            for i, point in enumerate(cluster_points):
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": point.tolist()
                    },
                    "properties": {
                        **cluster_props[i],
                        "cluster_type": "noise",
                        "pole_id": f"noise_{i}"
                    }
                }
                # clustered_geojson["features"].append(feature) # add noise points to geojson
        else:
            # Calculate cluster statistics
            cluster_geometry = MultiPoint([Point(p[0], p[1]) for p in cluster_points])
            cluster_centroid = cluster_geometry.centroid
            
            # Calculate average distance from points to centroid
            avg_distance = 0
            if len(cluster_points) > 1:
                distances = [haversine_distance(
                    cluster_centroid.x, cluster_centroid.y, 
                    point[0], point[1]
                ) for point in cluster_points]
                avg_distance = sum(distances) / len(distances)
            
            # Merge properties from all points in the cluster
            merged_props = {}
            for prop in cluster_props:
                for key, val in prop.items():
                    if key not in merged_props:
                        merged_props[key] = val
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [cluster_centroid.x, cluster_centroid.y]
                },
                "properties": {
                    **merged_props,
                    "type": "pole",
                    "cluster_id": int(cluster_id),
                    "point_count": int(np.sum(mask)),
                    "avg_distance_km": float(avg_distance),
                    "max_distance_km": float(max(distances) if len(cluster_points) > 1 else 0)
                }
            }
            clustered_geojson["features"].append(feature)
    
    return clustered_geojson

def visualize_clusters(geojson_data, output_file="cluster_visualization.png"):
    """
    Creates a visualization of the clusters
    """
    features = geojson_data["features"]
    
    # Extract coordinates and cluster IDs
    coords = []
    cluster_ids = []
    
    for feature in features:
        coords.append(feature["geometry"]["coordinates"])
        props = feature["properties"]
        cluster_id = props.get("cluster_id", props.get("pole_id", -1))
        cluster_ids.append(cluster_id)
    
    # Convert to NumPy arrays
    coords = np.array(coords)
    
    # Create plot
    plt.figure(figsize=(10, 10))
    
    # Plot noise points
    noise_points = coords[[str(cid).startswith("noise_") for cid in cluster_ids]]
    if len(noise_points) > 0:
        plt.scatter(noise_points[:, 0], noise_points[:, 1], c='black', marker='x', s=30, label='Noise')
    
    # Plot clusters with different colors
    cluster_ids_clean = [cid for cid in cluster_ids if not str(cid).startswith("noise_")]
    cluster_points = coords[[not str(cid).startswith("noise_") for cid in cluster_ids]]
    
    if len(cluster_points) > 0:
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=cluster_ids_clean, cmap='viridis', 
                   s=50, alpha=0.7)
    
    plt.title('Pole Clusters')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    # logger.info(f"Cluster visualization saved to {output_file}")

def remove_extra_clusters(clustered_geojson, distance_threshold=2.0, target_crs="EPSG:27700"):
    """Removes extra clusters by merging or removing close clusters, using projected coordinates."""
    features = clustered_geojson["features"]
    centroids_latlon = []  # Store centroids in lat/lon
    for feature in features:
        if feature["geometry"]["type"] == "Point":
            centroids_latlon.append(feature["geometry"]["coordinates"])
    centroids_latlon = np.array(centroids_latlon)

    if len(centroids_latlon) <= 1:
        return clustered_geojson  # Nothing to merge if there's only one or zero clusters

    source_crs = "EPSG:4326"
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    projected_centroids = np.array(list(transformer.transform(centroids_latlon[:, 0], centroids_latlon[:, 1]))).T

    distances = cdist(projected_centroids, projected_centroids)  # Calculate distances in projected coordinates

    to_remove = []
    to_merge = {}

    for i in range(len(projected_centroids)):
        for j in range(i + 1, len(projected_centroids)):
            if distances[i, j] < distance_threshold:
                if i not in to_remove and j not in to_remove:
                    to_merge.setdefault(i, []).append(j)
                    to_remove.append(j)

    # Merge clusters
    new_features = []
    merged_indices = set()

    for i, related_indices in to_merge.items():
        if i in merged_indices:
            continue

        all_points = [projected_centroids[i]]
        for index in related_indices:
            all_points.append(projected_centroids[index])
            merged_indices.add(index)

        merged_centroid_projected = MultiPoint([Point(p[0], p[1]) for p in all_points]).centroid
        inverse_transformer = pyproj.Transformer.from_crs(target_crs, source_crs, always_xy=True)
        merged_centroid_latlon = inverse_transformer.transform(merged_centroid_projected.x, merged_centroid_projected.y)

        new_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": list(merged_centroid_latlon)
            },
            "properties": features[i]["properties"]
        }
        new_features.append(new_feature)

    for i, feature in enumerate(features):
        if i not in to_remove and i not in to_merge:
            new_features.append(feature)

    clustered_geojson["features"] = new_features
    return clustered_geojson

if __name__ == "__main__":
    # Create timestamp folder for outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"clustering_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # input_file = "../../data/riseholme/detected_pole_coordinates.geojson"
    # input_file = "../../data/detected_pole_coordinates.geojson"
    # output_file = f"{output_dir}/clustered_poles.geojson"

    input_file = "../../data/jojo/detected_pole_coordinates.geojson"
    output_file = "../../data/jojo/clustered_poles.geojson"
    
    # Load input GeoJSON
    with open(input_file, "r") as file:
        input_geojson = json.load(file)
    
    # Process clustering with auto-tuning
    clustered_geojson = cluster_poles(
        input_geojson, 
        eps=None, # 0.8, # None, # Will be auto-tuned
        min_samples=None, # 2, # None,  # Will be auto-tuned
        metric="chebyshev", # chebyshev # euclidean
        auto_tune=True,
        min_samples_range=[2, 3, 4, 5, 6, 7, 8, 9, 10], 
        eps_range=np.linspace(0.1, 2.0, 40),
        target_crs="EPSG:27700",
        expected_poles=40 # 10556 Riseholme poles # 10556 JoJo poles
    )
    
    # Remove extra clusters
    clustered_geojson = remove_extra_clusters(clustered_geojson, distance_threshold=1.25, target_crs="EPSG:27700") # Identifies clusters that are closer than the distance_threshold

    # Save the clustered GeoJSON to a file
    with open(output_file, "w") as file:
        json.dump(clustered_geojson, file, indent=4)
    
    # Visualize the clustering results
    visualize_clusters(clustered_geojson, f"{output_dir}/cluster_visualization.png")
    
    # logger.info(f"Clustered GeoJSON saved to: {output_file}")
    # logger.info(f"All results saved to directory: {output_dir}")