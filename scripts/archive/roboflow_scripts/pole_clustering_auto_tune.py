import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from shapely.geometry import Point, MultiPoint
import matplotlib.pyplot as plt
from collections import Counter
import logging
import os
from datetime import datetime

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

def find_optimal_eps(coords, min_samples_range=[2, 3, 4], eps_range=np.linspace(0.00000002, 0.000002, 100)): # np.linspace(high, low, number of values between high and low)
    """
    Find optimal eps parameter using silhouette score
    """
    best_score = -1
    best_params = (0, 0)
    results = []
    
    coords_radians = np.radians(coords)
    
    for min_samples in min_samples_range:
        for eps in eps_range:
            print(f"Trying eps={eps:.10f}, min_samples={min_samples}")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
            labels = dbscan.fit_predict(coords_radians)
            
            # Skip configurations where all points are noise or all points are in one cluster
            if len(set(labels)) <= 1 or -1 not in set(labels):
                continue
                
            # Calculate score only on non-noise points
            non_noise_indices = labels != -1
            if sum(non_noise_indices) < 2:
                continue
                
            try:
                score = silhouette_score(coords_radians[non_noise_indices], labels[non_noise_indices])
                results.append((eps, min_samples, score, Counter(labels)[-1], len(set(labels)) - 1))
                
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)
            except:
                continue
    
    # if results:
    #     # Plot results
    #     plt.figure(figsize=(15, 10))
        
    #     # Plot 1: Silhouette Score
    #     plt.subplot(2, 2, 1)
    #     for min_samples in min_samples_range:
    #         filtered_results = [(eps, s) for eps, ms, s, _, _ in results if ms == min_samples]
    #         if filtered_results:
    #             eps_values, scores = zip(*filtered_results)
    #             plt.plot(eps_values, scores, 'o-', label=f'min_samples={min_samples}')
    #     plt.xlabel('Epsilon')
    #     plt.ylabel('Silhouette Score')
    #     plt.title('Silhouette Score vs Epsilon')
    #     plt.legend()
        
    #     # Plot 2: Number of Noise Points
    #     plt.subplot(2, 2, 2)
    #     for min_samples in min_samples_range:
    #         filtered_results = [(eps, n) for eps, ms, _, n, _ in results if ms == min_samples]
    #         if filtered_results:
    #             eps_values, noise_counts = zip(*filtered_results)
    #             plt.plot(eps_values, noise_counts, 'o-', label=f'min_samples={min_samples}')
    #     plt.xlabel('Epsilon')
    #     plt.ylabel('Number of Noise Points')
    #     plt.title('Noise Points vs Epsilon')
    #     plt.legend()
        
    #     # Plot 3: Number of Clusters
    #     plt.subplot(2, 2, 3)
    #     for min_samples in min_samples_range:
    #         filtered_results = [(eps, c) for eps, ms, _, _, c in results if ms == min_samples]
    #         if filtered_results:
    #             eps_values, cluster_counts = zip(*filtered_results)
    #             plt.plot(eps_values, cluster_counts, 'o-', label=f'min_samples={min_samples}')
    #     plt.xlabel('Epsilon')
    #     plt.ylabel('Number of Clusters')
    #     plt.title('Clusters vs Epsilon')
    #     plt.legend()
        
    #     plt.tight_layout()
    #     plt.savefig("clustering_parameter_analysis.png")
    #     # logger.info(f"Parameter analysis plot saved to clustering_parameter_analysis.png")
        
        return best_params
    else:
        return (0.0000002, 3)  # Default values if no good parameters found

def cluster_poles(geojson_data, eps=None, min_samples=None, metric="haversine", auto_tune=True):
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
    
    # Auto-tune parameters if requested
    if auto_tune and eps is None:
        # logger.info("Auto-tuning clustering parameters...")
        eps, min_samples = find_optimal_eps(coords)
        # logger.info(f"Selected parameters: eps={eps}, min_samples={min_samples}")
    else:
        eps = eps or 0.0000002
        min_samples = min_samples or 3

    print(f"Clustering with eps={eps:.10f}, min_samples={min_samples}, metric={metric}")
    
    # Run DBSCAN clustering
    coords_radians = np.radians(coords)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(coords_radians)
    
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

if __name__ == "__main__":
    # Create timestamp folder for outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"clustering_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    input_file = "../../data/detected_pole_coordinates.geojson"
    output_file = f"{output_dir}/clustered_poles.geojson"
    
    # Load input GeoJSON
    with open(input_file, "r") as file:
        input_geojson = json.load(file)
    
    # Process clustering with auto-tuning
    clustered_geojson = cluster_poles(
        input_geojson, 
        eps=None, # 0.0000002, # None, # Will be auto-tuned
        min_samples=2, # None,  # Will be auto-tuned
        metric="chebyshev", # chebyshev # haversine
        auto_tune=True
    )
    
    # Save the clustered GeoJSON to a file
    with open(output_file, "w") as file:
        json.dump(clustered_geojson, file, indent=4)
    
    # Visualize the clustering results
    visualize_clusters(clustered_geojson, f"{output_dir}/cluster_visualization.png")
    
    # logger.info(f"Clustered GeoJSON saved to: {output_file}")
    # logger.info(f"All results saved to directory: {output_dir}")