'''
This script processes GeoJSON data to cluster pole coordinates using the DBSCAN algorithm. The steps include:

1. Load Input Data: Reads a GeoJSON feature collection containing pole coordinates as Point features.
2. Extract and Prepare Data: Extracts coordinates and converts them into a NumPy array for clustering.
3. DBSCAN Clustering: Applies the DBSCAN algorithm with haversine distance, identifying clusters of poles based on spatial proximity.
4. Calculate Centroids: For each cluster, calculates the centroid using the Shapely library.
5. Generate Clustered GeoJSON: Creates a new GeoJSON feature collection containing the centroids of identified clusters as Point features, with properties like cluster_id.
6. Return Output: Returns the resulting clustered GeoJSON feature collection.
'''

import json
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, MultiPoint

def cluster_poles(geojson_data, eps=0.0000002, min_samples=3):
    """
    Clusters pole coordinates from a GeoJSON feature collection using DBSCAN and returns the clustered centroids as GeoJSON.
    
    :param geojson_data: A GeoJSON feature collection containing pole coordinates.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: A GeoJSON feature collection containing the clustered centroids.
    """
    # Extract coordinates from the GeoJSON data
    coordinates = [
        feature["geometry"]["coordinates"]
        for feature in geojson_data["features"]
        if feature["geometry"]["type"] == "Point"
    ]
    
    if not coordinates:
        print("No valid point features found in the input GeoJSON.")
        return {"type": "FeatureCollection", "features": []}
    
    # Convert to NumPy array
    coords = np.array(coordinates)
    
    # Run DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="haversine")
    coords_radians = np.radians(coords)
    labels = dbscan.fit_predict(coords_radians)
    
    # Create a new GeoJSON FeatureCollection for the clustered centroids
    clustered_geojson = {"type": "FeatureCollection", "features": []}
    
    # Process each cluster
    unique_labels = set(labels)
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue  # Skip noise points

        cluster_points = coords[labels == cluster_id]
        cluster_geometry = MultiPoint([Point(p[0], p[1]) for p in cluster_points])
        cluster_centroid = cluster_geometry.centroid

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [cluster_centroid.x, cluster_centroid.y]
            },
            "properties": {
                "type": "pole",
                "pole_id": int(cluster_id)
            }
        }
        clustered_geojson["features"].append(feature)
    
    return clustered_geojson

if __name__ == "__main__":
    input_file = "../data/detected_pole_coordinates.geojson"
    output_file = "../data/clustered_poles.geojson"
    
    # Load input GeoJSON
    with open(input_file, "r") as file:
        input_geojson = json.load(file)
    
    # Process clustering
    clustered_geojson = cluster_poles(input_geojson)
    
    # Save the clustered GeoJSON to a file
    with open(output_file, "w") as file:
        json.dump(clustered_geojson, file, indent=4)
    
    print(f"Clustered GeoJSON saved to: {output_file}")
