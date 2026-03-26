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
import pyproj

def cluster_poles(geojson_data, eps=2, min_samples=3, metric="chebyshev", target_crs="EPSG:27700"): # eps now in meters
    """
    Clusters pole coordinates from a GeoJSON feature collection using DBSCAN and returns the clustered centroids as GeoJSON.
    """
    coordinates = [
        feature["geometry"]["coordinates"]
        for feature in geojson_data["features"]
        if feature["geometry"]["type"] == "Point"
    ]

    if not coordinates:
        print("No valid point features found in the input GeoJSON.")
        return {"type": "FeatureCollection", "features": []}

    coords = np.array(coordinates)

    # Convert lat/lon to projected coordinates (meters)
    source_crs = "EPSG:4326"  # WGS 84 (latitude/longitude)
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    projected_coords = np.array(list(transformer.transform(coords[:, 0], coords[:, 1]))).T

    # Run DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric) #chebyshev or euclidean is much faster than haversine.
    labels = dbscan.fit_predict(projected_coords)

    # Create a new GeoJSON FeatureCollection for the clustered centroids
    clustered_geojson = {"type": "FeatureCollection", "features": []}

    # Process each cluster
    unique_labels = set(labels)
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue  # Skip noise points

        cluster_points = projected_coords[labels == cluster_id]
        cluster_geometry = MultiPoint([Point(p[0], p[1]) for p in cluster_points])
        cluster_centroid = cluster_geometry.centroid

        # Convert the centroid back to lat/lon
        lon, lat = transformer.transform(cluster_centroid.x, cluster_centroid.y, direction=pyproj.enums.TransformDirection.INVERSE)

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "type": "pole",
                "pole_id": int(cluster_id)
            }
        }
        clustered_geojson["features"].append(feature)

    return clustered_geojson

if __name__ == "__main__":
    # input_file = "../../data/detected_pole_coordinates.geojson"
    # output_file = "../../data/clustered_poles.geojson"

    input_file = "../../data/jojo/detected_pole_coordinates.geojson"
    output_file = "../../data/jojo/clustered_poles.geojson"
    
    # Load input GeoJSON
    with open(input_file, "r") as file:
        input_geojson = json.load(file)
    
    # Process clustering
    clustered_geojson = cluster_poles(input_geojson, eps=0.8, min_samples=2, metric="chebyshev") #  metric="chebyshev"
    
    # Save the clustered GeoJSON to a file
    with open(output_file, "w") as file:
        json.dump(clustered_geojson, file, indent=4)
    
    print(f"Clustered GeoJSON saved to: {output_file}")
