"""
Cluster detected vineyard poles to remove duplicates from overlapping drone images.
Uses spatial clustering (DBSCAN) to group nearby detections and keeps the highest confidence detection.
"""

import json
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import argparse
from pathlib import Path


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on earth (in meters).
    """
    R = 6371000  # Radius of earth in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def compute_distance_matrix(coordinates):
    """
    Compute pairwise distances between all coordinates using haversine formula.
    
    Args:
        coordinates: Nx2 array of [lon, lat] pairs
    
    Returns:
        NxN distance matrix in meters
    """
    n = len(coordinates)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            lon1, lat1 = coordinates[i]
            lon2, lat2 = coordinates[j]
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


def cluster_poles(geojson_path, pole_spacing=2.0, cluster_eps_factor=0.5, min_samples=1, 
                  output_path=None, verbose=True):
    """
    Cluster pole detections from overlapping drone images.
    
    Args:
        geojson_path: Path to input GeoJSON file
        pole_spacing: Approximate spacing between poles in meters (default: 2.0m)
        cluster_eps_factor: Factor to multiply pole_spacing for clustering threshold (default: 0.5)
                           Lower values = stricter clustering, higher = more aggressive merging
        min_samples: Minimum samples for DBSCAN core points (default: 1)
        output_path: Path for output GeoJSON (default: input_path with '_clustered' suffix)
        verbose: Print progress information
    
    Returns:
        Dictionary with clustering statistics
    """
    # Load GeoJSON
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    features = data['features']
    
    if verbose:
        print(f"Loaded {len(features)} pole detections")
    
    # Extract coordinates and confidence scores
    coordinates = []
    confidences = []
    original_features = []
    
    for feature in features:
        if feature.get('geometry') and feature['geometry'].get('coordinates'):
            coords = feature['geometry']['coordinates']
            conf = feature.get('properties', {}).get('confidence', 0.5)
            
            coordinates.append(coords)
            confidences.append(conf)
            original_features.append(feature)
    
    coordinates = np.array(coordinates)
    confidences = np.array(confidences)
    
    if verbose:
        print(f"Valid detections: {len(coordinates)}")
        print(f"Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
    
    # Compute distance matrix
    if verbose:
        print("Computing distance matrix...")
    
    dist_matrix = compute_distance_matrix(coordinates)
    
    # Perform DBSCAN clustering
    eps = pole_spacing * cluster_eps_factor  # Cluster radius in meters
    
    if verbose:
        print(f"Clustering with eps={eps:.2f}m (pole_spacing={pole_spacing}m × {cluster_eps_factor})")
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(dist_matrix)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    if verbose:
        print(f"Found {n_clusters} clusters")
        print(f"Noise points (isolated detections): {n_noise}")
    
    # Create clustered features - keep highest confidence detection per cluster
    clustered_features = []
    cluster_stats = []
    
    for cluster_id in set(labels):
        if cluster_id == -1:
            # Keep noise points as individual detections
            noise_indices = np.where(labels == -1)[0]
            for idx in noise_indices:
                clustered_features.append(original_features[idx])
            continue
        
        # Get all points in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_confidences = confidences[cluster_indices]
        
        # Find the detection with highest confidence
        best_idx_in_cluster = np.argmax(cluster_confidences)
        best_idx = cluster_indices[best_idx_in_cluster]
        
        # Calculate cluster statistics
        cluster_coords = coordinates[cluster_indices]
        cluster_center = cluster_coords.mean(axis=0)
        max_distance = 0
        for i in range(len(cluster_indices)):
            for j in range(i+1, len(cluster_indices)):
                idx_i, idx_j = cluster_indices[i], cluster_indices[j]
                max_distance = max(max_distance, dist_matrix[idx_i, idx_j])
        
        # Add metadata to the selected feature
        best_feature = original_features[best_idx].copy()
        best_feature['properties']['cluster_id'] = int(cluster_id)
        best_feature['properties']['cluster_size'] = int(len(cluster_indices))
        best_feature['properties']['cluster_max_distance_m'] = float(max_distance)
        best_feature['properties']['cluster_mean_confidence'] = float(cluster_confidences.mean())
        
        clustered_features.append(best_feature)
        
        cluster_stats.append({
            'cluster_id': int(cluster_id),
            'size': len(cluster_indices),
            'max_distance_m': float(max_distance),
            'best_confidence': float(cluster_confidences[best_idx_in_cluster]),
            'mean_confidence': float(cluster_confidences.mean())
        })
    
    # Create output GeoJSON
    output_data = {
        'type': 'FeatureCollection',
        'features': clustered_features
    }
    
    # Determine output path
    if output_path is None:
        input_path = Path(geojson_path)
        output_path = input_path.parent / f"{input_path.stem}_clustered{input_path.suffix}"
    
    # Save clustered GeoJSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    if verbose:
        print(f"\nClustering complete!")
        print(f"Original detections: {len(features)}")
        print(f"Clustered detections: {len(clustered_features)}")
        print(f"Reduction: {len(features) - len(clustered_features)} detections ({(1 - len(clustered_features)/len(features))*100:.1f}%)")
        print(f"\nOutput saved to: {output_path}")
        
        if cluster_stats:
            sizes = [s['size'] for s in cluster_stats if s['cluster_id'] >= 0]
            if sizes:
                print(f"\nCluster statistics:")
                print(f"  Mean cluster size: {np.mean(sizes):.1f}")
                print(f"  Max cluster size: {max(sizes)}")
                print(f"  Clusters with >1 detection: {sum(1 for s in sizes if s > 1)}")
    
    return {
        'original_count': len(features),
        'clustered_count': len(clustered_features),
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'cluster_stats': cluster_stats,
        'output_path': str(output_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Cluster vineyard pole detections to remove duplicates from overlapping images'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Path to input GeoJSON file with pole detections'
    )
    parser.add_argument(
        '--pole-spacing',
        type=float,
        default=2.0,
        help='Approximate spacing between poles in meters (default: 2.0)'
    )
    parser.add_argument(
        '--row-spacing',
        type=float,
        default=None,
        help='Approximate spacing between rows in meters (informational only)'
    )
    parser.add_argument(
        '--eps-factor',
        type=float,
        default=0.5,
        help='Clustering radius = pole_spacing × eps_factor (default: 0.5). Lower=stricter, higher=more aggressive'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=1,
        help='Minimum samples for DBSCAN core points (default: 1)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output GeoJSON path (default: input_clustered.geojson)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    if args.row_spacing:
        print(f"Note: Row spacing ({args.row_spacing}m) is noted but not used in clustering.")
        print("      Clustering is based on pole spacing only.\n")
    
    cluster_poles(
        geojson_path=args.input,
        pole_spacing=args.pole_spacing,
        cluster_eps_factor=args.eps_factor,
        min_samples=args.min_samples,
        output_path=args.output,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
