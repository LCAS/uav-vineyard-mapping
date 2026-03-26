import numpy as np
from sklearn.cluster import DBSCAN
import json
from itertools import product
from scipy.spatial.distance import cdist

def load_geojson(filepath):
    """Load GeoJSON and extract coordinates."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    coords = []
    for feature in data['features']:
        lon, lat = feature['geometry']['coordinates']
        coords.append([lat, lon])
    
    return np.array(coords), data

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in meters."""
    R = 6371000  # Earth radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def cluster_poles_dbscan(coords, distance_threshold_meters=1.5, min_samples=2, metric='haversine', algorithm='ball_tree'):
    """Cluster poles using DBSCAN."""
    if metric == 'haversine':
        coords_rad = np.radians(coords)
        kms_per_radian = 6371.0088
        epsilon = (distance_threshold_meters / 1000.0) / kms_per_radian
        db = DBSCAN(eps=epsilon, min_samples=min_samples, metric=metric, algorithm=algorithm)
        labels = db.fit_predict(coords_rad)
    else:
        # For non-haversine metrics, convert to approximate meters
        # At this latitude, 1 degree ≈ 111km, so scale epsilon accordingly
        epsilon_degrees = distance_threshold_meters / 111000.0
        db = DBSCAN(eps=epsilon_degrees, min_samples=min_samples, metric=metric, algorithm=algorithm)
        labels = db.fit_predict(coords)
    
    return labels

def match_clusters_to_ground_truth(detected_coords, detected_labels, ground_truth_coords, match_distance_m=2.0):
    """
    Match detected clusters to ground truth poles.
    Returns: true_positives, false_positives, false_negatives, precision, recall, f1
    """
    num_clusters = len(set(detected_labels)) - (1 if -1 in detected_labels else 0)
    
    matched_ground_truth = set()
    true_positives = 0
    
    # For each detected cluster, find nearest ground truth pole
    unique_labels = set(detected_labels)
    for label in unique_labels:
        if label == -1:  # Skip noise
            continue
        
        mask = detected_labels == label
        cluster_points = detected_coords[mask]
        cluster_center = cluster_points.mean(axis=0)
        
        # Find closest ground truth pole
        min_distance = float('inf')
        closest_gt_idx = None
        
        for gt_idx, gt_pole in enumerate(ground_truth_coords):
            dist = haversine_distance(
                cluster_center[0], cluster_center[1],
                gt_pole[0], gt_pole[1]
            )
            
            if dist < min_distance:
                min_distance = dist
                closest_gt_idx = gt_idx
        
        # If close enough and not already matched, it's a true positive
        if min_distance <= match_distance_m and closest_gt_idx not in matched_ground_truth:
            true_positives += 1
            matched_ground_truth.add(closest_gt_idx)
    
    false_positives = num_clusters - true_positives
    false_negatives = len(ground_truth_coords) - true_positives
    
    # Calculate metrics
    precision = true_positives / num_clusters if num_clusters > 0 else 0
    recall = true_positives / len(ground_truth_coords) if len(ground_truth_coords) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'num_clusters': num_clusters,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def optimize_dbscan_params(detected_geojson_path, ground_truth_geojson_path, 
                          distance_range=None, min_samples_range=None,
                          metrics=None, algorithms=None):
    """
    Test different DBSCAN parameters and find the best ones.
    """
    # Load data
    detected_coords, _ = load_geojson(detected_geojson_path)
    ground_truth_coords, _ = load_geojson(ground_truth_geojson_path)
    
    print(f"Loaded {len(detected_coords)} detected poles")
    print(f"Loaded {len(ground_truth_coords)} ground truth poles\n")
    
    # Default ranges
    if distance_range is None:
        distance_range = np.arange(0.1, 3.0, 0.01)  # 0.1m to 3.0m in 0.01m steps
    
    if min_samples_range is None:
        min_samples_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    if metrics is None:
        metrics = ['haversine']  # Default to haversine for GPS data
    
    if algorithms is None:
        algorithms = ['ball_tree']  # Default algorithm
    
    results = []
    
    total_combos = len(distance_range) * len(min_samples_range) * len(metrics) * len(algorithms)
    print(f"Testing {len(distance_range)} distances × {len(min_samples_range)} min_samples × {len(metrics)} metrics × {len(algorithms)} algorithms = {total_combos} combinations...\n")
    
    best_score = -1
    best_params = None
    
    for dist, min_samp, metric, algo in product(distance_range, min_samples_range, metrics, algorithms):
        # Skip invalid combinations
        if metric == 'haversine' and algo not in ['ball_tree', 'brute']:
            continue
        
        try:
            # Cluster
            labels = cluster_poles_dbscan(detected_coords, distance_threshold_meters=dist, 
                                         min_samples=min_samp, metric=metric, algorithm=algo)
            
            # Evaluate
            eval_metrics = match_clusters_to_ground_truth(detected_coords, labels, ground_truth_coords)
            eval_metrics['distance_threshold'] = round(dist, 2)
            eval_metrics['min_samples'] = min_samp
            eval_metrics['metric'] = metric
            eval_metrics['algorithm'] = algo
            
            results.append(eval_metrics)
            
            # Track best
            if eval_metrics['f1'] > best_score:
                best_score = eval_metrics['f1']
                best_params = {
                    'distance_threshold_meters': dist,
                    'min_samples': min_samp,
                    'metric': metric,
                    'algorithm': algo,
                    'f1': eval_metrics['f1']
                }
            
            print(f"dist={dist:.2f}m, min_samples={min_samp}, metric={metric}, algo={algo}: "
                  f"F1={eval_metrics['f1']:.3f}, Precision={eval_metrics['precision']:.3f}, "
                  f"Recall={eval_metrics['recall']:.3f}, TP={eval_metrics['true_positives']}, "
                  f"Clusters={eval_metrics['num_clusters']}")
        except Exception as e:
            print(f"Skipping dist={dist:.2f}m, min_samples={min_samp}, metric={metric}, algo={algo}: {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"BEST PARAMETERS:")
    print(f"  distance_threshold_meters: {best_params['distance_threshold_meters']:.2f}")
    print(f"  min_samples: {best_params['min_samples']}")
    print(f"  metric: {best_params['metric']}")
    print(f"  algorithm: {best_params['algorithm']}")
    print(f"  F1 Score: {best_params['f1']:.3f}")
    print(f"{'='*80}\n")
    
    # Sort by F1 score and show top 10
    results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)
    
    print("Top 10 parameter combinations:")
    print(f"{'Dist(m)':<10} {'Min_Samp':<10} {'Metric':<12} {'Algorithm':<12} {'F1':<8} {'Precision':<10} {'Recall':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-" * 100)
    for i, r in enumerate(results_sorted[:10]):
        print(f"{r['distance_threshold']:<10.2f} {r['min_samples']:<10} "
              f"{r['metric']:<12} {r['algorithm']:<12} "
              f"{r['f1']:<8.3f} {r['precision']:<10.3f} {r['recall']:<10.3f} "
              f"{r['true_positives']:<6} {r['false_positives']:<6} {r['false_negatives']:<6}")
    
    return best_params, results_sorted

if __name__ == "__main__":
    # Paths - adjust these to your data
    detected_geojson_path = "../../data/riseholme/vineyard_segmentation-22/train5_inference_results/august_2024/39_feet/detected_pole_coordinates.geojson"
    ground_truth_geojson_path = "../../ground_truth/riseholme/riseholme_pole_locations.geojson"
    
    # Define parameter ranges
    metrics_to_test = ['haversine', 'euclidean', 'manhattan', 'chebyshev']
    algorithms_to_test = ['ball_tree', 'brute', 'kd_tree', 'auto']
    
    # Run optimization
    best_params, all_results = optimize_dbscan_params(
        detected_geojson_path, 
        ground_truth_geojson_path,
        metrics=metrics_to_test,
        algorithms=algorithms_to_test
    )
    
    print("\nYou can now use these optimal parameters in skearn_dbscan.py:")
    print(f"  distance_threshold_meters={best_params['distance_threshold_meters']}")
    print(f"  min_samples={best_params['min_samples']}")
