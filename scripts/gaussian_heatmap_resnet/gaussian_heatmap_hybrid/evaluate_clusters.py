import json
import math
import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

# Optional: HDBSCAN
try:
    import hdbscan
except ImportError:
    hdbscan = None

# ===========================
# 1. CONFIGURATION RANGES
# ===========================
EPS_RANGES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5] 
MIN_SAMPLES_RANGES = [1, 3, 5, 8]
K_VALUES = [38, 40, 42]
MATCH_THRESHOLD_M = 2.5
EARTH_RADIUS = 6371000.0

# ===========================
# 2. HELPER FUNCTIONS
# ===========================
def load_coords(path: str) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    coords = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") == "Point":
            lon, lat = geom.get("coordinates")
            coords.append([lat, lon])
    return np.array(coords)

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * EARTH_RADIUS * math.atan2(math.sqrt(a), math.sqrt(1-a))

def pairwise_haversine_m(coords: np.ndarray) -> np.ndarray:
    rads = np.radians(coords)
    lat = rads[:, 0]
    lon = rads[:, 1]
    dlat = lat[:, None] - lat
    dlon = lon[:, None] - lon
    a = np.sin(dlat/2)**2 + np.cos(lat[:, None]) * np.cos(lat) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return EARTH_RADIUS * c

def get_centroids(coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    unique_labels = set(labels)
    if -1 in unique_labels: unique_labels.remove(-1)
    centroids = []
    for k in unique_labels:
        mask = (labels == k)
        cluster_points = coords[mask]
        mean_lat = np.mean(cluster_points[:, 0])
        mean_lon = np.mean(cluster_points[:, 1])
        centroids.append([mean_lat, mean_lon])
    if not centroids: return np.empty((0, 2))
    return np.array(centroids)

def save_geojson(centroids, filename):
    features = []
    for i, (lat, lon) in enumerate(centroids):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"pole_id": i}
        })
    with open(filename, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)

# ===========================
# 3. EVALUATION LOGIC
# ===========================
def evaluate_run(gt_coords, pred_coords):
    n_gt = len(gt_coords)
    n_pred = len(pred_coords)
    if n_pred == 0: return 0, n_gt, 0, 999.0

    cost_matrix = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            cost_matrix[i, j] = haversine_m(gt_coords[i, 0], gt_coords[i, 1],
                                            pred_coords[j, 0], pred_coords[j, 1])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    true_positives = 0
    errors = []
    for r, c in zip(row_ind, col_ind):
        dist = cost_matrix[r, c]
        if dist <= MATCH_THRESHOLD_M:
            true_positives += 1
            errors.append(dist)
            
    rmse = np.sqrt(np.mean(np.square(errors))) if errors else 0.0
    return true_positives, n_gt - true_positives, n_pred - true_positives, rmse

# ===========================
# 4. MAIN LOOP
# ===========================
def run_benchmark(raw_geojson, gt_geojson):
    print(f"🔹 Loading Raw Detections: {raw_geojson}")
    raw_coords = load_coords(raw_geojson)
    print(f"🔹 Loading Ground Truth:   {gt_geojson}")
    gt_coords = load_coords(gt_geojson)
    
    dist_matrix = pairwise_haversine_m(raw_coords)
    results = []

    # --- Grid Search ---
    print("\n🚀 Running Grid Search...")
    
    # DBSCAN
    for eps in EPS_RANGES:
        for min_s in MIN_SAMPLES_RANGES:
            model = DBSCAN(eps=eps, min_samples=min_s, metric="precomputed")
            labels = model.fit_predict(dist_matrix)
            centroids = get_centroids(raw_coords, labels)
            tp, fn, fp, rmse = evaluate_run(gt_coords, centroids)
            results.append({
                "Method": "DBSCAN", "Params": {"eps": eps, "min_samples": min_s},
                "TP": tp, "FN": fn, "FP": fp, "RMSE": rmse, "Centroids": centroids
            })

    # Agglomerative
    for dist_thresh in EPS_RANGES:
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh, metric="precomputed", linkage="average")
        labels = model.fit_predict(dist_matrix)
        centroids = get_centroids(raw_coords, labels)
        tp, fn, fp, rmse = evaluate_run(gt_coords, centroids)
        results.append({
            "Method": "Agglomerative", "Params": {"distance_threshold": dist_thresh},
            "TP": tp, "FN": fn, "FP": fp, "RMSE": rmse, "Centroids": centroids
        })

    # KMeans
    mean_lat = np.mean(raw_coords[:, 0])
    meters_per_deg = 111320.0
    xy_coords = np.column_stack([
        raw_coords[:, 1] * meters_per_deg * math.cos(math.radians(mean_lat)), 
        raw_coords[:, 0] * meters_per_deg
    ])
    for k in K_VALUES:
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = model.fit_predict(xy_coords)
        centroids = get_centroids(raw_coords, labels)
        tp, fn, fp, rmse = evaluate_run(gt_coords, centroids)
        results.append({
            "Method": "KMeans", "Params": {"n_clusters": k},
            "TP": tp, "FN": fn, "FP": fp, "RMSE": rmse, "Centroids": centroids
        })

    # --- Results ---
    df = pd.DataFrame(results)
    # Sort by TP (desc), then RMSE (asc), then FP (asc)
    df_sorted = df.sort_values(by=["TP", "RMSE", "FP"], ascending=[False, True, True])
    
    print("\n🏆 TOP 3 CONFIGURATIONS")
    print(df_sorted[["Method", "Params", "TP", "RMSE", "FP"]].head(3).to_string(index=False))

    # --- SAVE BEST ---
    best_row = df_sorted.iloc[0]
    best_centroids = best_row["Centroids"]
    best_filename = "best_clustered_poles.geojson"
    
    save_geojson(best_centroids, best_filename)
    
    print(f"\n✅ WINNER: {best_row['Method']} with {best_row['Params']}")
    print(f"💾 Saved {len(best_centroids)} optimized pole locations to: {best_filename}")
   
if __name__ == "__main__":
    # CHANGE THESE PATHS TO YOUR FILES
    RAW_PATH = "inference_results/hybrid_test/resnet101_20260210_163952/august_2024/65_feet/image_size_1280x960/poles.geojson" 
    GT_PATH = "../../../data/riseholme_rtk_gps_poles.geojson"
    
    run_benchmark(RAW_PATH, GT_PATH)