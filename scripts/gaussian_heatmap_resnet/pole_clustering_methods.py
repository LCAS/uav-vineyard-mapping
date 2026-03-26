import argparse
import json
import math
import os
import sys
from typing import List, Tuple

import numpy as np

try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "scikit-learn is required for this script. Install with: pip install scikit-learn"
    ) from exc

# Optional: HDBSCAN if installed
try:  # pragma: no cover - optional dependency
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None


def load_geojson_points(path: str) -> Tuple[np.ndarray, List[dict]]:
    with open(path, "r") as f:
        data = json.load(f)

    coords = []
    props = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") != "Point":
            continue
        lon, lat = geom.get("coordinates", [None, None])
        if lat is None or lon is None:
            continue
        coords.append([lat, lon])
        props.append(feat.get("properties", {}))

    if not coords:
        return np.empty((0, 2), dtype=float), []

    return np.array(coords, dtype=float), props


def haversine_distance_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def pairwise_haversine_m(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    dists = np.zeros((n, n), dtype=float)
    for i in range(n):
        lat1, lon1 = coords[i]
        for j in range(i + 1, n):
            lat2, lon2 = coords[j]
            d = haversine_distance_m(lat1, lon1, lat2, lon2)
            dists[i, j] = d
            dists[j, i] = d
    return dists


def cluster_dbscan(coords: np.ndarray, eps_m: float, min_samples: int) -> np.ndarray:
    dists = pairwise_haversine_m(coords)
    model = DBSCAN(eps=eps_m, min_samples=min_samples, metric="precomputed")
    return model.fit_predict(dists)


def cluster_agglomerative(coords: np.ndarray, distance_threshold_m: float) -> np.ndarray:
    dists = pairwise_haversine_m(coords)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold_m,
        metric="precomputed",
        linkage="average",
    )
    return model.fit_predict(dists)


def cluster_kmeans(coords: np.ndarray, k: int, reference_lat: float | None = None) -> np.ndarray:
    # KMeans assumes Euclidean; use accurate meters conversion from actual GPS data.
    # k is the number of clusters to find (adjust based on expected pole count or use heuristics).
    if reference_lat is None:
        reference_lat = float(np.mean(coords[:, 0]))
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(reference_lat))
    xy = np.column_stack([
        coords[:, 1] * meters_per_deg_lon,
        coords[:, 0] * meters_per_deg_lat,
    ])
    model = KMeans(n_clusters=k, n_init="auto", random_state=0)
    return model.fit_predict(xy)


def cluster_hdbscan(coords: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    if hdbscan is None:
        raise SystemExit("hdbscan is not installed. Install with: pip install hdbscan")
    dists = pairwise_haversine_m(coords)
    model = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    return model.fit_predict(dists)


def cluster_centroids(coords: np.ndarray, labels: np.ndarray) -> List[Tuple[float, float, int]]:
    centroids = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            continue
        subset = coords[idx]
        lat = float(np.mean(subset[:, 0]))
        lon = float(np.mean(subset[:, 1]))
        centroids.append((lat, lon, int(label)))
    return centroids


def save_clustered_geojson(
    coords: np.ndarray,
    props: List[dict],
    labels: np.ndarray,
    output_path: str,
) -> None:
    features = []
    for (lat, lon), prop, label in zip(coords, props, labels):
        out_props = dict(prop)
        out_props["cluster_id"] = int(label)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": out_props,
            }
        )

    out = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)


def save_centroids_geojson(
    centroids: List[Tuple[float, float, int]], output_path: str
) -> None:
    features = []
    for lat, lon, label in centroids:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"cluster_id": label},
            }
        )
    out = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Try multiple clustering methods for pole detections in GeoJSON."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to poles GeoJSON (e.g., poles_raw.geojson)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write clustered outputs",
    )
    parser.add_argument(
        "--method",
        choices=["dbscan", "agglomerative", "kmeans", "hdbscan"],
        default="dbscan",
        help="Clustering method to run",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all methods and write outputs for each",
    )
    parser.add_argument("--eps-m", type=float, default=1.5, help="DBSCAN radius in meters")
    parser.add_argument(
        "--min-samples", type=int, default=2, help="DBSCAN/HDBSCAN min samples"
    )
    parser.add_argument(
        "--distance-threshold-m",
        type=float,
        default=1.5,
        help="Agglomerative distance threshold in meters",
    )
    parser.add_argument("--k", type=int, default=50, help="KMeans cluster count")
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="HDBSCAN min cluster size",
    )
    parser.add_argument(
        "--reference-lat",
        type=float,
        default=None,
        help="Reference latitude for KMeans meters conversion (default: mean of pole coords)",
    )

    if argv is None:
        if len(sys.argv) == 1:
            raise SystemExit(
                "No CLI args provided. Edit the defaults in __main__ or pass args."
            )
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    coords, props = load_geojson_points(args.input)

    if coords.shape[0] == 0:
        raise SystemExit("No points found in input GeoJSON.")

    def run_method(method: str) -> None:
        if method == "dbscan":
            labels = cluster_dbscan(coords, args.eps_m, args.min_samples)
        elif method == "agglomerative":
            labels = cluster_agglomerative(coords, args.distance_threshold_m)
        elif method == "kmeans":
            ref_lat = args.reference_lat if args.reference_lat is not None else float(np.mean(coords[:, 0]))
            meters_per_deg_lat = 111320.0
            meters_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))
            print(f"  KMeans using reference latitude: {ref_lat:.6f} (from GeoJSON pole data)")
            print(f"    Meters per degree: lat={meters_per_deg_lat:.2f}m, lon={meters_per_deg_lon:.2f}m")
            labels = cluster_kmeans(coords, args.k, reference_lat=args.reference_lat)
        else:
            labels = cluster_hdbscan(coords, args.min_cluster_size, args.min_samples)

        base = os.path.splitext(os.path.basename(args.input))[0]
        clustered_path = os.path.join(
            args.output_dir, f"{base}_{method}_clustered.geojson"
        )
        centroids_path = os.path.join(
            args.output_dir, f"{base}_{method}_centroids.geojson"
        )

        save_clustered_geojson(coords, props, labels, clustered_path)
        centroids = cluster_centroids(coords, labels)
        save_centroids_geojson(centroids, centroids_path)

        num_clusters = len({label for label in labels if label != -1})
        num_noise = int(np.sum(labels == -1))

        print("Finished clustering.")
        print(f"Method: {method}")
        print(f"Clusters: {num_clusters}")
        print(f"Noise points: {num_noise}")
        print(f"Clustered points: {clustered_path}")
        print(f"Centroids: {centroids_path}")

    if args.run_all:
        methods = ["dbscan", "agglomerative", "kmeans", "hdbscan"]
        for method in methods:
            run_method(method)
    else:
        run_method(args.method)


if __name__ == "__main__":
    # Example CLI args (edit as needed):
    #   --input /path/to/poles_raw.geojson \
    #   --output-dir /path/to/output \
    #   --method dbscan \
    #   --eps-m 1.5 \
    #   --min-samples 2
    default_args = [
        "--input",
        "resnet_inference/vineyard_segmentation_paper_1/full_images_2_filtered/train_resnet101_20260203_135036/inference_results_full/65_feet_run_2/poles_raw.geojson",
        "--output-dir",
        "resnet_inference/vineyard_segmentation_paper_1/full_images_2_filtered/train_resnet101_20260203_135036/inference_results_full/65_feet_run_2/clustered_poles",
        "--method",
        "dbscan",
        "--eps-m", # Adjust this based on expected pole spacing and GPS noise. Too small may split poles; too large may merge nearby poles.
        "1.75", # DBSCAN radius in meters (adjust based on expected pole spacing and GPS noise)
        "--min-samples", # DBSCAN and HDBSCAN
        "5", # DBSCAN and HDBSCAN
        "--distance-threshold-m", # Agglomerative
        "1.75", # Agglomerative distance threshold in meters (similar to DBSCAN radius)
        "--k", # KMeans the number of clusters to find (adjust based on expected pole count or use heuristics)
        "10", # KMeans
        "--min-cluster-size", # HDBSCAN
        "5", # HDBSCAN
        "--run-all", # Run all methods with the above parameters (comment out to run just one method)
    ]
    main(default_args if len(sys.argv) == 1 else None)
