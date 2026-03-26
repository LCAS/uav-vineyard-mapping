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


def load_geojson_lines(path: str) -> List[np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)
    rows = []
    polygons = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        typ = geom.get("type")
        coords = geom.get("coordinates")
        if coords is None:
            continue
        if typ == "LineString":
            rows.append(np.array([[c[1], c[0]] for c in coords], dtype=float))
        elif typ == "MultiLineString":
            for line in coords:
                rows.append(np.array([[c[1], c[0]] for c in line], dtype=float))
        elif typ == "Polygon":
            # Polygon: [ [ring1], [ring2], ... ] (rings are lists of [lon, lat])
            rings = [np.array([[c[1], c[0]] for c in ring], dtype=float) for ring in coords]
            polygons.append(rings)
        elif typ == "MultiPolygon":
            # MultiPolygon: [ [ [ring1], [ring2] ], [ [ring1], ... ] ]
            for poly in coords:
                rings = [np.array([[c[1], c[0]] for c in ring], dtype=float) for ring in poly]
                polygons.append(rings)
    return rows, polygons


def _latlon_to_xy(lat: float, lon: float, ref_lat: float) -> Tuple[float, float]:
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))
    x = lon * meters_per_deg_lon
    y = lat * meters_per_deg_lat
    return x, y


def point_segment_distance_m(point: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float], ref_lat: float) -> float:
    # Convert to local meters using reference latitude then compute 2D point-segment distance
    px, py = _latlon_to_xy(point[0], point[1], ref_lat)
    ax, ay = _latlon_to_xy(a[0], a[1], ref_lat)
    bx, by = _latlon_to_xy(b[0], b[1], ref_lat)

    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay

    vv = vx * vx + vy * vy
    if vv == 0.0:
        # a and b are the same point
        dx = px - ax
        dy = py - ay
        return math.hypot(dx, dy)

    t = (wx * vx + wy * vy) / vv
    t = max(0.0, min(1.0, t))
    projx = ax + t * vx
    projy = ay + t * vy
    dx = px - projx
    dy = py - projy
    return math.hypot(dx, dy)


def point_in_ring(point: Tuple[float, float], ring: np.ndarray) -> bool:
    # ray casting algorithm for point-in-polygon where ring is Nx2 array of (lat, lon)
    lat, lon = point
    inside = False
    n = ring.shape[0]
    for i in range(n):
        j = (i + 1) % n
        yi, xi = ring[i, 0], ring[i, 1]
        yj, xj = ring[j, 0], ring[j, 1]
        intersect = ((xi > lon) != (xj > lon)) and (
            lat < (yj - yi) * (lon - xi) / (xj - xi + 1e-16) + yi
        )
        if intersect:
            inside = not inside
    return inside


def min_distance_to_rows_m(
    point: Tuple[float, float], rows: List[np.ndarray], polygons: List[List[np.ndarray]], ref_lat: float
) -> float:
    # If point is inside any polygon ring, distance is 0
    for poly in polygons:
        if not poly:
            continue
        # outer ring is poly[0], holes are poly[1:]
        if point_in_ring(point, poly[0]):
            return 0.0

    best = float("inf")
    # distance to line rows
    for row in rows:
        if row.shape[0] == 0:
            continue
        for i in range(row.shape[0] - 1):
            a = (float(row[i, 0]), float(row[i, 1]))
            b = (float(row[i + 1, 0]), float(row[i + 1, 1]))
            d = point_segment_distance_m(point, a, b, ref_lat)
            if d < best:
                best = d
                if best == 0.0:
                    return 0.0

    # distance to polygon edges (outer and inner rings)
    for poly in polygons:
        for ring in poly:
            if ring.shape[0] == 0:
                continue
            for i in range(ring.shape[0] - 1):
                a = (float(ring[i, 0]), float(ring[i, 1]))
                b = (float(ring[i + 1, 0]), float(ring[i + 1, 1]))
                d = point_segment_distance_m(point, a, b, ref_lat)
                if d < best:
                    best = d
                    if best == 0.0:
                        return 0.0

    return best


def filter_points_within_vinerows(
    coords: np.ndarray, props: List[dict], vine_rows_path: str, proximity_m: float
) -> Tuple[np.ndarray, List[dict]]:

    rows, polygons = load_geojson_lines(vine_rows_path)
    if not rows and not polygons:
        return coords, props

    # choose a reasonable reference latitude from vine rows/polygons mean
    all_lats = []
    for r in rows:
        if r.size:
            all_lats.extend(r[:, 0].tolist())
    for poly in polygons:
        for ring in poly:
            if ring.size:
                all_lats.extend(ring[:, 0].tolist())

    ref_lat = float(np.mean(all_lats)) if all_lats else float(np.mean(coords[:, 0]))

    keep_idx = []
    for i, (lat, lon) in enumerate(coords):
        d = min_distance_to_rows_m((lat, lon), rows, polygons, ref_lat)
        if d <= proximity_m:
            keep_idx.append(i)

    if not keep_idx:
        return np.empty((0, 2), dtype=float), []

    filtered_coords = coords[keep_idx]
    filtered_props = [props[i] for i in keep_idx]
    return filtered_coords, filtered_props


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
    parser.add_argument(
        "--vine-rows",
        default=None,
        help="Path to vine rows GeoJSON (LineString/MultiLineString). If provided, poles further than --vine-proximity-m will be removed",
    )
    parser.add_argument(
        "--vine-proximity-m",
        type=float,
        default=2.0,
        help="Maximum distance in metres from a vine row to keep a detected pole",
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

    if args.vine_rows:
        print(f"Inspecting vine rows: {args.vine_rows} (proximity {args.vine_proximity_m} m)")
        try:
            rows, polygons = load_geojson_lines(args.vine_rows)
        except Exception as exc:
            print(f"Failed to load vine rows {args.vine_rows}: {exc}. Skipping filtering.")
            rows, polygons = [], []

        if not polygons:
            print("No polygon features found in vine rows; skipping polygon-based filtering and using all pole detections for clustering.")
        else:
            filtered_coords, filtered_props = filter_points_within_vinerows(
                coords, props, args.vine_rows, args.vine_proximity_m
            )
            removed = coords.shape[0] - filtered_coords.shape[0]
            print(f"Filtered out {removed} pole detections not within {args.vine_proximity_m} m of a vine row")
            coords, props = filtered_coords, filtered_props

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
        "inference_results/hybrid_test/resnet101_20260210_163952/august_2024/65_feet/image_size_1280x960/poles.geojson",
        "--output-dir",
        "inference_results/hybrid_test/resnet101_20260210_163952/august_2024/65_feet/image_size_1280x960/clustered_poles",
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
        "--vine-rows",
        "inference_results/hybrid_test/resnet101_20260210_163952/august_2024/65_feet/image_size_1280x960/vine_rows.geojson",
        "--vine-proximity-m",
        "0.5", # Maximum distance in metres from a vine row to keep a detected pole (adjust based on expected GPS noise and row width)
        "--run-all", # Run all methods with the above parameters (comment out to run just one method)
    ]
    main(default_args if len(sys.argv) == 1 else None)
