import json
import os
from typing import List

import numpy as np
from tqdm import tqdm
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

# ======================
# EDIT THESE PARAMETERS
# ======================
INPUT_GEOJSON = "resnet_inference/vineyard_segmentation_paper_1/full_images_2_filtered/train_resnet101_20260203_135036/inference_results_full/39_feet/vine_rows.geojson"
OUTPUT_GEOJSON = "resnet_inference/vineyard_segmentation_paper_1/full_images_2_filtered/train_resnet101_20260203_135036/inference_results_full/39_feet/vine_rows_merged_polygons.geojson"
BUFFER_DEG = 0.000002
CLUSTER_BY_OFFSET = True
TARGET_ROW_COUNT = 10


def load_lines(geojson_path: str) -> List[LineString]:
    with open(geojson_path, "r") as f:
        data = json.load(f)

    lines = []
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            if len(coords) >= 2:
                lines.append(LineString(coords))
        elif geom.get("type") == "MultiLineString":
            for coords in geom.get("coordinates", []):
                if len(coords) >= 2:
                    lines.append(LineString(coords))
    return lines


def angle_of_line(line: LineString) -> float:
    coords = np.array(line.coords)
    if coords.shape[0] < 2:
        return 0.0
    dx = coords[-1, 0] - coords[0, 0]
    dy = coords[-1, 1] - coords[0, 1]
    return np.arctan2(dy, dx)


def dominant_angle(lines: List[LineString]) -> float:
    if not lines:
        return 0.0
    angles = np.array([angle_of_line(line) for line in lines])
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


def kmeans_1d(values: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
    if k <= 1:
        return np.zeros(len(values), dtype=int)
    centers = np.linspace(values.min(), values.max(), k)
    labels = np.zeros(len(values), dtype=int)

    for _ in range(max_iter):
        new_labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for i in range(k):
            if np.any(labels == i):
                centers[i] = values[labels == i].mean()
    return labels


def cluster_lines_by_offset(lines: List[LineString], target_rows: int) -> List[List[LineString]]:
    if len(lines) <= target_rows:
        return [lines]

    ang = dominant_angle(lines)
    normal = np.array([-np.sin(ang), np.cos(ang)])

    offsets = []
    for line in lines:
        centroid = line.centroid
        offsets.append(centroid.x * normal[0] + centroid.y * normal[1])
    offsets = np.array(offsets)

    labels = kmeans_1d(offsets, target_rows)
    clusters = [[] for _ in range(target_rows)]
    for line, label in zip(lines, labels):
        clusters[label].append(line)

    return [cluster for cluster in clusters if cluster]


def merge_to_polygons(lines: List[LineString], buffer_deg: float):
    if not lines:
        return []

    buffered = []
    for line in tqdm(lines, desc="Buffering lines", unit="line"):
        geom = line.buffer(buffer_deg)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_valid:
            buffered.append(geom)

    if not buffered:
        return []

    unioned = unary_union(buffered)
    if isinstance(unioned, Polygon):
        return [unioned]
    if isinstance(unioned, MultiPolygon):
        return list(unioned.geoms)
    if isinstance(unioned, GeometryCollection):
        return [g for g in unioned.geoms if isinstance(g, Polygon)]
    return []


def save_polygons(polygons: List[Polygon], output_path: str):
    features = []
    for idx, poly in enumerate(polygons):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(poly.exterior.coords)],
            },
            "properties": {
                "row_id": idx,
                "area": float(poly.area),
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)


def main():
    lines = load_lines(INPUT_GEOJSON)
    if not lines:
        raise SystemExit(f"No lines found in {INPUT_GEOJSON}")

    if CLUSTER_BY_OFFSET:
        clusters = cluster_lines_by_offset(lines, TARGET_ROW_COUNT)
        polygons = []
        for cluster in tqdm(clusters, desc="Merging clusters", unit="cluster"):
            polygons.extend(merge_to_polygons(cluster, BUFFER_DEG))
    else:
        polygons = merge_to_polygons(lines, BUFFER_DEG)
    if not polygons:
        raise SystemExit("No polygons produced. Try increasing BUFFER_DEG.")

    save_polygons(polygons, OUTPUT_GEOJSON)
    print(f"Merged {len(lines)} lines into {len(polygons)} polygons")
    print(f"Output: {OUTPUT_GEOJSON}")


if __name__ == "__main__":
    main()
