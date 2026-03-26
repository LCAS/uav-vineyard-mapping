import argparse
import json
import os
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import linemerge, unary_union

# ======================
# EDIT THESE PARAMETERS
# ======================
INPUT_GEOJSON = "resnet_inference/vineyard_segmentation_paper_1/full_images_2_filtered/train_resnet101_20260203_135036/inference_results_full/39_feet/vine_rows.geojson"
OUTPUT_GEOJSON = "resnet_inference/vineyard_segmentation_paper_1/full_images_2_filtered/train_resnet101_20260203_135036/inference_results_full/39_feet/vine_rows_merged.geojson"
USE_CLUSTERING = True
BUFFER_DEG = 0.000002
ANGLE_DEG = 12.0
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
    """Return angle in radians of the line's principal direction."""
    coords = np.array(line.coords)
    if coords.shape[0] < 2:
        return 0.0
    dx = coords[-1, 0] - coords[0, 0]
    dy = coords[-1, 1] - coords[0, 1]
    return np.arctan2(dy, dx)


def cluster_lines(lines: List[LineString], max_angle_diff_deg: float) -> List[List[LineString]]:
    """Cluster lines by orientation (simple binning)."""
    if not lines:
        return []

    max_angle_diff = np.deg2rad(max_angle_diff_deg)
    clusters: List[List[LineString]] = []

    for line in lines:
        ang = angle_of_line(line)
        placed = False
        for cluster in clusters:
            ref_ang = angle_of_line(cluster[0])
            if abs(np.arctan2(np.sin(ang - ref_ang), np.cos(ang - ref_ang))) <= max_angle_diff:
                cluster.append(line)
                placed = True
                break
        if not placed:
            clusters.append([line])

    return clusters


def merge_lines(lines: List[LineString], buffer_deg: float) -> List[LineString]:
    """Merge overlapping or touching lines using buffer + union + linemerge."""
    if not lines:
        return []

    buffered = []
    for line in lines:
        geom = line.buffer(buffer_deg)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_valid:
            buffered.append(geom)

    if not buffered:
        return []

    try:
        unioned = unary_union(buffered)
    except Exception:
        cleaned = [geom.buffer(0) for geom in buffered]
        cleaned = [geom for geom in cleaned if geom.is_valid]
        if not cleaned:
            return []
        unioned = unary_union(cleaned)
    if isinstance(unioned, (Polygon, MultiPolygon)):
        unioned = unioned.boundary
    elif isinstance(unioned, GeometryCollection):
        boundary_parts = []
        for geom in unioned.geoms:
            if isinstance(geom, (Polygon, MultiPolygon)):
                boundary_parts.append(geom.boundary)
            elif isinstance(geom, (LineString, MultiLineString)):
                boundary_parts.append(geom)
        if boundary_parts:
            unioned = unary_union(boundary_parts)
    if isinstance(unioned, LineString):
        return [unioned]
    if isinstance(unioned, MultiLineString):
        merged = linemerge(unioned)
    else:
        merged = linemerge(unioned)

    if isinstance(merged, LineString):
        return [merged]
    if isinstance(merged, MultiLineString):
        return list(merged.geoms)
    return []


def save_lines(lines: List[LineString], output_path: str):
    features = []
    for idx, line in enumerate(lines):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": list(line.coords),
            },
            "properties": {
                "row_id": idx,
                "num_points": len(line.coords),
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)


def line_centroid(line: LineString) -> Tuple[float, float]:
    centroid = line.centroid
    return centroid.x, centroid.y


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


def reduce_to_row_count(lines: List[LineString], target_rows: int, buffer_deg: float) -> List[LineString]:
    if len(lines) <= target_rows:
        return lines

    # Use dominant row orientation and cluster by offset along the normal.
    ang = dominant_angle(lines)
    normal = np.array([-np.sin(ang), np.cos(ang)])

    offsets = []
    for line in lines:
        cx, cy = line_centroid(line)
        offsets.append(cx * normal[0] + cy * normal[1])
    offsets = np.array(offsets)

    labels = kmeans_1d(offsets, target_rows)

    reduced_lines = []
    for row_idx in range(target_rows):
        group = [line for line, label in zip(lines, labels) if label == row_idx]
        if group:
            merged_group = merge_lines(group, buffer_deg)
            reduced_lines.extend(merged_group)
    return reduced_lines


def main():
    parser = argparse.ArgumentParser(description="Merge vine row LineStrings across images.")
    parser.add_argument("--input", default=INPUT_GEOJSON, help="Path to vine_rows.geojson")
    parser.add_argument("--output", default=OUTPUT_GEOJSON, help="Path to merged vine rows GeoJSON")
    parser.add_argument("--buffer-deg", type=float, default=BUFFER_DEG, help="Buffer in degrees for merging (approx).")
    parser.add_argument("--angle-deg", type=float, default=ANGLE_DEG, help="Max angle diff for clustering lines.")
    parser.add_argument("--cluster", action="store_true", default=USE_CLUSTERING, help="Cluster by orientation before merging.")
    args = parser.parse_args()

    lines = load_lines(args.input)
    if not lines:
        raise SystemExit(f"No lines found in {args.input}")

    if args.cluster:
        clusters = cluster_lines(lines, args.angle_deg)
        merged_lines = []
        for cluster in tqdm(clusters, desc="Merging clusters", unit="cluster"):
            merged_lines.extend(merge_lines(cluster, args.buffer_deg))
    else:
        merged_lines = merge_lines(lines, args.buffer_deg)

    merged_lines = reduce_to_row_count(merged_lines, TARGET_ROW_COUNT, args.buffer_deg)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_lines(merged_lines, args.output)
    print(f"Merged {len(lines)} lines into {len(merged_lines)} lines")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
