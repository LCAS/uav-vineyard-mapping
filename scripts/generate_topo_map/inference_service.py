import os
import sys
import json
import uuid
import datetime
import math
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from image_gps_pixel_show_poles import extract_exif, extract_number, get_gps_from_pixel

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

FOCAL_LENGTH_MM = 4.5
SENSOR_WIDTH_MM = 6.17
SENSOR_HEIGHT_MM = 4.55

DEFAULT_IMAGE_SIZE = (1280, 960)
DEFAULT_MIN_DISTANCE = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL_CACHE: Dict[Tuple[str, str], nn.Module] = {}


class HybridUNetResNet(nn.Module):
    def __init__(self, backbone: str = "resnet101"):
        super().__init__()

        if backbone == "resnet50":
            resnet = models.resnet50(weights=None)
            enc_ch = [64, 256, 512, 1024, 2048]
        else:
            resnet = models.resnet101(weights=None)
            enc_ch = [64, 256, 512, 1024, 2048]

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.upconv4 = nn.ConvTranspose2d(enc_ch[4], enc_ch[3], 2, stride=2)
        self.dec4 = nn.Conv2d(enc_ch[3] + enc_ch[3], enc_ch[3], 3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(enc_ch[3], enc_ch[2], 2, stride=2)
        self.dec3 = nn.Conv2d(enc_ch[2] + enc_ch[2], enc_ch[2], 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(enc_ch[2], enc_ch[1], 2, stride=2)
        self.dec2 = nn.Conv2d(enc_ch[1] + enc_ch[1], enc_ch[1], 3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(enc_ch[1], enc_ch[0], 2, stride=2)
        self.dec1 = nn.Conv2d(enc_ch[0] + enc_ch[0], 64, 3, padding=1)

        self.head_reg = nn.Conv2d(64, 2, 1)
        self.head_seg = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x0 = self.encoder0(x)
        x1 = self.pool0(x0)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        d4 = F.interpolate(self.upconv4(x5), size=x4.shape[2:], mode="bilinear", align_corners=True)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)
        d3 = F.interpolate(self.upconv3(d4), size=x3.shape[2:], mode="bilinear", align_corners=True)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)
        d2 = F.interpolate(self.upconv2(d3), size=x2.shape[2:], mode="bilinear", align_corners=True)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        d1 = F.interpolate(self.upconv1(d2), size=x0.shape[2:], mode="bilinear", align_corners=True)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)

        out_reg = torch.sigmoid(self.head_reg(d1))
        out_reg = F.interpolate(out_reg, size=x.shape[2:], mode="bilinear", align_corners=True)
        out_seg = self.head_seg(d1)
        out_seg = F.interpolate(out_seg, size=x.shape[2:], mode="bilinear", align_corners=True)
        return out_reg, out_seg


def _load_model(model_path: str, backbone: str) -> nn.Module:
    cache_key = (model_path, backbone)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = HybridUNetResNet(backbone=backbone).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _resize_heatmap(heatmap: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(heatmap, target_shape, interpolation=cv2.INTER_LINEAR)


def _get_peak_coordinates(heatmap: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    coords = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=threshold)
    return coords[:, ::-1]


def _haversine_distance_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _pairwise_haversine_m(coords_latlon: np.ndarray) -> np.ndarray:
    n = coords_latlon.shape[0]
    dists = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        lat1, lon1 = coords_latlon[i]
        for j in range(i + 1, n):
            lat2, lon2 = coords_latlon[j]
            d = _haversine_distance_m(lat1, lon1, lat2, lon2)
            dists[i, j] = d
            dists[j, i] = d
    return dists


def _to_local_meters(points: List[Dict[str, float]]) -> Tuple[np.ndarray, Tuple[float, float]]:
    lats = np.array([p["lat"] for p in points], dtype=np.float64)
    lons = np.array([p["lon"] for p in points], dtype=np.float64)
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))
    lat_to_m = 111111.0
    lon_to_m = 111111.0 * np.cos(np.radians(center_lat))
    x = (lons - center_lon) * lon_to_m
    y = (lats - center_lat) * lat_to_m
    return np.column_stack([x, y]), (center_lat, center_lon)


def _latlon_to_xy(lat: float, lon: float, ref_lat: float) -> Tuple[float, float]:
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))
    x = lon * meters_per_deg_lon
    y = lat * meters_per_deg_lat
    return x, y


def _point_segment_distance_m(
    point: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
    ref_lat: float,
) -> float:
    px, py = _latlon_to_xy(point[0], point[1], ref_lat)
    ax, ay = _latlon_to_xy(a[0], a[1], ref_lat)
    bx, by = _latlon_to_xy(b[0], b[1], ref_lat)

    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay

    vv = vx * vx + vy * vy
    if vv == 0.0:
        return math.hypot(px - ax, py - ay)

    t = (wx * vx + wy * vy) / vv
    t = max(0.0, min(1.0, t))
    projx = ax + t * vx
    projy = ay + t * vy
    return math.hypot(px - projx, py - projy)


def _point_in_ring(point: Tuple[float, float], ring: np.ndarray) -> bool:
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


def _min_distance_to_rows_m(
    point: Tuple[float, float],
    rows: List[np.ndarray],
    polygons: List[List[np.ndarray]],
    ref_lat: float,
) -> float:
    for poly in polygons:
        if not poly:
            continue
        if _point_in_ring(point, poly[0]):
            return 0.0

    best = float("inf")
    for row in rows:
        if row.shape[0] == 0:
            continue
        for i in range(row.shape[0] - 1):
            a = (float(row[i, 0]), float(row[i, 1]))
            b = (float(row[i + 1, 0]), float(row[i + 1, 1]))
            d = _point_segment_distance_m(point, a, b, ref_lat)
            if d < best:
                best = d
                if best == 0.0:
                    return 0.0

    for poly in polygons:
        for ring in poly:
            if ring.shape[0] == 0:
                continue
            for i in range(ring.shape[0] - 1):
                a = (float(ring[i, 0]), float(ring[i, 1]))
                b = (float(ring[i + 1, 0]), float(ring[i + 1, 1]))
                d = _point_segment_distance_m(point, a, b, ref_lat)
                if d < best:
                    best = d
                    if best == 0.0:
                        return 0.0

    return best


def _filter_poles_by_vine_rows(
    poles: List[Dict[str, float]],
    vine_rows_geojson: Dict,
    max_distance_m: float = 0.5,
) -> List[Dict[str, float]]:
    """
    Filter poles to keep only those within vine row polygons or within max_distance_m of them.
    
    Args:
        poles: List of pole dictionaries with 'lat', 'lon' keys
        vine_rows_geojson: GeoJSON FeatureCollection of vine row polygons
        max_distance_m: Maximum distance in meters from vine rows to keep poles (default 0.5m)
    
    Returns:
        Filtered list of poles
    """
    print(f"[vine-row-filter] start poles={len(poles)} max_distance_m={max_distance_m}")

    if not poles:
        return []
    
    features = vine_rows_geojson.get("features", [])
    if not features:
        # No vine rows detected, return all poles
        print("[vine-row-filter] no vine row polygons; keeping all poles")
        return poles
    
    rows: List[np.ndarray] = []
    polygons: List[List[np.ndarray]] = []
    for feat in features:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if coords is None:
            continue
        if gtype == "LineString":
            rows.append(np.array([[c[1], c[0]] for c in coords], dtype=np.float64))
        elif gtype == "MultiLineString":
            for line in coords:
                rows.append(np.array([[c[1], c[0]] for c in line], dtype=np.float64))
        elif gtype == "Polygon":
            rings = [np.array([[c[1], c[0]] for c in ring], dtype=np.float64) for ring in coords]
            polygons.append(rings)
        elif gtype == "MultiPolygon":
            for poly in coords:
                rings = [np.array([[c[1], c[0]] for c in ring], dtype=np.float64) for ring in poly]
                polygons.append(rings)

    if not rows and not polygons:
        print("[vine-row-filter] no line/polygon features; keeping all poles")
        return poles

    all_lats = []
    for row in rows:
        if row.size:
            all_lats.extend(row[:, 0].tolist())
    for poly in polygons:
        for ring in poly:
            if ring.size:
                all_lats.extend(ring[:, 0].tolist())

    ref_lat = float(np.mean(all_lats)) if all_lats else float(np.mean([p["lat"] for p in poles]))

    filtered_poles = []
    rejected = 0
    for pole in poles:
        point = (pole["lat"], pole["lon"])
        d = _min_distance_to_rows_m(point, rows, polygons, ref_lat)
        if d <= max_distance_m:
            filtered_poles.append(pole)
        else:
            rejected += 1

    print(f"[vine-row-filter] kept={len(filtered_poles)} rejected={rejected} rows={len(rows)} polygons={len(polygons)}")
    return filtered_poles


def _estimate_k(points_xy: np.ndarray, eps_m: float) -> int:
    n_points = points_xy.shape[0]
    if n_points <= 1:
        return 1
    min_x, min_y = np.min(points_xy, axis=0)
    max_x, max_y = np.max(points_xy, axis=0)
    area = max(0.0, (max_x - min_x) * (max_y - min_y))
    if area <= 0 or eps_m <= 0:
        return min(1, n_points)
    approx_cluster_area = math.pi * (eps_m ** 2)
    k = int(max(1.0, area / approx_cluster_area))
    return max(1, min(n_points, k))


def _cluster_poles(points: List[Dict[str, float]], eps_m: float, algorithm: str) -> List[Dict[str, float]]:
    if not points:
        return []

    algo = (algorithm or "dbscan").lower()
    if algo in ("none", "no_clustering", "raw"):
        return [
            {
                "lat": p["lat"],
                "lon": p["lon"],
                "confidence": p["confidence"],
                "count": 1,
                "cluster_id": idx,
            }
            for idx, p in enumerate(points)
        ]

    if algo == "dbscan":
        coords = np.array([[p["lat"], p["lon"]] for p in points], dtype=np.float64)
        dists = _pairwise_haversine_m(coords)
        labels = DBSCAN(eps=eps_m, min_samples=1, metric="precomputed").fit_predict(dists)
    elif algo in ("agglomerative", "average"):
        coords = np.array([[p["lat"], p["lon"]] for p in points], dtype=np.float64)
        dists = _pairwise_haversine_m(coords)
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=eps_m,
                linkage="average",
                metric="precomputed",
            )
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=eps_m,
                linkage="average",
                affinity="precomputed",
            )
        labels = clustering.fit_predict(dists)
    elif algo == "kmeans":
        xy, _center = _to_local_meters(points)
        k = _estimate_k(xy, eps_m)
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(xy)
    elif algo == "hdbscan":
        try:
            import hdbscan
        except ImportError as exc:
            raise ImportError("hdbscan is not installed. Run `pip install hdbscan`. ") from exc
        coords = np.array([[p["lat"], p["lon"]] for p in points], dtype=np.float64)
        dists = _pairwise_haversine_m(coords)
        labels = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric="precomputed").fit_predict(dists)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    clusters: Dict[int, List[Dict[str, float]]] = {}
    for point, label in zip(points, labels):
        clusters.setdefault(int(label), []).append(point)

    merged = []
    for label, cluster_points in clusters.items():
        lat = float(np.mean([p["lat"] for p in cluster_points]))
        lon = float(np.mean([p["lon"] for p in cluster_points]))
        conf = float(np.max([p["confidence"] for p in cluster_points]))
        merged.append({
            "lat": lat,
            "lon": lon,
            "confidence": conf,
            "count": len(cluster_points),
            "cluster_id": label,
        })
    return merged


def _build_geojson(points: List[Dict[str, float]]) -> Dict:
    features = []
    for idx, p in enumerate(points):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
            "properties": {
                "confidence": p["confidence"],
                "count": p.get("count", 1),
                "id": f"pole_{idx}",
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _build_gps_converter(meta: Dict) -> Tuple[float, float, float, float, float, float]:
    flight_yaw = meta["yaw"]["flight"]
    gimbal_yaw = meta["yaw"]["gimbal"]
    gps_lat = meta["gps"]["lat"]
    gps_lon = meta["gps"]["lon"]
    gps_alt = meta["gps"]["alt"]
    return flight_yaw, gimbal_yaw, gps_lat, gps_lon, gps_alt


def run_inference(
    input_dir: str,
    model_path: str,
    cache_root: str,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    backbone: str = "resnet101",
    confidence_threshold: float = 0.4,
    cluster_eps_m: float = 1.5,
    cluster_algo: str = "dbscan",
    min_distance: int = DEFAULT_MIN_DISTANCE,
    filter_by_vine_rows: bool = False,
    progress_callback=None,
) -> Tuple[str, Dict, Dict]:
    model = _load_model(model_path, backbone)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Image folder not found: {input_dir}")

    session_id = uuid.uuid4().hex
    session_dir = os.path.join(cache_root, session_id)
    os.makedirs(session_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]
    image_files.sort()
    total_images = len(image_files)

    if progress_callback:
        progress_callback(0, max(total_images, 1), "Preparing images")

    all_points: List[Dict[str, float]] = []
    meta_images = []

    for idx, filename in enumerate(image_files):
        img_path = os.path.join(input_dir, filename)
        image_pil = Image.open(img_path).convert("RGB")
        original_w, original_h = image_pil.size

        exif = extract_exif(img_path)
        (
            flight_yaw,
            _flight_pitch,
            _flight_roll,
            gimbal_yaw,
            _gimbal_pitch,
            _gimbal_roll,
            gps_lat,
            gps_lon,
            gps_alt,
            _fov,
            _,
            _,
            _,
        ) = exif

        if gps_lat is None:
            continue

        flight_yaw_num = extract_number(flight_yaw)
        gimbal_yaw_num = extract_number(gimbal_yaw)
        if not gimbal_yaw_num:
            gimbal_yaw_num = flight_yaw_num
        gps_alt_num = extract_number(gps_alt) if gps_alt else 0.0

        input_img = image_pil.resize(image_size, Image.BILINEAR)
        image_np = np.array(input_img, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_reg, pred_seg = model(image_tensor)

        heatmaps = pred_reg[0].cpu().permute(1, 2, 0).numpy()
        pole_map_lowres = heatmaps[:, :, 0].astype(np.float32)
        row_prob_map = torch.sigmoid(pred_seg[0, 0]).cpu().numpy().astype(np.float32)

        base_name = os.path.splitext(filename)[0]
        npz_path = os.path.join(session_dir, f"{base_name}_outputs.npz")
        np.savez_compressed(npz_path, pole=pole_map_lowres, row=row_prob_map)

        pole_map_full = _resize_heatmap(pole_map_lowres, (original_w, original_h))
        pole_peaks = _get_peak_coordinates(pole_map_full, confidence_threshold, min_distance=min_distance)

        def to_gps(px: int, py: int) -> Tuple[float, float]:
            return get_gps_from_pixel(
                px,
                py,
                original_w,
                original_h,
                flight_yaw_num,
                gimbal_yaw_num,
                gps_lat,
                gps_lon,
                gps_alt_num,
                FOCAL_LENGTH_MM,
                SENSOR_WIDTH_MM,
                SENSOR_HEIGHT_MM,
            )

        for px, py in pole_peaks:
            lat, lon = to_gps(px, py)
            conf = float(pole_map_full[int(py), int(px)])
            all_points.append({"lat": lat, "lon": lon, "confidence": conf})

        meta_images.append({
            "image_path": img_path,
            "npz_path": npz_path,
            "original_size": [original_w, original_h],
            "gps": {"lat": float(gps_lat), "lon": float(gps_lon), "alt": float(gps_alt_num)},
            "yaw": {"flight": float(flight_yaw_num or 0.0), "gimbal": float(gimbal_yaw_num or 0.0)},
            "camera": {
                "focal_length_mm": FOCAL_LENGTH_MM,
                "sensor_width_mm": SENSOR_WIDTH_MM,
                "sensor_height_mm": SENSOR_HEIGHT_MM,
            },
        })

        if progress_callback:
            progress_callback(idx + 1, max(total_images, 1), f"Processed {idx + 1}/{max(total_images, 1)}")

    raw_points_count = len(all_points)
    poles_geojson = _build_geojson(all_points)

    vine_rows_geojson = {"type": "FeatureCollection", "features": []}
    for image_meta in meta_images:
        npz_path = image_meta["npz_path"]
        if not os.path.isfile(npz_path):
            continue
        data = np.load(npz_path)
        if "row" in data:
            row_prob_map = data["row"]
            original_w, original_h = image_meta["original_size"]
            row_mask_full = cv2.resize(
                (row_prob_map > 0.5).astype(np.uint8),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )
            contours, _ = cv2.findContours(
                row_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            flight_yaw, gimbal_yaw, gps_lat, gps_lon, gps_alt = _build_gps_converter(image_meta)

            def to_gps(px: int, py: int) -> Tuple[float, float]:
                return get_gps_from_pixel(
                    px,
                    py,
                    original_w,
                    original_h,
                    flight_yaw,
                    gimbal_yaw,
                    gps_lat,
                    gps_lon,
                    gps_alt,
                    FOCAL_LENGTH_MM,
                    SENSOR_WIDTH_MM,
                    SENSOR_HEIGHT_MM,
                )

            for cnt in contours:
                if cv2.contourArea(cnt) > 2000:
                    approx = cv2.approxPolyDP(cnt, 5.0, True)
                    row_poly_px = approx.reshape(-1, 2)
                    row_gps = []
                    for px, py in row_poly_px:
                        lat, lon = to_gps(px, py)
                        row_gps.append([lon, lat])
                    if len(row_gps) > 2:
                        row_gps.append(row_gps[0])
                        vine_rows_geojson["features"].append({
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": [row_gps]},
                            "properties": {"image": image_meta["image_path"]},
                        })

    filtered_points = all_points
    filter_elapsed_s = 0.0
    if filter_by_vine_rows:
        if progress_callback:
            progress_callback(max(total_images, 1), max(total_images, 1), "Filtering poles by vine rows")
        filter_start = time.perf_counter()
        filtered_points = _filter_poles_by_vine_rows(all_points, vine_rows_geojson, max_distance_m=0.5)
        filter_elapsed_s = time.perf_counter() - filter_start
        if progress_callback:
            progress_callback(max(total_images, 1), max(total_images, 1), "Filtering complete")

    filtered_count = len(filtered_points)
    poles_geojson = _build_geojson(filtered_points)

    stats = {
        "images_total": total_images,
        "images_with_gps": len(meta_images),
        "raw_poles": raw_points_count,
        "filtered_poles": filtered_count,
        "vine_row_filtering": bool(filter_by_vine_rows),
        "filter_elapsed_s": round(filter_elapsed_s, 3),
        "clustered_poles": None,
        "cluster_eps_m": cluster_eps_m,
        "cluster_algo": cluster_algo,
        "confidence_threshold": confidence_threshold,
        "min_distance_px": min_distance,
    }

    meta = {
        "session_id": session_id,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "image_size": [image_size[0], image_size[1]],
        "images": meta_images,
        "vine_rows_geojson": vine_rows_geojson,
    }
    meta_path = os.path.join(cache_root, f"{session_id}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return session_id, poles_geojson, stats, vine_rows_geojson


def load_session_meta(cache_root: str, session_id: str) -> Dict:
    meta_path = os.path.join(cache_root, f"{session_id}.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Session metadata not found: {session_id}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_filter(
    cache_root: str,
    session_id: str,
    confidence_threshold: float,
    cluster_eps_m: float,
    cluster_algo: str = "dbscan",
    min_distance: int = DEFAULT_MIN_DISTANCE,
    rows_geojson: Dict | None = None,
    filter_by_vine_rows: bool = True,
    progress_callback=None,
) -> Dict:
    meta = load_session_meta(cache_root, session_id)
    all_points: List[Dict[str, float]] = []

    images = meta.get("images", [])
    total_images = len(images)
    if progress_callback:
        progress_callback(0, max(total_images, 1), "Loading cached heatmaps")

    for idx, image_meta in enumerate(images):
        npz_path = image_meta["npz_path"]
        if not os.path.isfile(npz_path):
            continue

        data = np.load(npz_path)
        pole_map_lowres = data["pole"]
        original_w, original_h = image_meta["original_size"]

        pole_map_full = _resize_heatmap(pole_map_lowres, (original_w, original_h))
        pole_peaks = _get_peak_coordinates(pole_map_full, confidence_threshold, min_distance=min_distance)

        flight_yaw, gimbal_yaw, gps_lat, gps_lon, gps_alt = _build_gps_converter(image_meta)

        def to_gps(px: int, py: int) -> Tuple[float, float]:
            return get_gps_from_pixel(
                px,
                py,
                original_w,
                original_h,
                flight_yaw,
                gimbal_yaw,
                gps_lat,
                gps_lon,
                gps_alt,
                FOCAL_LENGTH_MM,
                SENSOR_WIDTH_MM,
                SENSOR_HEIGHT_MM,
            )

        for px, py in pole_peaks:
            lat, lon = to_gps(px, py)
            conf = float(pole_map_full[int(py), int(px)])
            all_points.append({"lat": lat, "lon": lon, "confidence": conf})

        if progress_callback:
            progress_callback(idx + 1, max(total_images, 1), f"Processed {idx + 1}/{max(total_images, 1)}")

    raw_points_count = len(all_points)
    filtered_points = all_points
    filter_elapsed_s = 0.0
    if filter_by_vine_rows:
        meta_rows_geojson = meta.get("vine_rows_geojson", {"type": "FeatureCollection", "features": []})
        if rows_geojson:
            merged_features = list(meta_rows_geojson.get("features", [])) + list(rows_geojson.get("features", []))
            vine_rows_geojson = {"type": "FeatureCollection", "features": merged_features}
        else:
            vine_rows_geojson = meta_rows_geojson
        if progress_callback:
            progress_callback(max(total_images, 1), max(total_images, 1), "Filtering poles by vine rows")
        filter_start = time.perf_counter()
        filtered_points = _filter_poles_by_vine_rows(all_points, vine_rows_geojson, max_distance_m=0.5)
        filter_elapsed_s = time.perf_counter() - filter_start
        if progress_callback:
            progress_callback(max(total_images, 1), max(total_images, 1), "Filtering complete")

    if progress_callback:
        progress_callback(max(total_images, 1), max(total_images, 1), "Clustering poles")

    clustered = _cluster_poles(filtered_points, eps_m=cluster_eps_m, algorithm=cluster_algo)
    poles_geojson = _build_geojson(clustered)

    stats = {
        "images_total": total_images,
        "images_with_gps": len(images),
        "raw_poles": raw_points_count,
        "clustered_poles": len(clustered),
        "filtered_poles": len(filtered_points),
        "vine_row_filtering": bool(filter_by_vine_rows),
        "filter_elapsed_s": round(filter_elapsed_s, 3),
        "cluster_eps_m": cluster_eps_m,
        "cluster_algo": cluster_algo,
        "confidence_threshold": confidence_threshold,
        "min_distance_px": min_distance,
    }

    return {"poles": poles_geojson, "stats": stats}
