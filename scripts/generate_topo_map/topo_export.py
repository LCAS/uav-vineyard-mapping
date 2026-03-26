import datetime
import math
from typing import Dict, List, Tuple

import yaml
import numpy as np


# --- HELPER FUNCTIONS ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def convert_to_meters(coordinates, center_coordinates):
    center_lon, center_lat = center_coordinates
    lon, lat = coordinates
    lat_to_m = 111111
    lon_to_m = 111111 * math.cos(math.radians(center_lat))
    x_dist = (lon - center_lon) * lon_to_m
    y_dist = (lat - center_lat) * lat_to_m
    return x_dist, y_dist


def find_centre(topo_map_point_list, topo_map_line_list):
    point_coords = [point["coordinates"] for point in topo_map_point_list]
    line_coords = [coord for line in topo_map_line_list for coord in line["coordinates"]]
    all_coords = point_coords + line_coords
    if not all_coords:
        return (0.0, 0.0)
    latitudes = [coord[1] for coord in all_coords]
    longitudes = [coord[0] for coord in all_coords]
    # Return as (lon, lat) to match convert_to_meters and downstream consumers
    return (sum(longitudes) / len(longitudes), sum(latitudes) / len(latitudes))


def bearing(point1, point2):
    """Calculate bearing between two points (lat, lon)."""
    lat1, lon1 = point1
    lat2, lon2 = point2
    d_lon = lon2 - lon1
    y = math.sin(math.radians(d_lon)) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.cos(math.radians(d_lon))
    initial_bearing = math.atan2(y, x)
    return math.degrees(initial_bearing)


def interpolate_along_line(start_coord, end_coord, spacing_m):
    """Interpolate points along a line at regular spacing (in meters)."""
    from geopy.distance import geodesic
    lat1, lon1 = start_coord[1], start_coord[0]
    lat2, lon2 = end_coord[1], end_coord[0]
    
    total_distance = geodesic((lat1, lon1), (lat2, lon2)).meters
    num_points = max(2, int(total_distance / spacing_m) + 1)
    
    points = []
    for i in range(num_points):
        fraction = i / (num_points - 1) if num_points > 1 else 0
        lat = lat1 + (lat2 - lat1) * fraction
        lon = lon1 + (lon2 - lon1) * fraction
        points.append([lon, lat])
    
    return points


def extend_row_endpoints(start_coord, end_coord, extend_distance_m):
    """
    Extend a row line by extend_distance_m at both ends.
    Returns (new_start, new_end) in [lon, lat] format.
    """
    from geopy.distance import geodesic
    lat1, lon1 = start_coord[1], start_coord[0]
    lat2, lon2 = end_coord[1], end_coord[0]
    
    # Calculate bearing from start to end
    bear = bearing((lat1, lon1), (lat2, lon2))
    
    # Extend start point backwards
    new_start = geodesic(meters=extend_distance_m).destination((lat1, lon1), bear + 180)
    
    # Extend end point forwards
    new_end = geodesic(meters=extend_distance_m).destination((lat2, lon2), bear)
    
    return [new_start.longitude, new_start.latitude], [new_end.longitude, new_end.latitude]


def _to_local_xy(coords: List[List[float]], center_lat: float, center_lon: float) -> np.ndarray:
    lat_to_m = 111111.0
    lon_to_m = 111111.0 * math.cos(math.radians(center_lat))
    arr = np.array(coords, dtype=np.float64)
    x = (arr[:, 0] - center_lon) * lon_to_m
    y = (arr[:, 1] - center_lat) * lat_to_m
    return np.column_stack([x, y])


def _dominant_direction(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] < 2:
        return np.array([1.0, 0.0])
    cov = np.cov(points_xy.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    norm = np.linalg.norm(principal)
    if norm == 0:
        return np.array([1.0, 0.0])
    return principal / norm


def _order_rows_and_midlines(rows: List[List[List[float]]]) -> List[Dict[str, List[List[float]]]]:
    """
    Order rows perpendicular to the dominant row direction using PCA.
    This correctly handles rows running in any direction (N-S, E-W, NW-SE, etc.)
    by sorting them along the direction perpendicular to how they run.
    """
    if len(rows) < 2:
        return []

    # Get all coordinates
    all_coords = [coord for row in rows for coord in row]
    if not all_coords:
        return []

    # Find geographic center
    lons = np.array([c[0] for c in all_coords])
    lats = np.array([c[1] for c in all_coords])
    center_lon = float(np.mean(lons))
    center_lat = float(np.mean(lats))

    # Convert all coordinates to local meters for accurate PCA
    lat_to_m = 111111.0
    lon_to_m = 111111.0 * math.cos(math.radians(center_lat))
    
    xy_coords = np.array([
        [(lon - center_lon) * lon_to_m, (lat - center_lat) * lat_to_m]
        for lon, lat in zip(lons, lats)
    ], dtype=np.float64)

    # Compute PCA to find dominant row direction
    if xy_coords.shape[0] > 1:
        cov_matrix = np.cov(xy_coords.T)
        eigvals, eigvecs = np.linalg.eig(cov_matrix)
        dominant_idx = int(np.argmax(eigvals))
        dominant_dir = eigvecs[:, dominant_idx].astype(np.float64)
    else:
        dominant_dir = np.array([1.0, 0.0])

    # Normalize dominant direction
    norm = np.linalg.norm(dominant_dir)
    if norm > 0:
        dominant_dir = dominant_dir / norm

    # Perpendicular direction (rotate 90 degrees) - this is the direction between rows
    perp_dir = np.array([-dominant_dir[1], dominant_dir[0]])

    # Get row centers and project onto perpendicular direction
    row_projections = []
    for row_coords in rows:
        row_lons = np.array([c[0] for c in row_coords])
        row_lats = np.array([c[1] for c in row_coords])
        
        row_center_lon = float(np.mean(row_lons))
        row_center_lat = float(np.mean(row_lats))

        # Convert to meters
        x = (row_center_lon - center_lon) * lon_to_m
        y = (row_center_lat - center_lat) * lat_to_m
        xy = np.array([x, y])

        # Project onto perpendicular direction
        proj = float(np.dot(xy, perp_dir))

        row_projections.append({
            "proj": proj,
            "start": row_coords[0],
            "end": row_coords[-1],
        })

    # Sort rows by perpendicular projection
    row_projections.sort(key=lambda x: x["proj"])

    # Create midlines between consecutive ordered rows
    midlines = []
    for idx in range(len(row_projections) - 1):
        row_a = row_projections[idx]
        row_b = row_projections[idx + 1]

        mid_start = [
            (row_a["start"][0] + row_b["start"][0]) / 2.0,
            (row_a["start"][1] + row_b["start"][1]) / 2.0,
        ]
        mid_end = [
            (row_a["end"][0] + row_b["end"][0]) / 2.0,
            (row_a["end"][1] + row_b["end"][1]) / 2.0,
        ]
        midlines.append({
            "mid_row_id": f"row_{idx}_midline",
            "coordinates": [mid_start, mid_end],
        })

    return midlines


# --- AISLE-BASED TOPOLOGY (ported from debug_topo_map.py) ---
def get_lat_lon_scales(lat: float):
    """Return meters per degree for lat and lon at a specific latitude."""
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * math.radians(lat))
    m_per_deg_lon = 111412.84 * math.cos(math.radians(lat)) - 93.5 * math.cos(3 * math.radians(lat))
    return m_per_deg_lat, m_per_deg_lon


def _distance_m(p1: List[float], p2: List[float]) -> float:
    """Calculate Euclidean distance in meters between two [lon, lat] points."""
    lat_scale, lon_scale = get_lat_lon_scales(p1[1])
    dx = (p1[0] - p2[0]) * lon_scale
    dy = (p1[1] - p2[1]) * lat_scale
    return math.sqrt(dx * dx + dy * dy)


def _interpolate_line(p1: List[float], p2: List[float], fraction: float) -> List[float]:
    """Return a point at 'fraction' (0.0 to 1.0) between p1 and p2."""
    return [p1[0] + (p2[0] - p1[0]) * fraction, p1[1] + (p2[1] - p1[1]) * fraction]


def _extend_point(start_p: List[float], end_p: List[float], dist_m: float) -> List[float]:
    """Return a new point extended 'dist_m' away from end_p, along the vector start_p->end_p."""
    lat_scale, lon_scale = get_lat_lon_scales(end_p[1])

    # Vector in degrees
    vec_x = end_p[0] - start_p[0]
    vec_y = end_p[1] - start_p[1]

    # Normalize vector length roughly to meters
    len_deg = math.sqrt((vec_x * lon_scale) ** 2 + (vec_y * lat_scale) ** 2)
    if len_deg == 0:
        return end_p

    # Scale vector to extend distance
    scale_factor = dist_m / len_deg

    new_lon = end_p[0] + (vec_x * lon_scale * scale_factor) / lon_scale
    new_lat = end_p[1] + (vec_y * lat_scale * scale_factor) / lat_scale

    return [new_lon, new_lat]


def _polygon_to_centerline(polygon_coords: List[List[float]], num_samples: int = 32) -> List[List[float]]:
    """Approximate a polygon's centreline by sampling along its major axis.

    Steps:
    - Convert exterior ring to local XY about the polygon centroid
    - Compute principal axis (PCA) of the polygon points
    - For evenly spaced positions along the principal axis, find min/max
      perpendicular offsets and take their midpoint
    - Convert midpoints back to (lon, lat)

    This is a heuristic but works well for narrow vine-row polygons.
    """
    if not polygon_coords:
        return []

    ring = polygon_coords[0]
    arr = np.array(ring, dtype=np.float64)  # shape (N, 2) columns: lon, lat
    if arr.shape[0] < 3:
        return arr.tolist()

    center_lon = float(arr[:, 0].mean())
    center_lat = float(arr[:, 1].mean())

    # to local meters
    lat_to_m = 111111.0
    lon_to_m = 111111.0 * math.cos(math.radians(center_lat))
    xy = np.column_stack([ (arr[:, 0] - center_lon) * lon_to_m, (arr[:, 1] - center_lat) * lat_to_m ])

    # PCA / dominant direction
    cov = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    dominant_idx = int(np.argmax(eigvals))
    dominant = eigvecs[:, dominant_idx].astype(np.float64)
    if np.linalg.norm(dominant) == 0:
        dominant = np.array([1.0, 0.0])
    dominant = dominant / np.linalg.norm(dominant)
    perp = np.array([-dominant[1], dominant[0]])

    projections = xy @ dominant
    perp_offsets = xy @ perp

    min_p, max_p = float(projections.min()), float(projections.max())
    if max_p - min_p < 1e-6:
        # degenerate -> return centroid
        return [[float(center_lon), float(center_lat)]]

    samples = np.linspace(min_p, max_p, num_samples)
    centerline_xy = []
    for s in samples:
        # find points with nearest projection to s
        idx = np.argsort(np.abs(projections - s))[:8]
        sel_proj = projections[idx]
        sel_perp = perp_offsets[idx]
        min_perp = float(np.min(sel_perp))
        max_perp = float(np.max(sel_perp))
        mid_perp = 0.5 * (min_perp + max_perp)
        pt_xy = dominant * s + perp * mid_perp
        # back to lon/lat
        lon = center_lon + (pt_xy[0] / lon_to_m)
        lat = center_lat + (pt_xy[1] / lat_to_m)
        centerline_xy.append([lon, lat])

    # remove duplicates / near-duplicates
    out = []
    prev = None
    for p in centerline_xy:
        if prev is None or haversine(prev[1], prev[0], p[1], p[0]) > 0.01:
            out.append(p)
            prev = p
    return out


def generate_aisle_topology(vine_rows_geojson: Dict, node_spacing_m: float = 2.0, extend_distance_m: float = 3.0, cross_row_distance_m: float = 6.0):
    """
    Generate aisle (midline) topology between adjacent, sorted vine rows.
    Returns (nodes, edges) where nodes are dicts containing `topo_map_node_id` and `coordinates`.
    """
    features = vine_rows_geojson.get("features", [])

    # Accept both LineString rows and narrow Polygon rows (convert polygons -> centerline)
    rows = []
    for f in features:
        geom = f.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not coords:
            continue
        if gtype == "LineString":
            rows.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords}})
        elif gtype == "Polygon":
            centerline = _polygon_to_centerline(coords)
            if centerline and len(centerline) >= 2:
                rows.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": centerline}})

    if len(rows) < 2:
        return [], []

    # Sort rows West -> East by centroid longitude (stable and simple)
    def _row_centroid_lon(feature):
        coords = feature["geometry"]["coordinates"]
        return sum(c[0] for c in coords) / len(coords)

    rows.sort(key=_row_centroid_lon)

    aisles: List[List[List[float]]] = []
    nodes: List[Dict] = []
    edges: List[Dict] = []

    # Build aisles (midlines) between consecutive rows
    for i in range(len(rows) - 1):
        row_left = rows[i]["geometry"]["coordinates"]
        row_right = rows[i + 1]["geometry"]["coordinates"]

        # Ensure consistent direction between left and right
        dist_start_start = _distance_m(row_left[0], row_right[0])
        dist_start_end = _distance_m(row_left[0], row_right[-1])
        if dist_start_end < dist_start_start:
            row_right = list(reversed(row_right))

        num_samples = max(len(row_left), len(row_right), 10)
        midline_coords: List[List[float]] = []

        for k in range(num_samples):
            frac = k / (num_samples - 1)
            idx_l = frac * (len(row_left) - 1)
            p_l_base = row_left[int(idx_l)]
            p_l_next = row_left[min(int(idx_l) + 1, len(row_left) - 1)]
            p_left = _interpolate_line(p_l_base, p_l_next, idx_l % 1)

            idx_r = frac * (len(row_right) - 1)
            p_r_base = row_right[int(idx_r)]
            p_r_next = row_right[min(int(idx_r) + 1, len(row_right) - 1)]
            p_right = _interpolate_line(p_r_base, p_r_next, idx_r % 1)

            mid_pt = [(p_left[0] + p_right[0]) / 2.0, (p_left[1] + p_right[1]) / 2.0]
            midline_coords.append(mid_pt)

        # Extend midline endpoints
        if len(midline_coords) >= 2:
            start_ext = _extend_point(midline_coords[1], midline_coords[0], extend_distance_m)
            end_ext = _extend_point(midline_coords[-2], midline_coords[-1], extend_distance_m)
            midline_coords.insert(0, start_ext)
            midline_coords.append(end_ext)

        aisles.append(midline_coords)

    # Create nodes along aisles and linear edges
    aisle_node_ids: List[List[str]] = []
    for aisle_idx, coords in enumerate(aisles):
        current_ids: List[str] = []
        total_len = 0.0
        segment_lengths: List[float] = []
        for k in range(len(coords) - 1):
            d = _distance_m(coords[k], coords[k + 1])
            segment_lengths.append(d)
            total_len += d

        num_nodes = max(1, int(total_len / node_spacing_m))
        # Ensure at least 2 nodes (start/end)
        num_nodes = max(2, num_nodes)

        for n in range(num_nodes + 1):
            target_dist = (n / num_nodes) * total_len
            cum_dist = 0.0
            node_coord = coords[-1]
            for k, seg_len in enumerate(segment_lengths):
                if cum_dist + seg_len >= target_dist:
                    remain = target_dist - cum_dist
                    frac = remain / seg_len if seg_len > 0 else 0.0
                    node_coord = _interpolate_line(coords[k], coords[k + 1], frac)
                    break
                cum_dist += seg_len

            node_id = f"aisle_{aisle_idx}_node_{n}"
            current_ids.append(node_id)
            nodes.append({
                "topo_map_node_id": node_id,
                "coordinates": node_coord,
                "aisle_idx": aisle_idx,
                "node_idx": n,
            })

            if n > 0:
                edges.append({"from": current_ids[-2], "to": node_id, "type": "linear"})

        aisle_node_ids.append(current_ids)

    # Headland connections between adjacent aisles (start-to-start, end-to-end)
    for i in range(len(aisle_node_ids) - 1):
        edges.append({"from": aisle_node_ids[i][0], "to": aisle_node_ids[i + 1][0], "type": "headland"})
        edges.append({"from": aisle_node_ids[i][-1], "to": aisle_node_ids[i + 1][-1], "type": "headland"})

    # Optionally, add short cross-aisle links if close (honor cross_row_distance_m)
    if cross_row_distance_m and cross_row_distance_m > 0:
        for i, ids_a in enumerate(aisle_node_ids):
            for j in range(i + 1, len(aisle_node_ids)):
                ids_b = aisle_node_ids[j]
                for a_id in (ids_a[:1] + ids_a[-1:]):
                    coord_a = next((n["coordinates"] for n in nodes if n["topo_map_node_id"] == a_id), None)
                    for b_id in (ids_b[:1] + ids_b[-1:]):
                        coord_b = next((n["coordinates"] for n in nodes if n["topo_map_node_id"] == b_id), None)
                        if coord_a and coord_b:
                            d = _distance_m(coord_a, coord_b)
                            if d <= cross_row_distance_m:
                                edges.append({"from": a_id, "to": b_id, "type": "cross"})

    # --- Ensure every edge has a reciprocal (reverse) edge for all edge types ---
    # Build a set of (from,to,type) to detect duplicates and then add reverse entries.
    seen = set()
    new_edges = []
    for e in edges:
        key = (e.get("from"), e.get("to"), e.get("type"))
        if key in seen:
            continue
        seen.add(key)
        new_edges.append(e)
        rev_key = (e.get("to"), e.get("from"), e.get("type"))
        if rev_key not in seen:
            # append a reversed copy
            seen.add(rev_key)
            new_edges.append({"from": e.get("to"), "to": e.get("from"), "type": e.get("type")})

    edges = new_edges

    return nodes, edges


def build_topological_nodes_from_rows(
    rows_geojson,
    poles_geojson,
    node_spacing_m=2.0,
    extend_distance_m=3.0,
    cross_row_distance_m=6.0,
):
    """
    Build topological nodes by creating midlines between adjacent vine rows.
    Nodes are placed along each midline, with edges along the midline and
    optional end connections between nearby midlines.
    """
    # Prefer aisle-based topology generation (midlines between ordered rows)
    try:
        nodes, edges = generate_aisle_topology(
            rows_geojson,
            node_spacing_m=node_spacing_m,
            extend_distance_m=extend_distance_m,
            cross_row_distance_m=cross_row_distance_m,
        )
        return nodes, edges
    except Exception:
        # Fallback: no nodes
        return [], []


def generate_datum_yaml(center_coordinates):
    # center_coordinates is (lon, lat).  Coerce to plain Python floats to avoid
    # YAML emitting numpy scalar objects.
    lon, lat = center_coordinates
    return yaml.dump({"datum_latitude": float(lat), "datum_longitude": float(lon)}, default_flow_style=False)


def generate_topological_map(topo_map_point_list, last_updated, metric_map, name, center_coordinates, edges_list=None):
    # Build a tmap2-style structure (matches Orion `TMapTemplates` top-level keys)
    topo_map_data = {
        "meta": {"last_updated": last_updated},
        "metric_map": metric_map,
        "name": name,
        "pointset": name,
        "transformation": {
            "child": "topo_map",
            "parent": "map",
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
        "verts": {
            "verts": [
                {"verts": [
                    {"x": -0.13, "y":  0.213},
                    {"x": -0.242, "y":  0.059},
                    {"x": -0.213, "y": -0.13},
                    {"x": -0.059, "y": -0.242},
                    {"x":  0.13, "y": -0.213},
                    {"x":  0.242, "y": -0.059},
                    {"x":  0.213, "y":  0.13},
                    {"x":  0.059, "y":  0.242},
                ]},
                {"verts": [
                    {"x":  0.640, "y": -1.070},
                    {"x":  0.875, "y": -0.355},
                    {"x":  0.740, "y":  0.590},
                    {"x":  0.305, "y":  1.205},
                    {"x": -0.640, "y":  1.070},
                    {"x": -0.875, "y":  0.355},
                    {"x": -0.740, "y": -0.590},
                    {"x": -0.305, "y": -1.205},
                ]},
            ]
        },
        "nodes": [],
    }
    node_dict = {}

    # 1. Create Nodes
    for point in topo_map_point_list:
        lon_meters, lat_meters = convert_to_meters(point["coordinates"], center_coordinates)
        node_id = point["topo_map_node_id"]
        node_entry = {
            "meta": {"map": metric_map, "node": node_id, "pointset": name},
            "node": {
                "edges": [],
                "localise_by_topic": "",
                "name": node_id,
                "parent_frame": "map",
                "pose": {
                    "orientation": {"w": 1, "x": 0, "y": 0, "z": 0},
                    "position": {"x": lon_meters, "y": lat_meters, "z": 0},
                },
                "properties": {"xy_goal_tolerance": 0.3, "yaw_goal_tolerance": 0.1},
                "restrictions_planning": True,
                "restrictions_runtime": "obstacleFree_1",
                # Reference the predefined node shape (verts) from the top-level `verts` so
                # YAML dumper emits an alias (e.g. `verts: *vert1`) instead of repeating
                # per-node vert geometry.
                "verts": topo_map_data["verts"]["verts"][1]["verts"],
            },
        }
        topo_map_data["nodes"].append(node_entry)
        node_dict[node_id] = point

    # 2. Create Edges
    if edges_list:
        for edge in edges_list:
            from_node = edge.get("from")
            to_node = edge.get("to")
            if from_node in node_dict and to_node in node_dict:
                # Use the richer edge structure used by Orion (includes action_type, goal template, etc.)
                edge_entry = {
                    "action": "move_base",
                    "action_type": "move_base_msgs/MoveBaseGoal",
                    "config": [],
                    "edge_id": f"{from_node}_to_{to_node}",
                    "fail_policy": "fail",
                    "fluid_navigation": True,
                    "goal": {
                        "target_pose": {
                            "header": {"frame_id": "$node.parent_frame"},
                            "pose": "$node.pose",
                        }
                    },
                    "node": to_node,
                    "recovery_behaviours_config": "",
                    "restrictions_planning": True,
                    "restrictions_runtime": "obstacleFree_1",
                }
                for node_entry in topo_map_data["nodes"]:
                    if node_entry["node"]["name"] == from_node:
                        node_entry["node"]["edges"].append(edge_entry)
                        break

    return yaml.dump(topo_map_data)


def _round_coord(coord: List[float], precision: int = 7) -> Tuple[float, float]:
    return (round(coord[0], precision), round(coord[1], precision))


def build_topological_yaml(
    poles_geojson: Dict,
    rows_geojson: Dict,
    map_name: str,
    metric_map: str,
    node_spacing_m: float = 2.0,
    extend_distance_m: float = 3.0,
    cross_row_distance_m: float = 6.0,
) -> Tuple[str, str]:
    """
    Build topological map by interpolating nodes along detected vine rows.
    Rows are extended by extend_distance_m at both ends.
    Nodes are placed at regular intervals along each extended row.
    Edges connect consecutive nodes on the same row and nearby nodes across rows.
    
    Args:
        poles_geojson: GeoJSON of detected poles
        rows_geojson: GeoJSON of detected vine rows
        map_name: Name for the topological map
        metric_map: Name of the metric map
        node_spacing_m: Distance between nodes along rows (meters)
        extend_distance_m: Distance to extend each row at both ends (meters)
        cross_row_distance_m: Maximum distance for cross-row connections (meters)
    
    Returns:
        Tuple of (topo_yaml, datum_yaml)
    """
    # Build nodes and edges from rows
    topo_nodes, edge_connections = build_topological_nodes_from_rows(
        rows_geojson, 
        poles_geojson, 
        node_spacing_m=node_spacing_m,
        extend_distance_m=extend_distance_m,
        cross_row_distance_m=cross_row_distance_m
    )
    
    if not topo_nodes:
        # Fallback to pole-based topological map if row-based fails
        return _build_topological_yaml_from_poles(poles_geojson, rows_geojson, map_name, metric_map)
    
    topo_map_line_list = []
    for feature in rows_geojson.get("features", []):
        if feature.get("geometry", {}).get("type") != "LineString":
            continue
        coords = feature["geometry"]["coordinates"]
        topo_map_line_list.append({"coordinates": coords})

    center_coordinates = find_centre(topo_nodes, topo_map_line_list)
    last_updated = datetime.datetime.utcnow().isoformat() + "Z"

    topo_yaml = generate_topological_map(
        topo_nodes,
        last_updated,
        metric_map,
        map_name,
        center_coordinates,
        edges_list=edge_connections,
    )
    datum_yaml = generate_datum_yaml(center_coordinates)
    return topo_yaml, datum_yaml


def _build_topological_yaml_from_poles(
    poles_geojson: Dict,
    rows_geojson: Dict,
    map_name: str,
    metric_map: str,
) -> Tuple[str, str]:
    """
    Fallback: Build topological map using poles as nodes (original method).
    """
    topo_map_point_list = []
    coord_to_node = {}

    for idx, feature in enumerate(poles_geojson.get("features", [])):
        coord = feature["geometry"]["coordinates"]
        node_id = f"node_{idx + 1:03d}"
        topo_map_point_list.append({
            "coordinates": coord,
            "topo_map_node_id": node_id,
            "neighbors": [],
        })
        coord_to_node[_round_coord(coord)] = node_id

    topo_map_line_list = []
    for feature in rows_geojson.get("features", []):
        if feature.get("geometry", {}).get("type") != "LineString":
            continue
        coords = feature["geometry"]["coordinates"]
        topo_map_line_list.append({"coordinates": coords})
        for a, b in zip(coords[:-1], coords[1:]):
            node_a = coord_to_node.get(_round_coord(a))
            node_b = coord_to_node.get(_round_coord(b))
            if not node_a or not node_b:
                continue
            for point in topo_map_point_list:
                if point["topo_map_node_id"] == node_a and node_b not in point["neighbors"]:
                    point["neighbors"].append(node_b)
                if point["topo_map_node_id"] == node_b and node_a not in point["neighbors"]:
                    point["neighbors"].append(node_a)

    center_coordinates = find_centre(topo_map_point_list, topo_map_line_list)
    last_updated = datetime.datetime.utcnow().isoformat() + "Z"

    topo_yaml = generate_topological_map(
        topo_map_point_list,
        last_updated,
        metric_map,
        map_name,
        center_coordinates,
    )
    datum_yaml = generate_datum_yaml(center_coordinates)
    return topo_yaml, datum_yaml
