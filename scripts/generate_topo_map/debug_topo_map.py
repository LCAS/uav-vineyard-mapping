#!/usr/bin/env python3
"""
Debug script to generate topological map from saved GeoJSON files.
NOW CALCULATING MIDLINES (AISLES) INSTEAD OF ON-ROW PATHS.
"""

import json
import sys
import os
import glob
import math
from copy import deepcopy

# --- GEOMETRY HELPER FUNCTIONS ---

def get_lat_lon_scales(lat):
    """Return meters per degree for lat and lon at a specific latitude."""
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * math.radians(lat))
    m_per_deg_lon = 111412.84 * math.cos(math.radians(lat)) - 93.5 * math.cos(3 * math.radians(lat))
    return m_per_deg_lat, m_per_deg_lon

def distance_m(p1, p2):
    """Calculate Euclidean distance in meters between two [lon, lat] points."""
    lat_scale, lon_scale = get_lat_lon_scales(p1[1])
    dx = (p1[0] - p2[0]) * lon_scale
    dy = (p1[1] - p2[1]) * lat_scale
    return math.sqrt(dx**2 + dy**2)

def interpolate_line(p1, p2, fraction):
    """Return a point at 'fraction' (0.0 to 1.0) between p1 and p2."""
    return [
        p1[0] + (p2[0] - p1[0]) * fraction,
        p1[1] + (p2[1] - p1[1]) * fraction
    ]

def extend_point(start_p, end_p, dist_m):
    """Return a new point extended 'dist_m' away from end_p, along the vector start_p->end_p."""
    lat_scale, lon_scale = get_lat_lon_scales(end_p[1])
    
    # Vector in degrees
    vec_x = end_p[0] - start_p[0]
    vec_y = end_p[1] - start_p[1]
    
    # Normalize vector length roughly to meters
    len_deg = math.sqrt((vec_x * lon_scale)**2 + (vec_y * lat_scale)**2)
    
    if len_deg == 0: return end_p

    # Scale vector to extend distance
    scale_factor = dist_m / len_deg
    
    new_lon = end_p[0] + (vec_x * lon_scale * scale_factor) / lon_scale
    new_lat = end_p[1] + (vec_y * lat_scale * scale_factor) / lat_scale
    
    return [new_lon, new_lat]

def generate_aisle_topology(vine_rows_geojson, node_spacing_m=2.0, extend_distance_m=3.0):
    """
    Calculates midlines between sorted vine rows, populates them with nodes,
    and connects them.
    """
    rows = [f for f in vine_rows_geojson.get("features", []) if f["geometry"]["type"] == "LineString"]
    
    # Ensure rows are sorted spatially (West to East)
    def get_row_centroid(feature):
        coords = feature["geometry"]["coordinates"]
        return sum(c[0] for c in coords) / len(coords)
    rows.sort(key=get_row_centroid)

    aisles = []
    nodes = []
    edges = []
    
    print(f"   > Processing {len(rows)} rows to create {len(rows)-1} aisles...")

    # 1. Create Aisle Geometries (Midlines)
    for i in range(len(rows) - 1):
        row_left = rows[i]["geometry"]["coordinates"]
        row_right = rows[i+1]["geometry"]["coordinates"]
        
        # Check directionality: Ensure row_right runs in the same direction as row_left
        dist_start_start = distance_m(row_left[0], row_right[0])
        dist_start_end = distance_m(row_left[0], row_right[-1])
        
        # If the start of left is closer to the end of right, flip right for calculation
        if dist_start_end < dist_start_start:
            row_right = row_right[::-1]

        # Generate midline points
        # We sample both lines at regular intervals to average them
        num_samples = max(len(row_left), len(row_right), 10)
        midline_coords = []
        
        for k in range(num_samples):
            frac = k / (num_samples - 1)
            # Get point on left row
            idx_l = frac * (len(row_left) - 1)
            p_l_base = row_left[int(idx_l)]
            p_l_next = row_left[min(int(idx_l) + 1, len(row_left) - 1)]
            p_left = interpolate_line(p_l_base, p_l_next, idx_l % 1)
            
            # Get point on right row
            idx_r = frac * (len(row_right) - 1)
            p_r_base = row_right[int(idx_r)]
            p_r_next = row_right[min(int(idx_r) + 1, len(row_right) - 1)]
            p_right = interpolate_line(p_r_base, p_r_next, idx_r % 1)
            
            # Average
            mid_pt = [(p_left[0] + p_right[0])/2, (p_left[1] + p_right[1])/2]
            midline_coords.append(mid_pt)

        # Extend the midline
        if len(midline_coords) >= 2:
            start_ext = extend_point(midline_coords[1], midline_coords[0], extend_distance_m)
            end_ext = extend_point(midline_coords[-2], midline_coords[-1], extend_distance_m)
            midline_coords.insert(0, start_ext)
            midline_coords.append(end_ext)

        aisles.append(midline_coords)

    # 2. Generate Nodes along Aisles
    aisle_node_ids = [] # To keep track for cross-connections
    
    for aisle_idx, coords in enumerate(aisles):
        current_aisle_ids = []
        total_len = 0
        segment_lengths = []
        
        # Calculate total length of aisle
        for k in range(len(coords) - 1):
            d = distance_m(coords[k], coords[k+1])
            segment_lengths.append(d)
            total_len += d
            
        # Determine number of nodes
        num_nodes = max(2, int(total_len / node_spacing_m))
        
        # Place nodes
        for n in range(num_nodes + 1): # +1 to ensure we hit the end
            target_dist = (n / num_nodes) * total_len
            
            # Find position along polyline
            cum_dist = 0
            node_coord = coords[-1] # Default to end
            
            for k, seg_len in enumerate(segment_lengths):
                if cum_dist + seg_len >= target_dist:
                    remain = target_dist - cum_dist
                    frac = remain / seg_len if seg_len > 0 else 0
                    node_coord = interpolate_line(coords[k], coords[k+1], frac)
                    break
                cum_dist += seg_len
            
            node_id = f"aisle_{aisle_idx}_node_{n}"
            current_aisle_ids.append(node_id)
            
            nodes.append({
                "topo_map_node_id": node_id,
                "coordinates": node_coord,
                "aisle_idx": aisle_idx,
                "node_idx": n
            })
            
            # Add edge to previous node in this aisle
            if n > 0:
                edges.append({
                    "from": current_aisle_ids[-2],
                    "to": node_id,
                    "type": "linear"
                })
        
        aisle_node_ids.append(current_aisle_ids)

    # 3. Create Headland Connections (U-Turns)
    # Connect the start of Aisle i to start of Aisle i+1
    # Connect the end of Aisle i to end of Aisle i+1
    for i in range(len(aisle_node_ids) - 1):
        # Start connection
        edges.append({
            "from": aisle_node_ids[i][0],
            "to": aisle_node_ids[i+1][0],
            "type": "headland"
        })
        # End connection
        edges.append({
            "from": aisle_node_ids[i][-1],
            "to": aisle_node_ids[i+1][-1],
            "type": "headland"
        })

    return nodes, edges

# --- MAIN SCRIPT LOGIC ---

def load_geojson(filepath):
    """Load a GeoJSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_geojson(data, filepath):
    """Save a GeoJSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved: {filepath}")

def generate_debug_geojson(poles_geojson, vine_rows_geojson, node_spacing_m=2.0, 
                           extend_distance_m=3.0, cross_row_distance_m=6.0):
    """
    Generate topological map and create a combined GeoJSON for debugging.
    """
    
    # 1. Generate Topological Map using CUSTOM AISLE logic
    nodes, edges = generate_aisle_topology(
        vine_rows_geojson,
        node_spacing_m=node_spacing_m,
        extend_distance_m=extend_distance_m
    )
    
    # 2. Build Output GeoJSON
    features = []
    
    # Add Poles (Blue Points)
    for pole in poles_geojson.get("features", []):
        f = deepcopy(pole)
        f["properties"]["style"] = "blue-point"
        f["properties"]["type"] = "pole"
        features.append(f)
    
    # Add Vine Rows (Green Lines)
    rows = [f for f in vine_rows_geojson.get("features", []) if f["geometry"]["type"] == "LineString"]
    for i, row in enumerate(rows):
        f = deepcopy(row)
        f["properties"]["style"] = "green-line"
        f["properties"]["type"] = "vine_row"
        f["properties"]["row_id"] = i
        features.append(f)
        
    # Add Topo Nodes (Cyan Circles)
    node_lookup = {}
    for node in nodes:
        node_lookup[node["topo_map_node_id"]] = node["coordinates"]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": node["coordinates"]},
            "properties": {
                "type": "topo_node",
                "id": node["topo_map_node_id"],
                "aisle": node["aisle_idx"],
                "style": "cyan-circle"
            }
        })
        
    # Add Topo Edges (Orange Lines)
    for edge in edges:
        p1 = node_lookup.get(edge["from"])
        p2 = node_lookup.get(edge["to"])
        if p1 and p2:
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [p1, p2]},
                "properties": {
                    "type": "topo_edge",
                    "from": edge["from"],
                    "to": edge["to"],
                    "edge_type": edge["type"],
                    "style": "orange-dashed-line"
                }
            })

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
    }

def main():
    # If files are provided as arguments, use them. Otherwise attempt to auto-discover
    # latest saved files in the current directory matching the patterns
    if len(sys.argv) >= 3:
        poles_file = sys.argv[1]
        rows_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else "debug_topo_map_output.geojson"
    else:
        # Auto-discover latest poles_*.geojson and vine_rows_*.geojson
        cwd = os.getcwd()
        pole_candidates = sorted(glob.glob(os.path.join(cwd, "poles_*.geojson")), key=os.path.getmtime, reverse=True)
        row_candidates = sorted(glob.glob(os.path.join(cwd, "vine_rows_*.geojson")), key=os.path.getmtime, reverse=True)

        if pole_candidates and row_candidates:
            poles_file = pole_candidates[0]
            rows_file = row_candidates[0]
            output_file = "debug_topo_map_output.geojson"
            print(f"Auto-discovered poles: {poles_file}")
            print(f"Auto-discovered vine rows: {rows_file}")
        else:
            print("Usage: python debug_topo_map.py <poles.geojson> <vine_rows.geojson> [output.geojson]")
            print("\nOr drop GeoJSON files named 'poles_TIMESTAMP.geojson' and 'vine_rows_TIMESTAMP.geojson' into this folder and run without args.")
            sys.exit(1)
    
    # Validate input files
    if not os.path.exists(poles_file):
        print(f"✗ Error: Poles file not found: {poles_file}")
        sys.exit(1)
    if not os.path.exists(rows_file):
        print(f"✗ Error: Rows file not found: {rows_file}")
        sys.exit(1)
    
    print(f"📂 Loading poles from: {poles_file}")
    poles_geojson = load_geojson(poles_file)
    print(f"   ✓ Loaded {len(poles_geojson.get('features', []))} poles")
    
    print(f"📂 Loading vine rows from: {rows_file}")
    rows_geojson = load_geojson(rows_file)
    print(f"   ✓ Loaded {len(rows_geojson.get('features', []))} rows")
    
    print("\n🔧 Generating topological map...")
    # NOTE: extend_distance_m is set to 4.0 meters here as requested to extend beyond row ends
    debug_geojson = generate_debug_geojson(
        poles_geojson,
        rows_geojson,
        node_spacing_m=2.0,
        extend_distance_m=4.0,
        cross_row_distance_m=6.0
    )
    
    metadata = debug_geojson.get("metadata", {})
    print(f"   ✓ Generated {metadata.get('node_count', 0)} topological nodes")
    print(f"   ✓ Generated {metadata.get('edge_count', 0)} topological edges")
    
    print(f"\n💾 Saving debug GeoJSON to: {output_file}")
    save_geojson(debug_geojson, output_file)
    
    print(f"\n✨ Done! Open {output_file} in QGIS or similar to visualize.")


if __name__ == "__main__":
    # You can hardcode filenames here for quick runs.
    DEFAULT_POLES = "geojson/poles_2026-02-17T15-33-45.geojson"
    DEFAULT_ROWS = "geojson/vine_rows_2026-02-17T15-33-45.geojson"
    DEFAULT_OUTPUT = "debug_topo_map_output.geojson"

    if DEFAULT_POLES and DEFAULT_ROWS:
        # Override argv so main() uses these values
        sys.argv = [sys.argv[0], DEFAULT_POLES, DEFAULT_ROWS, DEFAULT_OUTPUT]

    main()