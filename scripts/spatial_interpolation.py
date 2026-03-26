'''
This script processes GeoJSON data of vineyard pole locaions and connect them to form rows.

1. Loads GeoJSON Data: Reads existing pole data.
2. Defines Row Structures: Uses prior knowledge; compass heading and poles per row.
3. Handles Missing Poles: Identifies gaps in rows and interpolates missing poles.
4. Assigns Row IDs: Maps poles to the nearest row using perpendicular distance.
5. Creates Rows as LineStrings: Represents rows as GeoJSON LineString features.
6. Updates GeoJSON: Combines pole and row features into a single updated GeoJSON file.
7. The final result is saved as updated_poles_and_rows.geojson, including interpolated poles and row assignments.
'''

import json
import numpy as np
import math

# Define prior knowledge
# Riseholme
poles_per_row = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4] # poles per row
compass_heading = 170  # compass heading of rows (degrees clockwise from north, north == 0, east == 90, south == 180, west == 270)

# Load the GeoJSON file (Riseholme)
with open('../data/clustered_poles.geojson', 'r') as f:
    geojson_data = json.load(f)

# JoJos first 10 rows from the west end
# poles_per_row = [5, 7, 8, 9, 10, 11, 13, 14, 15, 16] # poles per row
# compass_heading = 65  # compass heading of rows (degrees clockwise from north, north == 0, east == 90, south == 180, west == 270)

# Load the GeoJSON file (JoJo)
# with open('../data/jojo_row_posts_10_rows.geojson', 'r') as f:
#     geojson_data = json.load(f)

# Function to calculate the perpendicular distance from a point to a line segment
def perpendicular_distance(point, line_start, line_end):
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len = np.linalg.norm(line_vec)
    
    if line_len == 0:
        return np.linalg.norm(point_vec)  # If the line segment is just a point, return distance to the point
    
    # Project the point onto the line segment
    projection = np.dot(point_vec, line_vec) / line_len
    projection = np.clip(projection, 0, line_len)
    closest_point = line_start + projection * (line_vec / line_len)
    
    # Calculate the distance from the point to the closest point on the line
    return np.linalg.norm(np.array(point) - closest_point)

# Extract coordinates
coordinates = np.array([feature['geometry']['coordinates'] for feature in geojson_data['features']])

# Convert to mathematical heading (counter-clockwise from 0 deg North)
math_heading = (360 - compass_heading) % 360

# Convert heading to radians
heading_radians = math.radians(math_heading)

# Unit vector for the heading direction
direction_vector = np.array([math.cos(heading_radians), math.sin(heading_radians)])

# Project coordinates onto the heading direction
projections = np.dot(coordinates, direction_vector)

# Sort coordinates by their projections
coordinates_sorted = coordinates[np.argsort(projections)]

# Divide sorted coordinates into rows
row_assignments = []
start_idx = 0
missing_poles = []

for row_idx, expected_count in enumerate(poles_per_row):
    # Extract row coordinates
    row_coords = coordinates_sorted[start_idx:start_idx + expected_count]
    
    # Calculate the average pole spacing if there are enough points
    if len(row_coords) > 1:
        distances = np.linalg.norm(np.diff(row_coords, axis=0), axis=1)
        avg_spacing = np.mean(distances)
    else:
        avg_spacing = 1.0  # Default spacing if only one or no pole exists

    # Add missing poles
    while len(row_coords) < expected_count:
        # Find the largest gap between existing poles
        distances = np.linalg.norm(np.diff(row_coords, axis=0), axis=1)
        max_gap_idx = np.argmax(distances)
        gap_start = row_coords[max_gap_idx]
        gap_end = row_coords[max_gap_idx + 1]

        # Add a pole in the middle of the gap
        new_pole = (np.array(gap_start) + np.array(gap_end)) / 2
        row_coords = np.insert(row_coords, max_gap_idx + 1, new_pole, axis=0)
        missing_poles.append(new_pole)

    # Store row data
    row_assignments.append({
        "row_id": row_idx + 1,
        "coordinates": row_coords.tolist()
    })
    start_idx += expected_count

# Create GeoJSON features for rows as LineStrings
row_features = []
for row in row_assignments:
    row_features.append({
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": row['coordinates']
        },
        "properties": {
            "type": "row",
            "row_id": row['row_id']
        }
    })

# Add row_id to individual poles
pole_features = []
for feature, coord in zip(geojson_data['features'], coordinates_sorted):
    for row in row_assignments:
        if coord.tolist() in row['coordinates']:
            feature['properties']['row_id'] = row['row_id']
            pole_features.append(feature)
            break

# Find the max pole_id from existing poles
existing_poles = [feature for feature in geojson_data['features'] if feature['geometry']['type'] == 'Point']
max_pole_id = max(int(feature['properties'].get('pole_id', 0)) for feature in existing_poles) if existing_poles else 0

# Add missing poles to the pole features
for missing_pole in missing_poles:
    new_pole_id =+ max_pole_id + 1  # Increment the pole_id
    pole_features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": missing_pole.tolist()
        },
        "properties": {
            "type": "pole",
            "pole_id": new_pole_id,
            "row_id": None,
            "status": "interpolated"
        }
    })

# Combine both poles and rows into a single GeoJSON
updated_geojson = {
    "type": "FeatureCollection",
    "features": pole_features + row_features
}

# Now assign row IDs to the poles based on the closest row line
for feature in updated_geojson['features']:
    if feature['geometry']['type'] == 'Point':
        pole_coordinates = feature['geometry']['coordinates']
        
        # Initialize the minimum distance and corresponding row ID
        min_distance = float('inf')
        closest_row_id = None
        
        # Find the closest row by checking the distance to each row line
        for row in row_assignments:
            row_coords = row['coordinates']
            
            # Iterate over pairs of points (line segments) in the row
            for i in range(len(row_coords) - 1):
                line_start = row_coords[i]
                line_end = row_coords[i + 1]
                
                # Calculate the perpendicular distance from the pole to the row line segment
                dist = perpendicular_distance(pole_coordinates, line_start, line_end)
                
                # Update the minimum distance and closest row ID
                if dist < min_distance:
                    min_distance = dist
                    closest_row_id = row['row_id']
                    # print(f"Closest row to pole {pole_coordinates}: {closest_row_id}")
        
        # Assign the closest row ID to the pole
        feature['properties']['row_id'] = closest_row_id

# Combine both poles and rows into a single GeoJSON
updated_geojson = {
    "type": "FeatureCollection",
    "features": pole_features + row_features
}

# Save the updated GeoJSON
with open('../data/spatial_interpolation_poles_and_rows.geojson', 'w') as f:
    json.dump(updated_geojson, f, indent=2)

print("GeoJSON with missing poles interpolated saved as 'spatial_interpolation_poles_and_rows.geojson'")
