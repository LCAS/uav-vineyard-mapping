'''
This script processes GeoJSON data of vineyard pole locaions and connect them to form rows in stright lines.
Stright lines are formed form the fisrt pole to the last pole in each row, the intermedite poles are snapped to the rows at the spacing distance.

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

def geojson_poles_spatial_interpolation(input_file, output_file, poles_per_row, compass_heading):
    """
    Processes GeoJSON data of vineyard pole locations and connects them to form rows in straight lines.
    
    Args:
        input_file (str): Path to the input GeoJSON file.
        output_file (str): Path to save the updated GeoJSON file.
        poles_per_row (list): Number of poles expected per row.
        compass_heading (float): Compass heading of rows (degrees clockwise from north, north == 0).
    
    Returns:
        None
    """
    
    # Function to calculate a point along a line at a specific distance
    def interpolate_along_line(start, end, distance):
        line_vector = np.array(end) - np.array(start)
        line_length = np.linalg.norm(line_vector)
        unit_vector = line_vector / line_length
        return np.array(start) + unit_vector * distance

    # Load the GeoJSON file
    with open(input_file, 'r') as f:
        geojson_data = json.load(f)

    # Extract coordinates
    coordinates = np.array([feature['geometry']['coordinates'] for feature in geojson_data['features']])

    # Convert compass heading to mathematical heading
    math_heading = (360 - compass_heading) % 360
    heading_radians = math.radians(math_heading)
    direction_vector = np.array([math.cos(heading_radians), math.sin(heading_radians)])

    # Project coordinates onto the row direction for sorting
    projections = np.dot(coordinates, direction_vector)
    coordinates_sorted = coordinates[np.argsort(projections)]

    # Divide into rows and interpolate missing poles
    row_features = []
    pole_features = []
    start_idx = 0
    max_pole_id = 0  # Track the maximum pole_id
    for row_idx, expected_count in enumerate(poles_per_row):
        row_coords = coordinates_sorted[start_idx:start_idx + expected_count]

        if len(row_coords) < 2:
            print(f"Row {row_idx + 1} has insufficient poles for interpolation. Skipping.")
            continue

        # Interpolate missing poles along the line between the first and last pole
        first_pole, last_pole = row_coords[0], row_coords[-1]
        line_length = np.linalg.norm(last_pole - first_pole)
        avg_spacing = line_length / (expected_count - 1)
        interpolated_coords = [
            interpolate_along_line(first_pole, last_pole, i * avg_spacing)
            for i in range(expected_count)
        ]        

        # Create/update pole features
        for pole_idx, coord in enumerate(interpolated_coords):
            max_pole_id += 1
            pole_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": coord.tolist()
                },
                "properties": {
                    "type": "pole",
                    "pole_id": max_pole_id,
                    "row_id": row_idx + 1,
                    "status": "interpolated" if pole_idx >= len(row_coords) else "existing"
                }
            })

        start_idx += expected_count

    # Combine rows and poles into a single GeoJSON
    updated_geojson = {
        "type": "FeatureCollection",
        "features": pole_features
    }

    # Save the updated GeoJSON
    with open(output_file, 'w') as f:
        json.dump(updated_geojson, f, indent=2)

    print(f"GeoJSON with interpolated poles saved as '{output_file}'")


if __name__ == "__main__":
    # Example input parameters
    input_file = '../../data/clustered_poles.geojson'
    output_file = '../../data/spatial_interpolation_poles_stright.geojson'
    poles_per_row = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]  # Adjust based on your data
    compass_heading = 170  # Adjust based on your data

    geojson_poles_spatial_interpolation(input_file, output_file, poles_per_row, compass_heading)
