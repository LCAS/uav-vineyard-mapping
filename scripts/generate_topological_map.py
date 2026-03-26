import geojson
import math
from geopy.distance import geodesic

# Function to calculate bearing between two points
def bearing(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2

    d_lon = lon2 - lon1

    y = math.sin(math.radians(d_lon)) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.cos(math.radians(d_lon))

    initial_bearing = math.atan2(y, x)

    return math.degrees(initial_bearing)

def interpolate_points(coordinates, spacing, initial_offset, last_offset):
    interpolated_points = []
    start_point = coordinates[0]
    end_point = coordinates[1]
    line_length = geodesic(start_point, end_point).meters

    current_distance = initial_offset
    while current_distance <= line_length - last_offset:
        fraction = current_distance / line_length
        lat = start_point[0] + (end_point[0] - start_point[0]) * fraction
        lon = start_point[1] + (end_point[1] - start_point[1]) * fraction
        interpolated_points.append([lat, lon])
        current_distance += spacing

    # Add the last point if it's not already included (handling last_offset)
    if len(interpolated_points) == 0:  # If no points were added, add the end point with the last offset
        fraction = (line_length - last_offset) / line_length
        lat = start_point[0] + (end_point[0] - start_point[0]) * fraction
        lon = start_point[1] + (end_point[1] - start_point[1]) * fraction
        interpolated_points.append([lat, lon])
    elif geodesic(interpolated_points[-1], end_point).meters > last_offset / 2:  # If the last point is not close enough to the end point add the end point
        fraction = (line_length - last_offset) / line_length
        lat = start_point[0] + (end_point[0] - start_point[0]) * fraction
        lon = start_point[1] + (end_point[1] - start_point[1]) * fraction
        interpolated_points.append([lat, lon])

    return interpolated_points


def extend_line_points(geojson_data, distance_to_extend, node_spacing_along_row, node_spacing_row_initial_offset, node_spacing_row_last_offset, row_width):
    extended_points = []
    extended_lines = []
    interpolated_nodes = []

    for feature in geojson_data['features']:
        line_id = feature['properties']['vine_row_ids'][0]  # Use the first ID as the line ID
        line_coords = feature['geometry']['coordinates']
        start_point = line_coords[0]
        end_point = line_coords[-1]

        bearing_start_end = bearing(start_point, end_point)

        new_start_point = geodesic(meters=distance_to_extend).destination(start_point, bearing_start_end + 180)
        new_end_point = geodesic(meters=distance_to_extend).destination(end_point, bearing_start_end)

        extended_points.append(geojson.Feature(geometry=geojson.Point([new_start_point.latitude, new_start_point.longitude]), properties={'topo_map_node_id': str(line_id) + '_node_start', 'type': 'Point'}))
        extended_points.append(geojson.Feature(geometry=geojson.Point([new_end_point.latitude, new_end_point.longitude]), properties={'topo_map_node_id': str(line_id) + '_node_end', 'type': 'Point'}))

        interpolated_points = interpolate_points([[new_start_point.latitude, new_start_point.longitude], [new_end_point.latitude, new_end_point.longitude]], node_spacing_along_row, node_spacing_row_initial_offset, node_spacing_row_last_offset)

        point_number = 0
        for point_coordinates in interpolated_points:
            interpolated_nodes.append(geojson.Feature(geometry=geojson.Point(point_coordinates), properties={'topo_map_node_id': str(line_id) + '_node_' + str(point_number), 'type': 'Point'}))
            point_number += 1

        for i in range(len(interpolated_points) - 1):
            extended_lines.append(geojson.Feature(geometry=geojson.LineString([interpolated_points[i], interpolated_points[i + 1]]), properties={'topo_map_edge_id': str(line_id) + '_edge_' + str(i), 'type': 'LineString'}))

        extended_lines.append(geojson.Feature(geometry=geojson.LineString([[new_start_point.latitude, new_start_point.longitude], interpolated_points[0]]), properties={'topo_map_edge_id': str(line_id) + '_start_edge', 'type': 'LineString'}))
        extended_lines.append(geojson.Feature(geometry=geojson.LineString([interpolated_points[-1], [new_end_point.latitude, new_end_point.longitude]]), properties={'topo_map_edge_id': str(line_id) + '_end_edge', 'type': 'LineString'}))

        for point in extended_points:
            if point.geometry.type == 'Point':
                point_coords = point.geometry.coordinates
                if geodesic((new_start_point.latitude, new_start_point.longitude), point_coords).meters <= row_width and (new_start_point.latitude, new_start_point.longitude) != tuple(point_coords):
                    extended_lines.append(geojson.Feature(geometry=geojson.LineString([[new_start_point.latitude, new_start_point.longitude], point_coords]), properties={'topo_map_edge_id': 'start_to_' + point.properties['topo_map_node_id'], 'type': 'LineString'}))

        for point in extended_points:
            if point.geometry.type == 'Point':
                point_coords = point.geometry.coordinates
                if geodesic((new_end_point.latitude, new_end_point.longitude), point_coords).meters <= row_width and (new_end_point.latitude, new_end_point.longitude) != tuple(point_coords):
                    extended_lines.append(geojson.Feature(geometry=geojson.LineString([[new_end_point.latitude, new_end_point.longitude], point_coords]), properties={'topo_map_edge_id': 'end_to_' + point.properties['topo_map_node_id'], 'type': 'LineString'}))

    return geojson.FeatureCollection(extended_points + extended_lines + interpolated_nodes)


if __name__ == "__main__":
    try:
        with open("../data/mid_row_lines.geojson", "r") as f:
            data = geojson.load(f)

        # Example parameters
        distance_to_extend = 3  # meters #  how far the mid-row lines should be extended beyond their original start and end points.
        node_spacing_along_row = 3  # metersf
        node_spacing_row_initial_offset = 3  # meters
        node_spacing_row_last_offset = 3  # meters
        row_width = 6  # meters # width of the row it's used when connecting the extended points (the very ends of the extended lines) to nearby points.  The code checks if any other extended points are within this row_width distance and, if so, creates a connecting line.  

        combined_geojson = extend_line_points(data, distance_to_extend, node_spacing_along_row, node_spacing_row_initial_offset, node_spacing_row_last_offset, row_width)

        with open("../data/topologiacl_map.geojson", "w") as outfile:
            geojson.dump(combined_geojson, outfile, indent=2)

        print("Combined GeoJSON saved to topologiacl_map.geojson")

    except FileNotFoundError:
        print("Error: GeoJSON file not found. Please provide the correct file path.")
    except Exception as e:
        print(f"An error occurred: {e}")