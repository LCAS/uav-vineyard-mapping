import geojson
import numpy as np

def create_mid_row_lines(geojson_data):
    # ... (function code remains exactly the same as in the previous response)
    vine_rows = {}
    for feature in geojson_data['features']:
        if feature['geometry']['type'] == 'LineString':
            vine_row_id = feature['properties']['vine_row_id']
            vine_rows[vine_row_id] = feature['geometry']['coordinates']

    if not vine_rows:
        return geojson.FeatureCollection([])

    # 1. Calculate average x-coordinate for each vine row
    row_x_coords = {}
    for row_id, coords in vine_rows.items():
        x_coords = [coord[0] for coord in coords]  # Extract all x-coordinates
        avg_x = np.mean(x_coords)
        row_x_coords[row_id] = avg_x

    # 2. Sort vine row IDs based on average x-coordinate
    sorted_row_ids = sorted(vine_rows.keys(), key=lambda row_id: row_x_coords[row_id])

    mid_row_lines = []

    for i in range(len(sorted_row_ids) - 1):
        id1 = sorted_row_ids[i]
        id2 = sorted_row_ids[i + 1]
        coords1 = vine_rows[id1]
        coords2 = vine_rows[id2]

        # Calculate midpoints of start and end points
        start1 = np.array(coords1[0])
        end1 = np.array(coords1[-1])
        start2 = np.array(coords2[0])
        end2 = np.array(coords2[-1])

        mid_start = (start1 + start2) / 2
        mid_end = (end1 + end2) / 2

        mid_row_lines.append(geojson.Feature(
            geometry=geojson.LineString([mid_start.tolist(), mid_end.tolist()]),
            properties={"type": "mid_row_line", "vine_row_ids": [id1, id2]}
        ))

    return geojson.FeatureCollection(mid_row_lines)


if __name__ == "__main__":  # This is the important addition
    try:
        with open("../data/vineyard_poles_and_rows.geojson", "r") as f: # replace with your file
            data = geojson.load(f)

        mid_row_geojson = create_mid_row_lines(data)

        # combined_features = data['features'] + mid_row_geojson['features']
        # combined_geojson = geojson.FeatureCollection(combined_features)

        with open("../data/mid_row_lines.geojson", "w") as outfile:
            geojson.dump(mid_row_geojson, outfile, indent=2)

        print("Combined GeoJSON saved to ../data/mid_row_lines.geojson")

    except FileNotFoundError:
        print("Error: Input GeoJSON file not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")