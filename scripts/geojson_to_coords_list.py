import json

# Load the uploaded GeoJSON file
geojson_path = "../data/Detected_PostPositions_UTM_shifted.geojson"
with open(geojson_path, "r") as f:
    geojson_data = json.load(f)

# Extract coordinates from the GeoJSON features
gps_points = []
for feature in geojson_data["features"]:
    coords = feature["geometry"]["coordinates"]
    # Assuming coordinates are in (lon, lat) and we want (lat, lon)
    gps_points.append((coords[1], coords[0]))

print(gps_points)
