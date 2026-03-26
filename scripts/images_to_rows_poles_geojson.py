import json
import geojson
import io
import yaml  # pip install pyyaml
import roboflow_pole_detection_folder
import pole_clustering
import yolov9_rows_folder
import poles_to_rows
import mid_row_lines
import generate_topological_map
import topological_map_scripts.export_to_kml
import topological_map_scripts.export_to_topological_map
import topological_map_scripts.kml_to_tmap

vineyard_name = "Riseholme"

image_folder="../images/riseholme/august_2024/39_feet/"
# image_folder="../images/jojo/agri_tech_centre/RX1RII/"
output_folder_poles="../images/output/"
model_id_roboflow="vineyard_test/4"
# model_id_roboflow="vineyard_segmentation/7"

# Load the API key
with open("../config/api_key.json", 'r') as file:
    config = json.load(file)
ROBOFLOW_API_KEY = config.get("ROBOFLOW_API_KEY")

sensor_width_mm=11.04 # Riseholme DJI UAV sensor width in mm
# sensor_width_mm=35.8 # Agri Tech Centre JoJo sensor width in mm Sony DSC-RX1RM2

model_path_yolo_rows = "../weights/vine_row_segmentation/best.pt"
output_folder_rows = "../images/output/row_detection/"

pole_spacing = 5.65  # Pole spacing in meters

# 1. Run pole detection on a folder of images
print("1. Running pole detection on a folder of images...")
pole_unfiltered_geojson_data = roboflow_pole_detection_folder.detect_poles(
        image_folder=image_folder,
        output_folder=output_folder_poles,
        ROBOFLOW_API_KEY=ROBOFLOW_API_KEY,
        model_id=model_id_roboflow,
        sensor_width_mm=sensor_width_mm
    )

# 2. Process clustering of poles
print("2. Processing clustering of poles...")
poles_clustered_geojson = pole_clustering.cluster_poles(pole_unfiltered_geojson_data)

# 3 Run vine row detection on a folder of images
print("3. Running vine row detection on a folder of images...")
vine_row_geojson_data = yolov9_rows_folder.process_vine_row_segmentation(
        image_folder=image_folder, 
        output_folder=output_folder_rows, 
        model_path=model_path_yolo_rows, 
        sensor_width_mm=sensor_width_mm
    )
    
# 4 find what rows poles belong to and spatial interpolation to add missing poles
print("4. Connecting poles to rows...")
connected_rows_geojson_data = poles_to_rows.connect_vine_rows(
        vine_rows=vine_row_geojson_data, 
        poles=poles_clustered_geojson, 
        pole_spacing=pole_spacing
    )

# geojson_output = "../data/vineyard_poles_and_rows.geojson"
# with open(geojson_output, "w") as json_file:
#     json.dump(connected_rows_geojson_data, json_file, indent=4)
# print("Data saved to ../data/vineyard_poles_and_rows.geojson")

# 5 generate topological navigation map
print("5. Generating topological navigation map...")
mid_row_geojson = mid_row_lines.create_mid_row_lines(connected_rows_geojson_data)

distance_to_extend = 3  # meters #  how far the mid-row lines should be extended beyond their original start and end points.
node_spacing_along_row = 3  # metersf
node_spacing_row_initial_offset = 3  # meters
node_spacing_row_last_offset = 3  # meters
row_width = 6  # meters # width of the row it's used when connecting the extended points (the very ends of the extended lines) to nearby points.

topological_map_geojson = generate_topological_map.extend_line_points(mid_row_geojson, distance_to_extend, node_spacing_along_row, node_spacing_row_initial_offset, node_spacing_row_last_offset, row_width)

combined_features = topological_map_geojson['features'] + connected_rows_geojson_data['features']
combined_geojson = geojson.FeatureCollection(combined_features)

geojson_output = "../data/topological_map_with_poles_rows.geojson"
with open(geojson_output, "w") as json_file:
    json.dump(combined_geojson, json_file, indent=4)
print("Data saved to ../data/topological_map_with_poles_rows.geojson")

# 6 save topological map to yaml and datum to yaml
print("6. Saving topological map to yaml and datum to yaml...")
kml_file = topological_map_scripts.export_to_kml.create_kml_from_geojson(topological_map_geojson)

kml_bytes = kml_file.encode('utf-8')

# Creating BytesIO object to store the bytes
kml_bytes_io = io.BytesIO()
kml_bytes_io.write(kml_bytes)
kml_bytes_io.seek(0)

center_coordinates = topological_map_scripts.export_to_topological_map.find_centre_from_geojson(topological_map_geojson)

datum = {'datum_latitude': center_coordinates[1], 'datum_longitude': center_coordinates[0]}
tmap_yaml_file = topological_map_scripts.kml_to_tmap.run({'src': kml_bytes_io, 'datum': datum, 'location_name':vineyard_name, 'line_col':'ff2f2fd3', 'line_width':'4', 'fill_col':'c02f2fd3', 'shape_size':0.000005})                
tmap_yaml_bytes = io.BytesIO(tmap_yaml_file.encode('utf-8'))

# Save YAML file to disk
with open("../data/topological_map.tmap2.yaml", "w") as outfile:
    yaml.dump(yaml.safe_load(tmap_yaml_file), outfile, default_flow_style=False)

print("YAML file saved to topological_map.tmap2.yaml")

# Save topoligcal map datum file
yaml_datum = topological_map_scripts.export_to_topological_map.generate_datum_yaml((center_coordinates[1], center_coordinates[0]))
                
# Convert YAML content to bytes
yaml_datum_bytes = io.BytesIO(yaml_datum.encode('utf-8'))

# Save YAML file to disk
with open("../data/topological_map_datum.yaml", "w") as outfile:
    yaml.dump(yaml.safe_load(yaml_datum_bytes), outfile, default_flow_style=False)