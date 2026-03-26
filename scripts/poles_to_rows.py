import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from sklearn.decomposition import PCA
import json

def load_geojson_as_gdf(geojson_data):
    """Converts a GeoJSON dictionary into a GeoDataFrame."""
    return gpd.GeoDataFrame.from_features(geojson_data["features"], crs="EPSG:4326")

def connect_vine_rows(vine_rows, poles, pole_spacing):
    """
    Connects vine rows using pole data and interpolates missing poles.

    Parameters:
    - vine_rows: GeoDataFrame or GeoJSON dictionary
    - poles: GeoDataFrame or GeoJSON dictionary

    Returns:
    - GeoJSON dictionary with interpolated poles and connections
    """

    # Convert GeoJSON inputs to GeoDataFrames if necessary
    if isinstance(vine_rows, dict):
        vine_rows = load_geojson_as_gdf(vine_rows)
    if isinstance(poles, dict):
        poles = load_geojson_as_gdf(poles)

    degree_to_meter_latitude = 111000  # 1 degree of latitude = ~111,000 meters
    pole_spacing_degrees = (pole_spacing / degree_to_meter_latitude) * 1.5  # x1.5 fudge factor

    # Ensure CRS matches
    if vine_rows.crs != poles.crs:
        poles = poles.to_crs(vine_rows.crs)

    # Assign each pole to the nearest vine row
    def find_nearest_vine_row(pole):
        containing_rows = vine_rows[vine_rows.geometry.contains(pole)]
        if not containing_rows.empty:
            return containing_rows.index[0]
        else:
            distances = vine_rows.geometry.distance(pole)
            return distances.idxmin()

    poles["vine_row_id"] = poles.geometry.apply(find_nearest_vine_row)
    
    new_poles = []
    connection_lines = []
    
    for row_id in poles["vine_row_id"].unique():
        row_poles = poles[poles["vine_row_id"] == row_id]
        
        if len(row_poles) > 1:
            coords = np.array([(point.x, point.y) for point in row_poles.geometry])
            
            pca = PCA(n_components=1)
            transformed = pca.fit_transform(coords)
            
            row_poles["sort_order"] = transformed.flatten()
            row_poles = row_poles.sort_values(by="sort_order")
            
            line = LineString(row_poles.geometry.tolist())
            connection_lines.append({"geometry": line, "type": "connection_line", "vine_row_id": row_id})

            sorted_points = list(row_poles.geometry)
            interpolated_points = []
            
            for i in range(len(sorted_points) - 1):
                p1, p2 = sorted_points[i], sorted_points[i + 1]
                dist = p1.distance(p2)

                if dist > pole_spacing_degrees:
                    num_missing = int(dist // pole_spacing_degrees)
                    for j in range(1, num_missing + 1):
                        new_x = p1.x + (p2.x - p1.x) * (j / (num_missing + 1))
                        new_y = p1.y + (p2.y - p1.y) * (j / (num_missing + 1))
                        interpolated_point = Point(new_x, new_y)
                        interpolated_points.append({"geometry": interpolated_point, "vine_row_id": row_id})

            new_poles.extend(interpolated_points)

    geometries = [item['geometry'] for item in new_poles]
    vine_row_ids = [item['vine_row_id'] for item in new_poles]

    new_poles_gdf = gpd.GeoDataFrame(geometry=geometries, crs=vine_rows.crs)
    new_poles_gdf["vine_row_id"] = vine_row_ids
    new_poles_gdf["type"] = "interpolated_pole"

    lines_gdf = gpd.GeoDataFrame(connection_lines, geometry="geometry", crs=vine_rows.crs)

    vine_rows["type"] = "vine_row"
    poles["type"] = "pole"

    # final_gdf = gpd.GeoDataFrame(pd.concat([vine_rows, poles, new_poles_gdf, lines_gdf], ignore_index=True), crs=vine_rows.crs)
    final_gdf = gpd.GeoDataFrame(pd.concat([poles, new_poles_gdf, lines_gdf], ignore_index=True), crs=vine_rows.crs)
    
    # Convert to GeoJSON format
    geojson_data = final_gdf.to_json()

    return json.loads(geojson_data)

if __name__ == "__main__":
    # Load GeoJSON files as dictionaries
    with open("../data/detected_merged_vine_rows.geojson") as f:
        vine_rows = json.load(f)
    with open("../data/clustered_poles.geojson") as f:
        poles = json.load(f)

    pole_spacing = 5.65  # Pole spacing in meters

    connected_rows_geojson_data = connect_vine_rows(vine_rows, poles, pole_spacing)

    geojson_output = "../data/connected_poles_with_interpolation.geojson"
    with open(geojson_output, "w") as json_file:
        json.dump(connected_rows_geojson_data, json_file, indent=4)

    print("Data saved to ../data/connected_poles_with_interpolation.geojson")