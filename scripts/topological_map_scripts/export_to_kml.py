import geojson
import simplekml

def create_kml_from_geojson(geojson_data):
    """Creates a KML string from a GeoJSON FeatureCollection containing points and lines."""

    kml = simplekml.Kml()

    for feature in geojson_data['features']:
        if feature['geometry']['type'] == 'Point':
            coords = feature['geometry']['coordinates']
            kml.newpoint(name=feature['properties']['topo_map_node_id'], coords=[(coords[0], coords[1])])
        elif feature['geometry']['type'] == 'LineString':
            coords = feature['geometry']['coordinates']
            linestring = kml.newlinestring(name=feature['properties']['topo_map_edge_id'])
            linestring.coords = [(coord[0], coord[1], 0.0) for coord in coords]  # Add altitude (0.0)
            linestring.altitudemode = simplekml.AltitudeMode.clamptoground

    return kml.kml()

if __name__ == "__main__":  # The important addition
    try:
        with open("../../data/topological_map.geojson", "r") as f:  # Replace with your GeoJSON file
            geojson_data = geojson.load(f)

        kml_string = create_kml_from_geojson(geojson_data)

        with open("../../data/topological_map.kml", "w") as outfile:
            outfile.write(kml_string)

        print("KML file created: topological_map.kml")

    except FileNotFoundError:
        print("Error: GeoJSON file not found. Please provide the correct file path.")
    except Exception as e:
        print(f"An error occurred: {e}")