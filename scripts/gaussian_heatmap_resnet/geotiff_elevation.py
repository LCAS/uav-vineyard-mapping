import rasterio
from rasterio.sample import sample_gen
import sys

# --- CONFIGURATION ---
# 1. SET THE PATH TO YOUR *NEW* DSM FILE
GEOTIFF_DSM_PATH = "../../images/riseholme/august_2024/dem/39_feet/dsm.tif"

# 2. SET A TEST COORDINATE (find a [Lon, Lat] from your *new* GeoJSON)
TEST_LON_LAT = (-0.12345, 51.54321)
# ---------------------


def test_elevation_lookup(geotiff_file, lon, lat):
    """
    Opens a GeoTIFF file and samples the values at a specific Lon/Lat.
    """
    print(f"--- Testing DSM File ---")
    print(f"File: {geotiff_file}")
    print(f"Test Coordinate (Lon, Lat): ({lon}, {lat})\n")

    try:
        with rasterio.open(geotiff_file) as dataset:
            
            # --- 1. Print Debug Information ---
            print("--- GeoTIFF Metadata ---")
            print(f"CRS: {dataset.crs}")
            print(f"Bounds: {dataset.bounds}")
            print(f"Number of Bands: {dataset.count}")
            print(f"Data Type (Band 1): {dataset.dtypes[0]}")
            nodata_val = dataset.nodata
            print(f"'No Data' Value: {nodata_val}\n")

            if dataset.count > 1:
                print("WARNING: This file has more than 1 band. It might be an orthophoto (RGB).")
                print("A DEM/DSM should have only 1 band (elevation).")

            if 'uint8' in str(dataset.dtypes[0]):
                 print("WARNING: Data type is 'uint8'. This usually means color (0-255), not altitude.")
                 print("A DEM/DSM usually has a 'float' or 'int' data type.")
            
            print("\n--- Sampling (Band 1) ---")
            
            # Sample the file at the given Lon/Lat
            # We tell sample_gen that our input coordinate is in WGS84 (Lat/Lon)
            # and it will handle the reprojection to the file's CRS.
            try:
                sampled_values = list(sample_gen(dataset, [(lon, lat)], indexes=1, crs='EPSG:4326'))
            except Exception as e:
                if "unexpected keyword argument 'crs'" in str(e):
                    print("\n---!! ERROR !!---")
                    print("Your version of rasterio is too old. We will try the manual workaround.")
                    print("Please see the *next* file (generate_masks_with_dsm.py) which includes this workaround.")
                    return
                else:
                    raise e

            if not sampled_values:
                print("Error: The sampling generator returned no data.")
                return

            value = sampled_values[0][0] # Get value from first band

            # --- 4. Print the Result ---
            print("\n--- Result ---")
            if nodata_val is not None and value == nodata_val:
                print(f"Result: 'NO DATA' value ({nodata_val}) found at this coordinate.")
                print("This means your test coordinate is outside the file's boundary.")
            else:
                print(f"SUCCESS: Found elevation at ({lon}, {lat})")
                print(f"Elevation (Band 1): {value}")
                
                if 'float' in str(dataset.dtypes[0]) or 'int' in str(dataset.dtypes[0]):
                     print("\nANALYSIS: This looks like a valid DEM/DSM file. You can proceed!")
                
    except FileNotFoundError:
        print(f"Error: File not found at '{geotiff_file}'")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    if GEOTIFF_DSM_PATH == "/path/to/your/new_vineyard_DSM.tif":
        print("Error: Please edit 'test_dsm.py' and set the GEOTIFF_DSM_PATH variable.")
    else:
        test_elevation_lookup(GEOTIFF_DSM_PATH, TEST_LON_LAT[0], TEST_LON_LAT[1])
