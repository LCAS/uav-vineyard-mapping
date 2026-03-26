import rasterio
import numpy as np

# Path to your DEM
dem_path = "../../images/agri_tech_centre/dsm/DSM_JOJO's.tif"

def inspect_dem_values(path):
    try:
        with rasterio.open(path) as src:
            print(f"--- DEM INSPECTION: {path} ---")
            print(f"CRS: {src.crs.to_string()}")
            print(f"Bounds: {src.bounds}")
            
            # 1. Read the center pixel
            # We calculate the center x/y in the map's coordinate system
            center_x = (src.bounds.left + src.bounds.right) / 2
            center_y = (src.bounds.top + src.bounds.bottom) / 2
            
            # Sample the value
            # src.sample expects a list of (x,y) tuples
            val_gen = src.sample([(center_x, center_y)])
            val = next(val_gen)[0]
            
            # Check for "No Data" (empty space)
            nodata = src.nodata
            if nodata is not None and val == nodata:
                print(f"\n[!] Center pixel is 'No Data' ({nodata}).")
                print("    Trying to find a valid pixel...")
                # Read a small window in the middle to find a real number
                window = rasterio.windows.Window(src.width//2, src.height//2, 10, 10)
                data = src.read(1, window=window)
                valid_data = data[data != nodata]
                if len(valid_data) > 0:
                    val = valid_data[0]
                else:
                    print("    Could not find valid data in the center.")
                    return

            print(f"\n>>> SAMPLED ELEVATION: {val:.2f} meters <<<")
            
            # 2. Compare with Drone Logic
            # Based on your previous ExifTool dump:
            drone_gps_abs = 264.9  # Ellipsoid
            drone_rel = 24.6
            implied_ground_ellipsoid = drone_gps_abs - drone_rel # ~240.3m
            
            print("\n--- DIAGNOSIS ---")
            diff = abs(val - implied_ground_ellipsoid)
            
            if diff < 10:
                print(f"Result: MATCH (Diff: {diff:.1f}m)")
                print("Your DEM is likely ELLIPSOIDAL (WGS84).")
                print("Set AUTO_ALIGN_ALTITUDE = False (or True, it won't hurt).")
            elif 40 < diff < 60:
                print(f"Result: MISMATCH (Diff: {diff:.1f}m)")
                print("Your DEM is likely GEOID (Mean Sea Level).")
                print("Your Drone is ~50m 'higher' than the map thinks.")
                print("You MUST set AUTO_ALIGN_ALTITUDE = True.")
            else:
                print(f"Result: UNCLEAR (Diff: {diff:.1f}m)")
                print("The difference is unusual. AUTO_ALIGN_ALTITUDE = True is recommended.")

    except Exception as e:
        print(f"Error: {e}")

inspect_dem_values(dem_path)