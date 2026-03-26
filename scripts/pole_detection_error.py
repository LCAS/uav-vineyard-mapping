import json
from geopy.distance import geodesic
import math

# Load detected pole coordinates (YOLO detected poles)
with open('../data/detected_clustered_pole_coordinates.geojson') as f:
# with open('../data/clustering_output/clustering_results_20250320_162033_chebyshev/clustered_poles.geojson') as f:
    detected_poles = json.load(f)

# Load RTK GPS pole coordinates
with open('../data/riseholme_rtk_gps_poles.geojson') as f:
# with open('../data/jojo_rtk_gps_poles.geojson') as f:
    rtk_poles = json.load(f)

# Function to calculate the closest RTK pole for each detected pole
def calculate_closest_pole_error(detected_poles, rtk_poles):
    errors = []
    squared_errors = []
    absolute_errors = [] #added absolute errors
    for detected_pole in detected_poles['features']:
        detected_coords = detected_pole['geometry']['coordinates']

        closest_error = float('inf')
        closest_pole = None

        for rtk_pole in rtk_poles['features']:
            rtk_coords = rtk_pole['geometry']['coordinates']
            distance = geodesic(detected_coords[::-1], rtk_coords[::-1]).meters

            if distance < closest_error:
                closest_error = distance
                closest_pole = rtk_pole

        errors.append({
            'detected_pole': detected_pole['geometry']['coordinates'],
            'closest_pole': closest_pole['geometry']['coordinates'],
            'error_in_meters': closest_error
        })
        squared_errors.append(closest_error**2)
        absolute_errors.append(abs(closest_error))
    return errors, squared_errors, absolute_errors

# Function to calculate the closest detected pole for each RTK pole
def calculate_closest_detected_pole_error(rtk_poles, detected_poles):
    errors = []
    squared_errors = []
    absolute_errors = [] #added absolute errors
    for rtk_pole in rtk_poles['features']:
        rtk_coords = rtk_pole['geometry']['coordinates']

        closest_error = float('inf')
        closest_pole = None

        for detected_pole in detected_poles['features']:
            detected_coords = detected_pole['geometry']['coordinates']
            distance = geodesic(rtk_coords[::-1], detected_coords[::-1]).meters

            if distance < closest_error:
                closest_error = distance
                closest_pole = detected_pole

        errors.append({
            'rtk_pole': rtk_pole['geometry']['coordinates'],
            'closest_detected_pole': closest_pole['geometry']['coordinates'],
            'error_in_meters': closest_error
        })
        squared_errors.append(closest_error**2)
        absolute_errors.append(abs(closest_error))
    return errors, squared_errors, absolute_errors

# # Call the function and get the error data
# errors, squared_errors, absolute_errors = calculate_closest_pole_error(detected_poles, rtk_poles)

# # Output the results and calculate the average error
# total_error = 0
# for error in errors:
#     print(f"Detected pole at {error['detected_pole']} has an error of {error['error_in_meters']} meters.")
#     print(f"Closest RTK pole is at {error['closest_pole']}")
#     print("-" * 50)
#     total_error += error['error_in_meters']

# # Calculate and print the average error (MAE)
# average_error = sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0
# print(f"Average error (MAE): {average_error} meters")

# # Calculate RMSE
# rmse = math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else 0
# print(f"RMSE: {rmse} meters")

# print(f"Number of RTK GPS poles: {len(rtk_poles['features'])}")
# print(f"Number of Detected poles: {len(detected_poles['features'])}")


# Call the function and get the error data (note the reversed order of arguments)
errors, squared_errors, absolute_errors = calculate_closest_detected_pole_error(rtk_poles, detected_poles)

# Output the results and calculate the average error
total_error = 0
for error in errors:
    print(f"RTK pole at {error['rtk_pole']} has an error of {error['error_in_meters']} meters.")
    print(f"Closest detected pole is at {error['closest_detected_pole']}")
    print("-" * 50)
    total_error += error['error_in_meters']

# Calculate and print the average error (MAE)
average_error = sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0
print(f"Average error (MAE): {average_error} meters")

# Calculate RMSE
rmse = math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else 0
print(f"RMSE: {rmse} meters")

print(f"Number of RTK GPS poles: {len(rtk_poles['features'])}")
print(f"Number of Detected poles: {len(detected_poles['features'])}")