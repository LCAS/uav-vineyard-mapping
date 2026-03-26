import json
import math
import os
import numpy as np

def process_vineyard_data_with_labelled_ends(all_posts_data, end_posts_data):
    """
    Groups lidar posts into vine rows and creates LineString features using 
    pre-labelled end posts to determine the row direction and row numbers.

    This version uses a vector-based approach to robustly calculate the average
    row direction and then assigns all posts to their nearest labeled row.

    Args:
        all_posts_data (dict): The GeoJSON data for all posts.
        end_posts_data (dict): The GeoJSON data for the labelled end posts.

    Returns:
        dict: A new GeoJSON FeatureCollection with LineString and Point features.
    """
    
    # 1. Calculate the average row direction from labelled end posts
    end_posts = end_posts_data['features']
    row_endpoints = {}
    
    # Group the end posts by their row number. Ensure row_num is always a string.
    # for post in end_posts:
    #     row_num = str(post['properties']['Row'])  # Convert the row number to a string
    #     if row_num not in row_endpoints:
    #         row_endpoints[row_num] = []
    #     row_endpoints[row_num].append(post['geometry']['coordinates'])

    for post in end_posts:
        row_num = str(post['properties']['Row'])
        if row_num not in row_endpoints:
            row_endpoints[row_num] = []
        row_endpoints[row_num].append(post['geometry']['coordinates'][:2])
    
    # Calculate direction vectors for each labeled row and find the average vector
    direction_vectors = []
    for row_num, coords in row_endpoints.items():
        if len(coords) == 2:
            # Calculate the vector from the first point to the second
            vector = [coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]]
            direction_vectors.append(vector)

    if not direction_vectors:
        raise ValueError("Could not calculate average direction. No valid end-post pairs found.")
        
    avg_direction_vector = np.mean(direction_vectors, axis=0)
    
    # Normalize the average vector
    norm = np.linalg.norm(avg_direction_vector)
    if norm == 0:
        raise ValueError("Average direction vector is zero.")
    avg_direction_vector = avg_direction_vector / norm
    
    # Calculate the perpendicular vector for grouping
    # This vector is rotated 90 degrees clockwise (or counter-clockwise)
    perpendicular_vector = np.array([-avg_direction_vector[1], avg_direction_vector[0]])
    
    # 2. Map all posts to the closest labeled row
    # all_posts = [feature['geometry']['coordinates'] for feature in all_posts_data['features']]
    all_posts = [feature['geometry']['coordinates'][:2] for feature in all_posts_data['features']]
    
    # Calculate the projection for each known row's center point
    row_projections = {}
    for row_num, coords in row_endpoints.items():
        if len(coords) == 2:
            center_lon = (coords[0][0] + coords[1][0]) / 2
            center_lat = (coords[0][1] + coords[1][1]) / 2
            proj = np.dot(np.array([center_lon, center_lat]), perpendicular_vector)
            row_projections[proj] = row_num

    # Now, group all posts by finding the closest reference row projection
    posts_grouped_by_row = {row_num: [] for row_num in row_endpoints.keys()}
    
    for post_coord in all_posts:
        # post_proj = np.dot(np.array(post_coord), perpendicular_vector)
        post_proj = np.dot(np.array(post_coord[:2]), perpendicular_vector)
        
        # Find the closest known row projection by minimizing the absolute difference
        closest_proj = min(row_projections.keys(), key=lambda p: abs(p - post_proj))
        row_num = row_projections[closest_proj]
        
        posts_grouped_by_row[row_num].append(post_coord)

    # 3. Sort posts within rows and create final output features
    sorted_row_numbers = sorted(posts_grouped_by_row.keys())
    output_features = []
    
    for row_number_str in sorted_row_numbers:
        row_points = posts_grouped_by_row[row_number_str]
        
        # Sort points within the row along the direction of the row itself.
        # row_points_sorted = sorted(row_points, key=lambda p: np.dot(np.array(p), avg_direction_vector))
        row_points_sorted = sorted(row_points, key=lambda p: np.dot(np.array(p[:2]), avg_direction_vector))
        
        line_string = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": row_points_sorted
            },
            "properties": {
                "row_number": f"{int(row_number_str):02d}"  # Use the original row number
            }
        }
        output_features.append(line_string)
        
        # Also add the original points with the correct row number property
        for point in row_points_sorted:
            output_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": point
                },
                "properties": {
                    "row_number": f"{int(row_number_str):02d}"
                }
            })
            
    output_geojson = {
        "type": "FeatureCollection",
        "features": output_features
    }
    
    return output_geojson

# Main script execution
if __name__ == "__main__":
    # Define file paths
    # all_posts_file = '../../ground_truth/jojo/jojo_lidar_posts.geojson'
    # end_posts_file = '../../ground_truth/jojo/jojo_end_posts_labelled.geojson'
    # output_file_name = '../../ground_truth/vine_rows_from_ends.geojson'
    
    all_posts_file = '../../ground_truth/coolhurst/other/south_block/coolhurst_lidar_posts_south_west_block.geojson'
    end_posts_file = '../../ground_truth/coolhurst/other/south_block/coolhurst_end_posts_south_west_block_labelled.geojson'
    output_file_name = '../../ground_truth/vine_rows_from_ends.geojson'
    
    # Check if necessary files exist
    if not os.path.exists(all_posts_file):
        print(f"Error: The file '{all_posts_file}' was not found. Please ensure it is uploaded.")
    elif not os.path.exists(end_posts_file):
        print(f"Error: The file '{end_posts_file}' was not found. Please ensure it is uploaded.")
    else:
        try:
            with open(all_posts_file, 'r') as f:
                all_posts_geojson = json.load(f)
            
            with open(end_posts_file, 'r') as f:
                end_posts_geojson = json.load(f)
            
            # Process the data using the end posts to determine the angle
            processed_data = process_vineyard_data_with_labelled_ends(all_posts_geojson, end_posts_geojson)
            
            # Save the resulting GeoJSON
            with open(output_file_name, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            print(f"The processed GeoJSON file has been saved as '{output_file_name}'.")
            
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from one of the input files. Please check their formats.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
