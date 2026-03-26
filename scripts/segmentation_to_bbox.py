import json
import math

def convert_segmentation_to_bbox(segmentation):
    # Convert flat list to coordinate pairs
    if isinstance(segmentation[0], float) or isinstance(segmentation[0], int):
        points = list(zip(segmentation[::2], segmentation[1::2]))
    else:
        points = segmentation

    x_coords = [math.floor(p[0]) for p in points]
    y_coords = [math.floor(p[1]) for p in points]
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]

# Load the JSON data from the file
with open('ripeness_class_annotations.json', 'r') as file:
    data = json.load(file)

# Remove all bounding box data and convert segmentation masks to bounding box data
for annotation in data['annotations']:
    if 'bbox' in annotation:
        del annotation['bbox']
    segmentation = annotation.get('segmentation')

    # Only process polygon segmentations
    if isinstance(segmentation, list):
        try:
            if isinstance(segmentation[0], list):
                annotation['bbox'] = convert_segmentation_to_bbox(segmentation[0])
            else:
                annotation['bbox'] = convert_segmentation_to_bbox(segmentation)
        except Exception as e:
            print(f"Error processing annotation ID {annotation.get('id')}: {e}")
    elif isinstance(segmentation, dict):
        print(f"Skipping RLE segmentation for annotation ID {annotation.get('id')}")

# Save the modified JSON data back to the file
with open('ripeness_class_annotations_bbox.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Bounding box data has been removed and segmentation masks have been converted to bounding box data. The updated data is saved to 'ripeness_class_annotations_bbox.json'.")
