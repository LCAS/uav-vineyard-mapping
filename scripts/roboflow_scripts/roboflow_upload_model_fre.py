import roboflow
import json

import sys
sys.path.append('/home/cabbage/Code/yolov9')

# Load the API key
with open('../../config/api_key.json', 'r') as file:
    config = json.load(file)
ROBOFLOW_API_KEY = config.get("ROBOFLOW_API_KEY_FRE")

rf = roboflow.Roboflow(api_key=ROBOFLOW_API_KEY)

project = rf.workspace("field-robot-event").project("fruit_detector-o8puf")

#can specify weights_filename, default is "weights/best.pt"
version = project.version(4)

version.deploy(model_type="yolov11", model_path=f"/home/cheddar/code/vineyard_detection/data/datasets/trained/fruit_detector-v4_yolox/run1")