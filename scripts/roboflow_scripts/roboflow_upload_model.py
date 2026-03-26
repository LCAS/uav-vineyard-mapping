import roboflow
import json

import sys
sys.path.append('/home/cabbage/Code/yolov9')

# Load the API key
with open('../../config/api_key.json', 'r') as file:
    config = json.load(file)
ROBOFLOW_API_KEY = config.get("ROBOFLOW_API_KEY")
# ROBOFLOW_API_KEY = config.get("ROBOFLOW_API_KEY_FRE")

rf = roboflow.Roboflow(api_key=ROBOFLOW_API_KEY)

#project = rf.workspace("vista-qsopb").project("vineyard_test")
project = rf.workspace().project("vineyard_segmentation")

#can specify weights_filename, default is "weights/best.pt"
version = project.version(20)
# version.deploy("yolov9", "/home/cheddar/code/yolov9/runs/train/exp3", "weights/best.pt")
# version.deploy(model_type="yolov9", model_path=f"/home/cheddar/code/yolov9/runs/train/exp4")
version.deploy(model_type="yolov11", model_path=f"/home/cheddar/code/vineyard_detection/data/datasets/trained/vineyard_segmentation-20/run1")