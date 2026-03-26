'''
Downloads dataset from roboflow

To train dataset run in yolov9 directory

active python env first in yolov9 directory
source yolov9-env/bin/activate

python train.py --batch 4 --epochs 25 --img 640 --device 0 --min-items 0 --close-mosaic 15 --data data/vineyard/vineyard_test-6/data.yaml --weights weights/gelan-c.pt --cfg models/detect/gelan-c.yaml --hyp hyp.scratch-high.yaml

python train.py --batch 4 --epochs 25 --img 640 --device 0 --min-items 0 --close-mosaic 15 --data data/vineyard/vineyard_test-6/data.yaml --weights weights/yolov9-c.pt --cfg models/detect/yolov9-c.yaml --hyp hyp.scratch-high.yaml

python train.py --batch 4 --epochs 25 --img 640 --device 0 --min-items 0 --close-mosaic 15 --data data/vineyard/vineyard_segmentation-1/data.yaml --weights weights/yolov9c-seg.pt --cfg models/segment/yolov9-c-dseg.yaml --hyp hyp.scratch-high.yaml

--device cpu
--device 0 == use GPU
'''

from roboflow import Roboflow
import json

# Load the API key
with open('../../config/api_key.json', 'r') as file:
    config = json.load(file)
ROBOFLOW_API_KEY = config.get("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
# project = rf.workspace().project("vineyard_test")
# project = rf.workspace("vista-qsopb").project("vineyard_segmentation")
project = rf.workspace("vineyarddetection").project("vineyard_segmentation_paper")
version = project.version(2)

# Download dataset in YOLO format
dataset = version.download("coco-segmentation")  # Options: "yolov5", "yolov8", "coco", "coco-segmentation", etc.

# Print dataset location
print(f"Dataset downloaded to: {dataset.location}")