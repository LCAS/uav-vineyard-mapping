from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image
from roboflow import Roboflow
import yaml

#Instance
# model = YOLO('yolov11n-seg.yaml')  # build a new model from YAML
model = YOLO('yolo11x.pt')  # Transfer the weights from a pretrained model (recommended for training)

# define number of classes based on YAML
with open("../../data/datasets/fruit_detector-4/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

#Define a project --> Destination directory for all results
project = "../../data/datasets/trained/fruit_detector-v4_yolox/"
#Define subdirectory for this specific training
name = "run1" #note that if you run the training again, it creates a directory: 200_epochs-2

# Train the model
# results = model.train(data='../../data/datasets/vineyard_segmentation-8/data.yaml',
#                       project=project,
#                       name=name,
#                       epochs=100,
#                       patience=0, # set patience=0 to disable early stopping
#                       batch=16,
#                       #imgsz=640,
#                       device=0)

results = model.train(
    data='../../data/datasets/fruit_detector-4/data.yaml',
    project=project,
    name=name,
    epochs=40,
    patience=15,
    batch=4,
    device=0,
    optimizer='Adam',
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    augment=True,
    amp=True,
    imgsz=(640, 640)
)
