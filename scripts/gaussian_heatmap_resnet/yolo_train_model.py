from ultralytics import YOLO

def main():
    model = YOLO("../roboflow_scripts/yolo11x-obb.pt")

    model.train(
        data='dataset_split/data.yaml',
        project="./models/patch_images/yolo/trained/",
        name="vineyard_object_detection_obb_2",
        epochs=100,
        # imgsz=640,
        imgsz=[800, 800], # [height, width]
        batch=2,
        val=True,
        patience=25, 
        optimizer='auto',
        augment=True,
        amp=True,
        exist_ok=True,
        task="obb"
    )

if __name__ == "__main__":
    main()