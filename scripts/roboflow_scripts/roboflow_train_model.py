from ultralytics import YOLO

def main():
    # model = YOLO("yolo11x.pt")
    # model = YOLO("yolo11x-obb.pt")
    model = YOLO("yolo11x-seg.pt")

    # Train normally
    model.train(
        data='../../data/datasets/vineyard_segmentation-22/data.yaml',
        project = "../../data/datasets/trained/vineyard_segmentation-22/",
        epochs=60,
        # imgsz=640,
        # imgsz=[720, 960], # [height, width] # 1520 x 2028
        # imgsz=[760, 1014],
        imgsz=[640, 640],
        batch=1,
        val=True,
        patience=25, 
        optimizer='auto',
        augment=True,
        amp=True,
        exist_ok=False # Automatically overwrite existing results folder if true, if false, create new folder with incremented name
    )

if __name__ == "__main__":
    main()