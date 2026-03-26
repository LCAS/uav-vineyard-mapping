from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import random
import json

import wandb
import pandas

SEED = random.randint(0,1000)
OPTIMIZER='auto' #{'RAdam', 'NAdam', 'auto', 'AdamW', 'Adamax', 'Adam', 'SGD', 'RMSProp'}

# Set the random seed for reproducibility
random.seed(SEED)

# Load the API key
with open('../config/api_key.json', 'r') as file:
    config = json.load(file)
W_AND_B_API_KEY = config.get("W_AND_B_API_KEY")
DATASET_VERSION = "11"

def train_custom_yolov11(
    data_yaml_path,
    # model_size='yolo11x',
    model_size='yolo11x-seg',
    epochs=300,
    batch_size=8, # -1: automatically determine the batch size that can be efficiently processed based on your device's capabilities
    img_size=640,
    project_name='yolov11_custom'
):
    # Verify the data YAML file
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    print(f"Dataset configuration loaded. Classes: {data_config['names']}")

    wandb.login(key=W_AND_B_API_KEY)
    wandb.init(project="vineyard_segmentation", entity="jondave-university-of-lincoln", tensorboard=True)

    # Load a pre-trained YOLOv11 model
    model = YOLO(f"../yolo_models/{model_size}.pt")
    model_name=f"{project_name}_D{DATASET_VERSION}_B{batch_size}_{SEED}_{OPTIMIZER}"

    # Train and tune the model on your custom dataset
    result_grid = model.tune(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=f"tuned_models",
        name=model_name,
        patience=25,                                              
        optimizer=str(OPTIMIZER), #SGD, Adam, AdamW, NAdam, RAdam, RMSProp
        exist_ok=True,
        pretrained=True,
        verbose=True,
        save_period = 5,
        seed=SEED,
        plots=True,             # Generate plots for TensorBoard
        save=True,              # Save checkpoint
        val=True,
        use_ray=True,
        iterations=50,
        # grace_period=25,
        gpu_per_trial=1
    )

    wandb.finish()

    # print(f"Best Tuning Results:")
    # for i, result in enumerate(result_grid):
    #     print(f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}")

    trial_results = []

    for i, trial in enumerate(result_grid):
        print(f"Trial #{i} config:", trial.config)
        print(f"Trial #{i} metrics:", trial.metrics)

        if trial.metrics and isinstance(trial.metrics, dict):
            trial_data = {
                **trial.config,
                **trial.metrics
            }
        else:
            # fallback: only save config
            trial_data = trial.config

        trial_results.append(trial_data)

    # Convert to DataFrame and save
    df = pandas.DataFrame(trial_results)
    if not df.empty:
        df.to_csv("TUNING_RESULTS.csv", index=False)
        print(df.head())
    else:
        print("⚠️ No trial results were found or metrics were not logged.")

    try:
        print(f"\n\n\nBest Tuning Results (again):")
        print(f"{result_grid.get_dataframe()}")

        result_grid.get_dataframe().to_csv(
            "TUNING_RESULTS_2.csv",
            sep=',',
            encoding='utf-8',
            index=True,    # keep index (trial number)
            header=True    # write column headers
        )
    except Exception as e:
        print("⚠️ Could not save to csv:", str(e))

    print(f"\n\n\nBest Tuning Results (again) (again):")

    # Check if the model has a usable scoring function
    scoring_function = None
    try:
        scoring_function = model.metrics if callable(model.metrics) else model.metrics()
    except Exception as e:
        print("⚠️ Could not determine model scoring function:", str(e))

    # Try to get the best result based on that scoring function
    best_result = None
    if scoring_function:
        try:
            best_result = result_grid.get_best_result(scoring_function)
        except Exception as e:
            print("⚠️ Failed to get best result from result grid:", str(e))

    # Print best trial metrics if available
    if best_result:
        try:
            for key, value in best_result.metrics().items():
                print(f"{key}: {value}")
        except Exception as e:
            print("⚠️ Failed to extract metrics from best result:", str(e))
    else:
        print("⚠️ No best result available.")

    # print(f"\n\n\nBest Tuning Results (again) (again):")
    # for key, value in result_grid.get_best_result(model.metrics()).metrics().items():
    #     print(key, ":", value)



    # model_path = Path(f"./tuned_models/{model_name}/weights/best.pt")
    # print(f"Tuning completed. Best model saved at: {model_path}")
    
    return None #model_path

def export_model(model_path, format='onnx'):
    """
    Export the trained model to different formats.
    
    Args:
        model_path: Path to the trained model
        format: Export format (onnx, tflite, coreml, etc.)
    
    Returns:
        Path to the exported model
    """
    model = YOLO(model_path)
    exported_path = model.export(format=format)
    print(f"Model exported to {format.upper()} format at: {exported_path}")
    return exported_path

def main():
    # Path to your YAML file with dataset configuration
    # data_yaml_path = f"/home/cheddar/code/vineyard_detection/data/datasets/vineyard_test-{DATASET_VERSION}/data.yaml"
    data_yaml_path = f"/home/cheddar/code/vineyard_detection/data/datasets/vineyard_segmentation-{DATASET_VERSION}/data.yaml"
    
    # Train the model
    model_path = train_custom_yolov11(
        data_yaml_path=data_yaml_path,
        model_size='yolo11n-seg',  # Choose from: yolov11n, yolov11s, yolov11m, yolov11l, yolov11x
        epochs=25,
        project_name='yolov11x_vineyard_segmentation',
        img_size=1280,
        batch_size=2
    )
    
    # Export the model (optional)
    # exported_model_path = export_model(model_path, format='onnx')
    
    print("YOLOv11 tuning pipeline completed successfully!")

if __name__ == "__main__":
    # Install ultralytics if not already installed
    # os.system("pip install ultralytics")
    main()