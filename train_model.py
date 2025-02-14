from ultralytics import YOLO
import torch


if __name__ == "__main__":
    BASE_PATH_YAML = "configs"
    BASE_PATH_MODEL = "runs"
    TRAIN_PATH = "trains"
    PREDICT_PATH = "predicts"
    PREDICT_IMAGES_PATH = "data/clean/predict_images"
    NAME_DATASET = "sessile_main_kvasir_seg"

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = YOLO(f"{BASE_PATH_MODEL}/{TRAIN_PATH}/{NAME_DATASET}/yolo11n.pt")

    # Train model
    results = model.train(
        data=f"{BASE_PATH_YAML}/{NAME_DATASET}/dataset.yaml",
        epochs=10,
        imgsz=640,
        batch=-1,
        save_period=5,  # Save model every 10 epochs
        cache=True,  # Cache images for faster training
        device=device,
        name=NAME_DATASET,
        project=f"{BASE_PATH_MODEL}/{TRAIN_PATH}",
        exist_ok=True,
    )

    # Best model
    best_model = YOLO(f"{BASE_PATH_MODEL}/{TRAIN_PATH}/{NAME_DATASET}/weights/best.pt")

    # Validation model
    metrics = best_model.val()
    print(metrics.box.map)
    print(metrics.box.map)  # mAP50-95
    print(metrics.box.map50)  # mAP50
    print(metrics.box.map75)  # mAP75
    print(metrics.box.maps)  # list of mAP50-95 for each category

    # Export to ONNX format
    best_model.export(
        format="onnx",
        opset=12,
        device=device,
    )
