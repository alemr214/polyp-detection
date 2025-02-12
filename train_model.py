from ultralytics import YOLO
import torch


if __name__ == "__main__":
    BASE_PATH_YAML = "configs"
    NAME_DATASET = "sessile_main_kvasir_seg"
    BASE_PATH_MODEL = "runs/trains"

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = YOLO(f"{BASE_PATH_MODEL}/{NAME_DATASET}/yolo11n.pt")

    results = model.train(
        data=f"{BASE_PATH_YAML}/{NAME_DATASET}/dataset.yaml",
        epochs=10,
        imgsz=640,
        batch=-1,
        device=device,
        name=NAME_DATASET,
        project=BASE_PATH_MODEL,
        exist_ok=True,
    )

    best_model = YOLO(f"{BASE_PATH_MODEL}/{NAME_DATASET}/weights/best.pt")

    best_model.export(
        format="onnx",
        opset=12,
        device=device,
    )
