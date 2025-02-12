from ultralytics import YOLO

model = YOLO("runs/trains/yolo11n.pt")

results = model.train(
    data="configs/sessile_main_kvasir_seg/dataset.yaml",
    epochs=10,
    imgsz=640,
    batch=-1,
    device="mps",
    name="sessile_main_kvasir_seg",
    project="runs/trains",
    exist_ok=True,
)
