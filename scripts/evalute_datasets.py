import os
import glob


def read_boxes(file_path):
    """Lee un archivo de etiquetas YOLO y devuelve una lista de cajas."""
    boxes = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = parts
                boxes.append((float(cls), float(x), float(y), float(w), float(h)))
    return boxes


def convert_to_corners(box):
    """Convierte (x_center, y_center, width, height) a (x1, y1, x2, y2)."""
    cls, x_center, y_center, width, height = box
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return (x1, y1, x2, y2)


def compute_iou(box1, box2):
    """Calcula el IoU entre dos cajas."""
    x1_1, y1_1, x2_1, y2_1 = convert_to_corners(box1)
    x1_2, y1_2, x2_2, y2_2 = convert_to_corners(box2)

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


# Directorios de etiquetas
gt_dir = "data/clean/cvc_clinic_db/labels/test"
pred_dir = "runs/predict/cvc_clinic_db/labels"

gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))

TP_total = 0
FP_total = 0
FN_total = 0
TN_total = 0

iou_threshold = 0.5

for gt_file in gt_files:
    filename = os.path.basename(gt_file)
    pred_file = os.path.join(pred_dir, filename)

    gt_boxes = read_boxes(gt_file)
    pred_boxes = read_boxes(pred_file) if os.path.exists(pred_file) else []

    matched_pred_indices = set()
    TP = 0
    FN = 0

    for gt in gt_boxes:
        best_iou = 0
        best_pred_idx = -1
        for i, pred in enumerate(pred_boxes):
            if i in matched_pred_indices:
                continue
            iou = compute_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = i
        if best_iou >= iou_threshold:
            TP += 1
            matched_pred_indices.add(best_pred_idx)
        else:
            FN += 1

    FP = len(pred_boxes) - len(matched_pred_indices)

    # Cálculo de TN (cuando no hay detecciones en imágenes sin objetos)
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        TN = 1
    else:
        TN = 0  # En este caso no podemos calcular TN con precisión sin segmentación adicional

    TP_total += TP
    FN_total += FN
    FP_total += FP
    TN_total += TN

sensitivity = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
specificity = TN_total / (TN_total + FP_total) if (TN_total + FP_total) > 0 else 0

print("Resultados:")
print("TP:", TP_total)
print("FN:", FN_total)
print("FP:", FP_total)
print("TN:", TN_total)
print("Sensibilidad (Recall):", sensitivity)
print("Especificidad:", specificity)
