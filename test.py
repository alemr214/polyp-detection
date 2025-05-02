# Importa la función xywh2xyxy para convertir cajas de (x_center, y_center, width, height) a (x1, y1, x2, y2)
from ultralytics.utils.ops import xywh2xyxy
from scripts.manage_data import count_lines_in_file, detect_files
import torch
import os
import glob


def calculate_iou(box1, box2):
    """
    Calcula el Intersection over Union (IoU) entre dos cajas en formato (x1, y1, x2, y2).

    Parámetros:
      box1, box2: Listas o tensores con 4 elementos representando las coordenadas de la caja.
    Retorna:
      El valor del IoU, que es el área de intersección dividida por el área de unión.
    """
    # Calcula las coordenadas de la intersección
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calcula el área de intersección
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    # Calcula el área de cada caja
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Área de unión
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def load_boxes_from_file(file_path):
    """
    Lee un archivo .txt y devuelve una lista de cajas convertidas al formato (x1, y1, x2, y2).

    Se espera que cada línea del archivo tenga el formato:
      clase, x_center, y_center, width, height

    Parámetros:
      file_path: Ruta del archivo .txt que contiene las coordenadas.
    Retorna:
      Una lista de cajas en formato (x1, y1, x2, y2)
    """
    boxes = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                coords = list(map(float, parts[1:5]))
                # Se crea un tensor a partir de las coordenadas y se convierte al formato (x1, y1, x2, y2)
                box = xywh2xyxy(torch.tensor(coords))
                # Se convierte el tensor a lista para facilitar las operaciones (opcional)
                boxes.append(box.tolist())
            except Exception as e:
                print(f"Error {e} procesando la línea: {line} en {file_path}")
    return boxes


def evaluate_predictions(gt_dir, pred_dir, iou_threshold=0.5):
    """
    Evalúa las predicciones en comparación con el ground truth.

    Para cada archivo se cuentan:
      - Verdaderos Positivos (TP): Predicciones que tienen IoU >= umbral y que emparejan una única caja de ground truth.
      - Falsos Positivos (FP): Predicciones que no encuentran un match o que se emparejan a una caja ya detectada.
      - Falsos Negativos (FN): Cajas de ground truth que no fueron detectadas por ninguna predicción.

    Parámetros:
      gt_dir: Directorio de ground truth (archivos .txt)
      pred_dir: Directorio de predicciones (archivos .txt)
      iou_threshold: Umbral de IoU para considerar una predicción como acierto.
    Retorna:
      Una tupla (TP, FP, FN) y, si se desea, se pueden calcular métricas como recall o precision.
    """
    gt_files = detect_files(gt_dir, [".txt"])

    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0  # Contador de falsos negativos

    for gt_file in gt_files:
        base_name = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, base_name)

        gt_boxes = load_boxes_from_file(gt_file)
        pred_boxes = (
            load_boxes_from_file(pred_file) if os.path.exists(pred_file) else []
        )

        total_gt += len(gt_boxes)
        matched_gt = [False] * len(gt_boxes)

        # Para cada predicción, se busca su mejor match en las cajas ground truth
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for idx, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= iou_threshold and not matched_gt[best_gt_idx]:
                total_tp += 1
                matched_gt[best_gt_idx] = True
            else:
                total_fp += 1

        # Las cajas ground truth no emparejadas se cuentan como falsos negativos para la imagen actual
        fn = matched_gt.count(False)
        total_fn += fn

    # Se pueden calcular métricas adicionales, por ejemplo, recall y precision
    recall = total_tp / total_gt if total_gt > 0 else 0
    fp_rate = total_fp / total_gt if total_gt > 0 else 0

    print("Total objetos reales (GT):", total_gt)
    print("Verdaderos Positivos (TP):", total_tp)
    print("Falsos Positivos (FP):", total_fp)
    print("Falsos Negativos (FN):", total_fn)
    print("Recall:", recall)
    print("Tasa de Falsos Positivos:", fp_rate)

    return total_tp, total_fp, total_fn


if __name__ == "__main__":
    for dataset in ["polypgen"]:
        # Define las rutas a los directorios de ground truth y predicciones
        ground_truth_directory = f"data/clean/{dataset}/labels/test_single"  # Ajusta según tu estructura de carpetas
        predictions_directory = f"runs/predict/{dataset}_single/labels"  # Ajusta según tu estructura de carpetas

        evaluate_predictions(
            ground_truth_directory, predictions_directory, iou_threshold=0.95
        )
