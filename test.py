import os
import glob


def yolo_to_box(coords):
    """
    Convierte las coordenadas de YOLO (x_center, y_center, width, height)
    a formato de caja: (x1, y1, x2, y2).
    Se asume que las coordenadas están normalizadas y en la misma escala.
    """
    x_center, y_center, width, height = coords
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return (x1, y1, x2, y2)


def compute_iou(box1, box2):
    """
    Calcula el IoU entre dos cajas en formato (x1, y1, x2, y2).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    # Área de cada caja
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area


def load_boxes_from_file(file_path):
    """
    Lee un archivo .txt y devuelve una lista de cajas.
    Cada línea debe tener: clase, x_center, y_center, width, height
    """
    boxes = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Si se tiene la etiqueta de clase, se ignora o se puede usar según se requiera
            # Se asume que las coordenadas vienen después de la clase
            try:
                # Convertir a float los valores de coordenadas
                coords = list(map(float, parts[1:5]))
                box = yolo_to_box(coords)
                boxes.append(box)
            except Exception as e:
                print(f"Error procesando la línea: {line} en {file_path}")
    return boxes


def evaluate_predictions(gt_dir, pred_dir, iou_threshold=0.5):
    """
    Evalúa las predicciones comparándolas con el ground truth.
    Se calcula la sensibilidad y el false positive rate:
        - Sensibilidad = True Positives / total de objetos reales
        - False Positive Rate = False Positives / total de objetos reales
    Se espera que los archivos de ambos directorios tengan el mismo nombre.
    """
    gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))

    total_gt = 0  # Total de objetos reales
    total_tp = 0  # Total de aciertos (true positives)
    total_fp = 0  # Total de falsos positivos

    for gt_file in gt_files:
        # Suponemos que el nombre del archivo es el mismo en ambas carpetas
        base_name = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, base_name)

        gt_boxes = load_boxes_from_file(gt_file)
        pred_boxes = (
            load_boxes_from_file(pred_file) if os.path.exists(pred_file) else []
        )

        total_gt += len(gt_boxes)

        matched_gt = [False] * len(gt_boxes)

        # Para cada predicción, se busca la mejor coincidencia en ground truth
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for idx, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            # Si la mejor coincidencia supera el umbral y ese objeto real aún no ha sido asociado, es un acierto
            if best_iou >= iou_threshold and not matched_gt[best_gt_idx]:
                total_tp += 1
                matched_gt[best_gt_idx] = True
            else:
                # Si no se cumple la coincidencia, se cuenta como falso positivo
                total_fp += 1

    sensitivity = total_tp / total_gt if total_gt > 0 else 0
    false_positive_rate = total_fp / total_gt if total_gt > 0 else 0

    print("Total objetos reales:", total_gt)
    print("True Positives:", total_tp)
    print("False Positives:", total_fp)
    print("Sensibilidad (Recall):", sensitivity)
    print("False Positive Rate:", false_positive_rate)

    return sensitivity, false_positive_rate


# Ejemplo de uso:
if __name__ == "__main__":
    # Define las rutas a los directorios con ground truth y predicciones
    ground_truth_directory = (
        "data/clean/polypgen/labels/test_sequence"  # Cambia por tu ruta
    )
    predictions_directory = (
        "runs/predict/polypgen_sequence/labels"  # Cambia por tu ruta
    )

    # Puedes ajustar el umbral IoU según tu criterio
    evaluate_predictions(
        ground_truth_directory, predictions_directory, iou_threshold=0.5
    )
