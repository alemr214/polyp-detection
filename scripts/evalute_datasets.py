# Importa la función xywh2xyxy del módulo ultralytics.utils.ops para convertir
# cajas en formato (x_center, y_center, width, height) a formato (x1, y1, x2, y2)
from ultralytics.utils.ops import xywh2xyxy

# Importa la librería torch, que se usa para crear y manipular tensores
import torch

# Importa el módulo os para trabajar con rutas del sistema operativo y operaciones relacionadas
import os

# Importa el módulo glob para buscar archivos que coincidan con un patrón (por ejemplo, *.txt)
import glob


def calculate_iou(box1: list[int], box2: list[int]):
    """
    Calcula el Intersection over Union (IoU) entre dos cajas en formato (x1, y1, x2, y2).

    Parámetros:
      box1, box2: Listas o tuplas con 4 elementos representando las coordenadas de la caja.
    Retorna:
      El valor del IoU, que es el área de intersección dividida por el área de unión.
    """
    # Calcula la coordenada x del vértice superior izquierdo de la intersección
    x1 = max(box1[0], box2[0])
    # Calcula la coordenada y del vértice superior izquierdo de la intersección
    y1 = max(box1[1], box2[1])
    # Calcula la coordenada x del vértice inferior derecho de la intersección
    x2 = min(box1[2], box2[2])
    # Calcula la coordenada y del vértice inferior derecho de la intersección
    y2 = min(box1[3], box2[3])

    # Determina la anchura del área de intersección (asegurándose de que no sea negativa)
    inter_width = max(0, x2 - x1)
    # Determina la altura del área de intersección (asegurándose de que no sea negativa)
    inter_height = max(0, y2 - y1)
    # Calcula el área de intersección
    inter_area = inter_width * inter_height

    # Calcula el área de la primera caja
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    # Calcula el área de la segunda caja
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calcula el área de unión de ambas cajas
    union_area = box1_area + box2_area - inter_area

    # Evita la división por cero: si el área de unión es cero, retorna 0
    if union_area == 0:
        return 0
    # Retorna el valor de IoU como la razón entre el área de intersección y el área de unión
    return inter_area / union_area


def load_boxes_from_file(file_path):
    """
    Lee un archivo .txt y devuelve una lista de cajas.

    Se espera que cada línea del archivo tenga el formato:
      clase, x_center, y_center, width, height
    La etiqueta de clase se ignora (o se puede utilizar según se requiera).

    Parámetros:
      file_path: Ruta del archivo .txt que contiene las coordenadas.
    Retorna:
      Una lista de cajas en formato (x1, y1, x2, y2)
    """
    boxes = []  # Inicializa la lista que almacenará las cajas
    # Abre el archivo en modo lectura
    with open(file_path, "r") as f:
        # Itera sobre cada línea del archivo
        for line in f:
            # Elimina espacios en blanco al inicio y final de la línea
            line = line.strip()
            # Si la línea está vacía, se salta
            if not line:
                continue
            # Divide la línea en partes usando espacios en blanco como separador
            parts = line.split()
            # Se asume que la primera parte es la clase y los siguientes elementos son las coordenadas
            try:
                # Convierte las siguientes 4 partes (coordenadas) a valores float
                coords = list(map(float, parts[1:5]))
                # Convierte las coordenadas de (x_center, y_center, width, height) a (x1, y1, x2, y2)
                # Se crea un tensor con torch para pasarlo a la función de conversión
                box = xywh2xyxy(torch.tensor(coords))
                # Añade la caja convertida a la lista de cajas
                boxes.append(box)
            except Exception as e:
                # Si ocurre un error durante el procesamiento, se imprime el error, la línea y la ruta del archivo
                print(f"Error {e} procesando la línea: {line} en {file_path}")
    # Retorna la lista de cajas procesadas
    return boxes


def evaluate_predictions(gt_dir, pred_dir, iou_threshold=0.5):
    """
    Evalúa las predicciones en comparación con el ground truth (verdad de terreno).

    Calcula la sensibilidad (recall) y el false positive rate (tasa de falsos positivos):
      - Sensibilidad = True Positives / Total de objetos reales
      - False Positive Rate = False Positives / Total de objetos reales

    Se asume que los archivos en ambos directorios (gt_dir y pred_dir) tienen el mismo nombre.

    Parámetros:
      gt_dir: Directorio que contiene los archivos de ground truth.
      pred_dir: Directorio que contiene los archivos con las predicciones.
      iou_threshold: Umbral para considerar una predicción como acierto (default 0.5).

    Retorna:
      Una tupla con la sensibilidad y el false positive rate.
    """
    # Obtiene la lista de archivos .txt en el directorio de ground truth
    gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))

    total_gt = 0  # Contador del total de objetos reales en ground truth
    total_tp = 0  # Contador de predicciones correctas (true positives)
    total_fp = 0  # Contador de predicciones incorrectas (false positives)

    # Itera sobre cada archivo de ground truth
    for gt_file in gt_files:
        # Obtiene el nombre del archivo (sin la ruta)
        base_name = os.path.basename(gt_file)
        # Construye la ruta del archivo de predicción correspondiente asumiendo que tienen el mismo nombre
        pred_file = os.path.join(pred_dir, base_name)

        # Lee las cajas del archivo de ground truth
        gt_boxes = load_boxes_from_file(gt_file)
        # Lee las cajas del archivo de predicción si el archivo existe; de lo contrario, utiliza una lista vacía
        pred_boxes = (
            load_boxes_from_file(pred_file) if os.path.exists(pred_file) else []
        )

        # Suma el número de cajas de ground truth al total de objetos reales
        total_gt += len(gt_boxes)

        # Crea una lista de booleanos para marcar qué cajas de ground truth ya han sido emparejadas
        matched_gt = [False] * len(gt_boxes)

        # Para cada caja predicha, se busca la caja de ground truth con la mayor IoU
        for pred_box in pred_boxes:
            best_iou = 0  # Inicializa la mejor IoU encontrada para la predicción actual
            best_gt_idx = (
                -1
            )  # Inicializa el índice de la caja ground truth con la mejor IoU
            # Itera sobre cada caja de ground truth, manteniendo el índice y la caja
            for idx, gt_box in enumerate(gt_boxes):
                # Calcula el IoU entre la caja predicha y la caja de ground truth actual
                iou = calculate_iou(pred_box, gt_box)
                # Si se encuentra una IoU mayor que la actual, actualiza el mejor valor e índice
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            # Si la mejor IoU es mayor o igual que el umbral y la caja ground truth aún no ha sido emparejada:
            if best_iou >= iou_threshold and not matched_gt[best_gt_idx]:
                total_tp += 1  # Se considera un acierto (True Positive)
                matched_gt[best_gt_idx] = (
                    True  # Marca la caja ground truth como emparejada
                )
            else:
                # Si la predicción no cumple con el umbral o la caja ya fue emparejada,
                # se cuenta la predicción como un falso positivo (False Positive)
                total_fp += 1

    # Calcula la sensibilidad (recall): número de aciertos dividido por el total de objetos reales
    sensitivity = total_tp / total_gt if total_gt > 0 else 0
    # Calcula el false positive rate: número de falsos positivos dividido por el total de objetos reales
    false_positive_rate = total_fp / total_gt if total_gt > 0 else 0

    # Imprime los resultados de la evaluación en consola
    print("Total objetos reales:", total_gt)
    print("True Positives:", total_tp)
    print("False Positives:", total_fp)
    print("Sensibilidad (Recall):", sensitivity)
    print("False Positive Rate:", false_positive_rate)

    # Retorna la sensibilidad y el false positive rate
    return sensitivity, false_positive_rate


# Bloque principal: se ejecuta únicamente si el script se ejecuta directamente, no cuando se importa como módulo
if __name__ == "__main__":
    # Define la ruta del directorio con los archivos de ground truth
    ground_truth_directory = "data/clean/polypgen/labels/test_sequence"  # Cambia esta ruta según tu estructura de carpetas
    # Define la ruta del directorio con los archivos de predicción
    predictions_directory = "runs/predict/polypgen_sequence/labels"  # Cambia esta ruta según tu estructura de carpetas

    # Llama a la función evaluate_predictions con el umbral IoU deseado (0.5 en este caso)
    evaluate_predictions(
        ground_truth_directory, predictions_directory, iou_threshold=0.5
    )
