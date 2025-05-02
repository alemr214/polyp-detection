from .manage_data import detect_files
from ultralytics.utils.ops import xywh2xyxy
import torch
import numpy as np
import os


def calculate_iou(box_gt: np.ndarray, box_pred: np.ndarray) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes, the ground truth and the predicted box for these case, ensurence that the boxes are in the format (x1, y1, x2, y2) and returning 0 if the inter area is 0.

    Args:
        box_gt (np.ndarray): Coordinates of the ground truth bounding box.
        box_pred (np.ndarray): Coordinates of the predicted bounding box.

    Returns:
        float: IoU value, which is the area of intersection divided by the area of union.
    """
    #
    x1_inter = max(box_gt[0], box_pred[0])
    y1_inter = max(box_gt[1], box_pred[1])
    x2_inter = min(box_gt[2], box_pred[2])
    y2_inter = min(box_gt[3], box_pred[3])

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    width_inter = x2_inter - x1_inter
    height_inter = y2_inter - y1_inter

    area_inter = width_inter * height_inter

    area_gt = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])

    area_union = area_gt + area_pred - area_inter

    iou = area_inter / area_union if area_union > 0 else 0.0

    return iou


def get_boxes_from_file(source_path: str) -> list:
    """
    Return a list of boxes from a file, ensuring that the boxes are in the format (x1, y1, x2, y2) and returning 0 if the inter area is 0.

    Args:
        source_path (str): Path to the file containing the boxes.

    Returns:
        list: List of boxes in the format (x1, y1, x2, y2).
    """
    boxes = []
    with open(source_path, "r") as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            if len(parts) == 5:
                coords = list(map(float, parts[1:5]))
                box = xywh2xyxy(torch.tensor(coords))
                boxes.append(box)
    return boxes


def evalute_predictions(gt_path: str, pred_path: str, iou_threshold: float = 0.5):
    """
        Get

    Args:
        gt_path (str): _description_
        pred_path (str): _description_
        iou_threshold (float, optional): _description_. Defaults to 0.5.
    """
    # Counters
    total_gt = 0
    total_pred = 0

    # True Positives, False Positives, False Negatives
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Sensibility and False Positive Rate
    sensibility = 0
    fp_rate = 0

    gt_files = detect_files(gt_path, [".txt"])

    for gt_file in gt_files:
        # Load predicted files
        base_name = os.path.basename(gt_file)
        pred_file = os.path.join(pred_path, base_name)

        gt_boxes = get_boxes_from_file(gt_file)
        pred_boxes = get_boxes_from_file(pred_file) if os.path.exists(pred_file) else []

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        for gt_box, pred_box in zip(gt_boxes, pred_boxes):
            iou = calculate_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                total_tp += 1
            if iou == 0:
                total_fp += 1

    total_fn = total_gt - total_tp
    sensibility = total_tp / total_gt if total_gt > 0 else 0
    fp_rate = total_fp / total_pred if total_pred > 0 else 0

    print(f"GT files: {len(gt_files)}")

    print(f"GT boxes: {total_gt}")
    print(f"Pred boxes: {total_pred}")
    print(f"True Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"Sensibility: {sensibility * 100:.2f} %")
    print(f"False Positive Rate: {fp_rate * 100:.2f} %")
