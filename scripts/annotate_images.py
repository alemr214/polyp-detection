import os
import torch
import numpy as np
from torchvision.io import read_image, ImageReadMode

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Detect images in a folder an return a sorted list of images path
def detect_imgs(infolder: str, ext: str = ".png") -> np.ndarray:
    """
    Returns a sorted list of images path with some extension (.png as default) in a directory path

    Args:
        infolder (str): Images path
        ext (str, optional): Extension of images. Defaults to ".png".

    Returns:
        np.ndarray: Sorted list of image paths
    """
    items = os.listdir(infolder)
    flist = [
        os.path.join(infolder, names)
        for names in items
        if names.endswith(ext) or names.endswith(ext.upper())
    ]
    return np.sort(flist)


# Create directory if no exists
def create_dir(path: str) -> None:
    """
    Create directory if no exists

    Args:
        path (str): Directory path
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: Creating directory. {path}")


# Save bounding box in YOLO format
def save_bbox(txt_path: str, line: str) -> None:
    """
    Save bounding box in YOLO format in a txt file with the specified path

    Args:
        txt_path (str): Path to save the txt file
        line (str): Parameters of the bounding box in YOLO format
    """
    txt_path = txt_path + ".txt"
    with open(txt_path, "w") as myfile:
        myfile.write(line + "\n")


# Calculate bounding box coordinates
def valRect(coord: torch.Tensor) -> list:
    """
    Calculate bounding box coordinates from a list of coordinates (x, y) of a mask object

    Args:
        coord (np.ndarray): List of coordinates (x, y) of a mask object in a binary mask with shape (n, 1, 2) where:
                            n: number of pixels in the mask object
                            1: number of channels
                            2: x and y coordinates

    Returns:
        list: Returns a list with the coordinates of the bounding box [xmin, ymin, xmax, ymax]
    """
    xmin = torch.min(coord[:, 0]) + 1
    ymin = torch.min(coord[:, 1]) + 1
    xmax = torch.max(coord[:, 0]) + 1
    ymax = torch.max(coord[:, 1]) + 1
    return [xmin.item(), ymin.item(), xmax.item(), ymax.item()]
