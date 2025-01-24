import os
import torch
import numpy as np

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
