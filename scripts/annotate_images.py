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
