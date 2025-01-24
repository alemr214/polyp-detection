import shutil
import os
import numpy as np


def create_dir(output_path: str) -> None:
    """
    Create directory if no exists

    Args:
        output_path (str): Output directory
    """
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    except OSError:
        print(f"Error: Creating directory. {output_path}")


# Detect images in a folder an return a sorted list of images path
def detect_imgs(source_path: str, image_ext: str = ".png") -> np.ndarray:
    """
    Returns a sorted list of images path with some extension (.png as default) in a directory path

    Args:
        source_path (str): Images path
        image_ext (str, optional): Extension of images. Defaults to ".png".

    Returns:
        np.ndarray: Sorted list of image paths
    """
    items = os.listdir(source_path)
    flist = [
        os.path.join(source_path, names)
        for names in items
        if names.endswith(image_ext) or names.endswith(image_ext.upper())
    ]
    return np.sort(flist)


# Copy images from the source path to the output path to save us a backup in case of corruption
def copy_images(source_path: str, output_path: str, image_ext: str) -> None:
    """
    Copy images from the source path to the output path

    Args:
        source_path (str): source path of the images
        output_path (str): output path to save the images
        image_ext (str): image extension
    """
    create_dir(output_path)  # Create output directory if not exists

    # Get list of image paths using the detect_imgs function
    images = detect_imgs(source_path, image_ext)

    # Copy each image to the output path
    for img_path in images:
        try:
            output_img_path = os.path.join(output_path, os.path.basename(img_path))
            shutil.copy(img_path, output_img_path)
        except Exception as e:
            print(f"Error al copiar {img_path}: {e}")
