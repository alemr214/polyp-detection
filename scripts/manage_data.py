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
            print(f"Error to copy {img_path}: {e}")


def detect_numbers_in_name(file_name: str) -> int | str:
    """
    Detect if a file name is a number and return it to int type else return it to string type

    Args:
        file_name (str): File name

    Returns:
        int | str: File name as int or str
    """
    name, _ = os.path.splitext(file_name)
    return int(name) if name.isdigit() else name


def rename_files(source_path: str, prefix: str, file_ext: str):
    """
    Rename files in a directory with a prefix and a number counter

    Args:
        source_path (str): source path of the files
        prefix (str): prefix to rename the files
        file_ext (str): file extension

    Raises:
        FileNotFoundError: If the source path is not found
    """
    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Directory {source_path} not found.")

    files = sorted(
        [f for f in os.listdir(source_path) if f.endswith(file_ext)],
        key=detect_numbers_in_name,
    )

    for i, file in enumerate(files, start=1):
        new_name = f"{prefix}_{i:05d}{file_ext}"
        current_path = os.path.join(source_path, file)
        new_path = os.path.join(source_path, new_name)

        os.rename(current_path, new_path)
        print(f"{file} -> {new_name}")
