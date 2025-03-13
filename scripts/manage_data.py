import os
import random
import yaml
import shutil
import numpy as np


# Create directory in a output path
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
def detect_imgs(
    source_path: str, image_ext: list[str] | str = [".png", ".jpg", ".tif"]
) -> np.ndarray:
    """
    Returns a sorted list of images path with some extension (defaults: .png, .jpg, .tif) in a directory path

    Args:
        source_path (str): Images path
        image_ext (list[str] | str, optional): Extension fo images. Defaults to [".png", ".jpg", ".tif"].

    Returns:
        np.ndarray: Sorted list of image paths
    """
    items = os.listdir(source_path)
    flist = [
        os.path.join(source_path, item_name)
        for item_name in items
        if any(item_name.endswith(ext.lower()) for ext in image_ext)
    ]
    return np.sort(flist)


# Copy images from the source path to the output path to save us a backup in case of corruption
def copy_images(source_path: str, output_path: str) -> None:
    """
    Copy images from the source path to the output path

    Args:
        source_path (str): source path of the images
        output_path (str): output path to save the images
        image_ext (str): image extension
    """
    create_dir(output_path)

    # Get list of image paths using the detect_imgs function
    images = detect_imgs(source_path)

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


def rename_files(source_path: str, prefix: str) -> None:
    """
    Rename files in a directory with a prefix and a number counter

    Args:
        source_path (str): source path of the files
        prefix (str): prefix to rename the files

    Raises:
        FileNotFoundError: If the source path is not found
    """
    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Directory {source_path} not found.")

    files = sorted(
        [file for file in os.listdir(source_path)],
        key=detect_numbers_in_name,
    )

    for i, file in enumerate(files, start=1):
        _, ext = os.path.splitext(file)
        new_name = f"{prefix}_{i:05d}{ext}"
        current_path = os.path.join(source_path, file)
        new_path = os.path.join(source_path, new_name)

        os.rename(current_path, new_path)
        print(f"{file} -> {new_name}")


def count_images(source_path: str) -> int:
    """
    Count the number of images in a directory

    Args:
        source_path (str): Source path of the images

    Returns:
        int: Number of images
    """
    return len(detect_imgs(source_path))


def move_files(source_dir: str, dest_dir: str, files: list) -> None:
    """
    Move files from source to destination directory.

    Args:
        source_dir (str): Source directory of files
        dest_dir (str): Destination directory to move files
        files (list): List of filenames to move
    """
    create_dir(dest_dir)

    for file in files:
        try:
            source_file_path = os.path.join(source_dir, file)
            dest_file_path = os.path.join(dest_dir, file)
            shutil.move(source_file_path, dest_file_path)
            print(f"Moved {file} to {dest_dir}")
        except Exception as e:
            print(f"Error moving {file}: {e}")


def split_data(
    image_path: str,
    label_path: str,
    train_ratio: float = 0.7,
    seed: int = 42,
):
    """
    Split data into train, validation, and test sets and move them to their respective directories given a train ratio

    Args:
        image_path (str): image path to split
        label_path (str): label path to split
        train_ratio (float, optional): Train ratio to split data. Defaults to 0.7.
        seed (int, optional): Random number seed. Defaults to 42.
    """
    random.seed(seed)

    # Get sorted list of image and label files
    image_files = sorted(os.listdir(image_path))
    label_files = sorted(os.listdir(label_path))

    # Combine image and label pairs and shuffle them
    data = list(zip(image_files, label_files))
    random.shuffle(data)

    # Split the data into train, validation, and test sets
    train_end = int(train_ratio * len(data))  # End index for train subset
    val_end = train_end + int(
        (1 - train_ratio) / 2 * len(data)
    )  # End index for val subset

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    subsets = ["train", "val", "test"]
    splitted_data = [train_data, val_data, test_data]
    for subset, split_data in list(zip(subsets, splitted_data)):
        move_files(
            image_path, os.path.join(image_path, subset), [img for img, _ in split_data]
        )
        move_files(
            label_path, os.path.join(label_path, subset), [lbl for _, lbl in split_data]
        )

    print("Data splitted and moved successfully")


def create_yaml_file(
    base_dataset_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    number_classes: int,
    classes_names: list[str],
    yaml_file_path: str,
) -> None:
    """
    Create a YAML file with the dataset information.

    Args:
        base_dataset_path (str): Dataset path where the train, val and test folders are located.
        train_path (str): Relative path to the base folder.
        val_path (str): Relative path to the base folder.
        test_path (str): Relative path to the base folder.
        number_classes (int): Number of classes in the dataset.
        classes_names (list[str]): List with the classes names.
        yaml_file_path (str): Path to save the YAML file.
    """
    data = {
        "path": base_dataset_path,
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": number_classes,
        "names": classes_names,
    }

    create_dir(yaml_file_path)
    with open(yaml_file_path + "dataset.yaml", "w") as file:
        yaml.dump(data, file)

    print(f"YAML file created at {yaml_file_path}")
