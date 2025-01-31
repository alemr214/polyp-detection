from scripts.annotate_images import process_images
from scripts.manage_data import (
    copy_images,
    rename_files,
    split_data,
)
from scripts.prepare_models import create_yaml_file

if __name__ == "__main__":
    # Configuration paths to clean the dataset
    BASE_PATH_RAW = ""
    BASE_PATH_CLEAN = ""
    BASE_PATH_YAML = "./configs"
    NAME_DATASET = ""

    # Data paths
    IMAGES_FOLDER = f"{BASE_PATH_RAW}/{NAME_DATASET}/images"  # To copy images from
    IMAGES_EXT = ""  # Image extension
    MASKS_FOLDER = (
        f"{BASE_PATH_RAW}/{NAME_DATASET}/masks"  # To process masks from images
    )
    MASKS_EXT = ""  # Mask extension
    OUTPUT_IMAGES_FOLDER = (
        f"{BASE_PATH_CLEAN}/{NAME_DATASET}/images"  # To save the copy images
    )
    OUTPUT_LABELS_FOLDER = f"{BASE_PATH_CLEAN}/{NAME_DATASET}/labels"  # To save the .txt bbox annotations from masks
    OUTPUT_MASKS_FOLDER = f"{BASE_PATH_CLEAN}/{NAME_DATASET}/masks"
    CLASS_INDEX = 0  # Class index to save the annotations
    PREFIX = "image"  # Prefix to rename files ("images")

    copy_images(IMAGES_FOLDER, OUTPUT_IMAGES_FOLDER, IMAGES_EXT)
    copy_images(MASKS_FOLDER, OUTPUT_MASKS_FOLDER, MASKS_EXT)

    process_images(
        OUTPUT_IMAGES_FOLDER,
        IMAGES_EXT,
        MASKS_FOLDER,
        MASKS_EXT,
        OUTPUT_LABELS_FOLDER,
        CLASS_INDEX,
    )

    rename_files(OUTPUT_IMAGES_FOLDER, PREFIX, IMAGES_EXT)
    rename_files(OUTPUT_LABELS_FOLDER, PREFIX, "")  # .txt for labels
    rename_files(OUTPUT_MASKS_FOLDER, PREFIX, MASKS_EXT)

    split_data(OUTPUT_IMAGES_FOLDER, OUTPUT_LABELS_FOLDER)

    # Create the YAML file
    create_yaml_file(
        BASE_PATH_CLEAN,
        f"{NAME_DATASET}/images/train",
        f"{NAME_DATASET}/images/val",
        f"{NAME_DATASET}/images/test",
        CLASS_INDEX,
        [""],
        BASE_PATH_YAML,
    )
