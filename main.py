from scripts.annotate_images import process_images
from scripts.manage_data import copy_images, rename_files, split_data, create_yaml_file


if __name__ == "__main__":
    # Configuration base paths to manage raw and clean information
    PATH_RAW = "data/raw"
    PATH_CLEAN = "data/clean"
    BASE_PATH_YAML = "configs"
    NAME_DATASET = "sessile_main_kvasir_seg"

    # Designated paths of data to process raw data
    IMAGES_FOLDER = f"{PATH_RAW}/{NAME_DATASET}/images"
    MASKS_FOLDER = f"{PATH_RAW}/{NAME_DATASET}/masks"

    # Output paths
    OUTPUT_IMAGES_FOLDER = f"{PATH_CLEAN}/{NAME_DATASET}/images"
    OUTPUT_LABELS_FOLDER = f"{PATH_CLEAN}/{NAME_DATASET}/labels"
    OUTPUT_MASKS_FOLDER = f"{PATH_CLEAN}/{NAME_DATASET}/masks"

    # Class index to save the annotations
    CLASS_INDEX = 0

    # Prefix to rename files ("images")
    PREFIX = "image"

    # Copy images and masks to the output folder
    copy_images(IMAGES_FOLDER, OUTPUT_IMAGES_FOLDER)
    copy_images(MASKS_FOLDER, OUTPUT_MASKS_FOLDER)

    # Process images and masks to create labels
    process_images(
        OUTPUT_IMAGES_FOLDER,
        MASKS_FOLDER,
        OUTPUT_LABELS_FOLDER,
        CLASS_INDEX,
    )

    # Rename images, masks, and labels
    rename_files(OUTPUT_IMAGES_FOLDER, PREFIX)
    rename_files(OUTPUT_LABELS_FOLDER, PREFIX)
    rename_files(OUTPUT_MASKS_FOLDER, PREFIX)

    # Split data into train, validation, and test
    split_data(OUTPUT_IMAGES_FOLDER, OUTPUT_LABELS_FOLDER)

    # Create YAML file with the dataset information
    create_yaml_file(
        f"{PATH_CLEAN}/{NAME_DATASET}",
        "images/train",
        "images/val",
        "images/test",
        1,
        ["polyp"],
        f"{BASE_PATH_YAML}/{NAME_DATASET}/",
    )
