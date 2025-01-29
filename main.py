from scripts.annotate_images import process_images
from scripts.manage_data import copy_images, rename_files, split_data

if __name__ == "":
    # Configuration
    BASE_PATH = ""
    NAME_DATASET = ""

    # Data paths
    IMAGES_FOLDER = ""  # To copy images from
    IMAGES_EXT = ""  # Image extension
    MASKS_FOLDER = ""  # To process masks from images
    MASKS_EXT = ""  # Mask extension
    OUTPUT_IMAGES_FOLDER = ""  # To save the copy images
    OUTPUT_LABELS_FOLDER = ""  # To save the .txt bbox annotations from masks
    OUTPUT_MASKS_FOLDER = ""
    CLASS_INDEX = 0  # Class index to save the annotations
    PREFIX = ""  # Prefix to rename files ("images")

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
    rename_files(OUTPUT_LABELS_FOLDER, PREFIX, "")  # txt for labels
    rename_files(OUTPUT_MASKS_FOLDER, PREFIX, MASKS_EXT)

    split_data(OUTPUT_IMAGES_FOLDER, OUTPUT_LABELS_FOLDER)
