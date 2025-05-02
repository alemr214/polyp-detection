# %%
from scripts.process_images import annotate_images, draw_bounding_boxes_on_images
from scripts.manage_data import (
    copy_images,
    count_lines_in_file,
    rename_files,
    split_data,
    create_yaml_file,
    count_files,
)
from scripts.yolo_utils import (
    get_best_model,
    train_model,
    validate_model,
    export_model,
    make_predicts,
)
from scripts.evalute_datasets import evalute_predictions
import os

# %%
# Configuration base paths to manage raw and clean information from a dataset
BASE_PATH = os.getcwd()
PATH_RAW = "data/raw"
PATH_CLEAN = "data/clean"
BASE_PATH_YAML = "configs"
NAME_DATASET = "cvc_clinic_db"
BASE_PATH_MODEL = "runs"
TRAIN_PATH = "train"
PREDICT_PATH = "predict"
VALIDATE_PATH = "validate"

# Designated paths of data to process raw data
IMAGES_FOLDER = f"{PATH_RAW}/{NAME_DATASET}/PNG/Original"
MASKS_FOLDER = f"{PATH_RAW}/{NAME_DATASET}/PNG/Ground Truth"

# Output paths
OUTPUT_IMAGES_FOLDER = f"{PATH_CLEAN}/{NAME_DATASET}/images"
OUTPUT_LABELS_FOLDER = f"{PATH_CLEAN}/{NAME_DATASET}/labels"
OUTPUT_MASKS_FOLDER = f"{PATH_CLEAN}/{NAME_DATASET}/masks"
OUTPUT_BBOX_FOLDER = f"{PATH_CLEAN}/{NAME_DATASET}/bbox"

# Class index to save the annotations
CLASS_INDEX = 0

# Prefix to rename files ("images")
PREFIX_IMAGE = "image"

# %%
# Copy images and masks to the output folder
copy_images(IMAGES_FOLDER, OUTPUT_IMAGES_FOLDER)
copy_images(MASKS_FOLDER, OUTPUT_MASKS_FOLDER)

# %%
# Process images and masks to create labels
annotate_images(
    OUTPUT_IMAGES_FOLDER,
    OUTPUT_MASKS_FOLDER,
    OUTPUT_LABELS_FOLDER,
    CLASS_INDEX,
)

# %%
# Rename images, masks, and labels
rename_files(OUTPUT_IMAGES_FOLDER, PREFIX_IMAGE)
rename_files(OUTPUT_LABELS_FOLDER, PREFIX_IMAGE)
rename_files(OUTPUT_MASKS_FOLDER, PREFIX_IMAGE)

# %%
# Draw bounding boxes on images
draw_bounding_boxes_on_images(
    OUTPUT_IMAGES_FOLDER, OUTPUT_MASKS_FOLDER, OUTPUT_BBOX_FOLDER
)

# %%
# Split data into train, validation, and test
split_data(OUTPUT_IMAGES_FOLDER, OUTPUT_LABELS_FOLDER)

# %%
# Create YAML file with the dataset information
create_yaml_file(
    f"{BASE_PATH}/{PATH_CLEAN}/{NAME_DATASET}",
    "images/train",
    "images/validation",
    "images/test_single",
    1,
    ["polyp"],
    f"{BASE_PATH_YAML}/{NAME_DATASET}/",
)


# %%
# Counters
total_images = 0
total_polyps = 0
# Count images in each folder
for folder in ["train", "val", "test"]:
    print(
        f"Images {folder}: {count_files(f'{OUTPUT_IMAGES_FOLDER}/{folder}', ['.jpg', '.png', '.tif'])}"
    )
    print(f"Polyps {folder}: {count_lines_in_file(f'{OUTPUT_LABELS_FOLDER}/{folder}')}")
    total_images += count_files(
        f"{OUTPUT_IMAGES_FOLDER}/{folder}", [".jpg", ".png", ".tif"]
    )
    total_polyps += count_lines_in_file(f"{OUTPUT_LABELS_FOLDER}/{folder}")

print(f"Total images: {total_images}")
print(f"Total polyps: {total_polyps}")

# %%
# Train model
train_model(
    f"{BASE_PATH_MODEL}/{TRAIN_PATH}/{NAME_DATASET}/yolo11n.pt",
    f"{BASE_PATH_YAML}/{NAME_DATASET}/dataset.yaml",
    epoches=5,
    image_size=640,
    batch_size=-1,
    save_period=1,
    name=NAME_DATASET,
    project=f"{BASE_PATH_MODEL}/{TRAIN_PATH}",
)

# %%
# Predict model
for dataset in [
    "cvc_clinic_db",
    "cvc_colon_db",
    "etis_laribpolypdb",
    "kvasir_seg",
    "sessile_main_kvasir_seg",
]:
    make_predicts(
        f"{BASE_PATH_MODEL}/{TRAIN_PATH}/{dataset}",
        f"{PATH_CLEAN}/{dataset}/images/test",
        name=f"{dataset}",
        project=f"{BASE_PATH_MODEL}/{PREDICT_PATH}",
    )


# %%
# Validate model
results = validate_model(
    f"{BASE_PATH_MODEL}/{TRAIN_PATH}/{NAME_DATASET}",
    f"{BASE_PATH_YAML}/{NAME_DATASET}/dataset.yaml",
    name=NAME_DATASET,
    project=f"{BASE_PATH_MODEL}/{VALIDATE_PATH}",
)


# %%
# Evalute model
for database in ["polypgen"]:
    print(f"Evaluating {database} dataset")
    gt_image = f"data/clean/{database}/labels/test_sequence"
    pred_image = f"runs/predict/{database}_sequence/labels"
    evalute_predictions(gt_image, pred_image, iou_threshold=0.95)
