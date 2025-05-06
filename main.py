# %%
# Exporting funcitons to work
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
BASE_PATH_MODEL = "runs"
TRAIN_PATH = "train"
PREDICT_PATH = "predict"
VALIDATE_PATH = "validate"

# Class index to save the annotations
CLASS_INDEX = 0

# Prefix to rename files ("image")
PREFIX_IMAGE = "image"

# %%
for dataset in [
    "cvc_clinic_db",
    "cvc_colon_db",
    "etis_laribpolypdb",
    "kvasir_seg",
    "sessile_main_kvasir_seg",
]:
    # Designated paths of data to process raw data
    IMAGES_FOLDER = f"{PATH_RAW}/{dataset}/images"
    MASKS_FOLDER = f"{PATH_RAW}/{dataset}/masks"
    # Output paths
    OUTPUT_IMAGES_FOLDER = f"{PATH_CLEAN}/{dataset}/images"
    OUTPUT_MASKS_FOLDER = f"{PATH_CLEAN}/{dataset}/masks"

    # Copy images and masks to the output folder
    copy_images(IMAGES_FOLDER, OUTPUT_IMAGES_FOLDER)
    copy_images(MASKS_FOLDER, OUTPUT_MASKS_FOLDER)

# %%
for dataset in [
    "cvc_clinic_db",
    "cvc_colon_db",
    "etis_laribpolypdb",
    "kvasir_seg",
    "sessile_main_kvasir_seg",
]:
    # Output paths
    OUTPUT_IMAGES_FOLDER = f"{PATH_CLEAN}/{dataset}/images"
    OUTPUT_LABELS_FOLDER = f"{PATH_CLEAN}/{dataset}/labels"
    OUTPUT_MASKS_FOLDER = f"{PATH_CLEAN}/{dataset}/masks"

    # Process images and masks to create labels
    annotate_images(
        OUTPUT_IMAGES_FOLDER,
        OUTPUT_MASKS_FOLDER,
        OUTPUT_LABELS_FOLDER,
        CLASS_INDEX,
    )

# %%
for dataset in [
    "cvc_clinic_db",
    "cvc_colon_db",
    "etis_laribpolypdb",
    "kvasir_seg",
    "sessile_main_kvasir_seg",
]:
    # Output paths
    OUTPUT_IMAGES_FOLDER = f"{PATH_CLEAN}/{dataset}/images"
    OUTPUT_LABELS_FOLDER = f"{PATH_CLEAN}/{dataset}/labels"
    OUTPUT_MASKS_FOLDER = f"{PATH_CLEAN}/{dataset}/masks"

    # Rename images, masks, and labels
    rename_files(OUTPUT_IMAGES_FOLDER, PREFIX_IMAGE)
    rename_files(OUTPUT_LABELS_FOLDER, PREFIX_IMAGE)
    rename_files(OUTPUT_MASKS_FOLDER, PREFIX_IMAGE)

# %%
for dataset in [
    "cvc_clinic_db",
    "cvc_colon_db",
    "etis_laribpolypdb",
    "kvasir_seg",
    "sessile_main_kvasir_seg",
]:
    # Output paths
    OUTPUT_IMAGES_FOLDER = f"{PATH_CLEAN}/{dataset}/images"
    OUTPUT_MASKS_FOLDER = f"{PATH_CLEAN}/{dataset}/masks"
    OUTPUT_BBOX_FOLDER = f"{PATH_CLEAN}/{dataset}/bbox"

    # Draw bounding boxes on images
    draw_bounding_boxes_on_images(
        OUTPUT_IMAGES_FOLDER, OUTPUT_MASKS_FOLDER, OUTPUT_BBOX_FOLDER
    )

# %%
for dataset in [
    "cvc_clinic_db",
    "cvc_colon_db",
    "etis_laribpolypdb",
    "kvasir_seg",
    "sessile_main_kvasir_seg",
]:
    # Output paths
    OUTPUT_IMAGES_FOLDER = f"{PATH_CLEAN}/{dataset}/images"
    OUTPUT_LABELS_FOLDER = f"{PATH_CLEAN}/{dataset}/labels"

    # Split data into train, validation, and test
    split_data(OUTPUT_IMAGES_FOLDER, OUTPUT_LABELS_FOLDER)

# %%
for dataset in [
    "cvc_clinic_db",
    "cvc_colon_db",
    "etis_laribpolypdb",
    "kvasir_seg",
    "sessile_main_kvasir_seg",
]:
    # Output paths
    OUTPUT_IMAGES_FOLDER = f"{PATH_CLEAN}/{dataset}/images"
    OUTPUT_LABELS_FOLDER = f"{PATH_CLEAN}/{dataset}/labels"

    # Create YAML file with the dataset information
    create_yaml_file(
        f"{BASE_PATH}/{PATH_CLEAN}/{dataset}",
        "images/train",
        "images/val",
        "images/test",
        1,
        ["polyp"],
        f"{BASE_PATH_YAML}/{dataset}/",
    )


# %%
for dataset in ["polypgen"]:
    # Output paths
    OUTPUT_IMAGES_FOLDER = f"{PATH_CLEAN}/{dataset}/images"
    OUTPUT_LABELS_FOLDER = f"{PATH_CLEAN}/{dataset}/labels"
    # Counters
    total_images = 0
    total_polyps = 0

    print(f"Dataset: {dataset.upper()}")
    # Count images in each folder
    for folder in ["train", "validation", "test_single", "test_sequence"]:
        print(
            f"Images {folder}: {count_files(f'{OUTPUT_IMAGES_FOLDER}/{folder}', ['.jpg', '.png', '.tif'])}"
        )
        print(
            f"Polyps {folder}: {count_lines_in_file(f'{OUTPUT_LABELS_FOLDER}/{folder}')}"
        )
        total_images += count_files(
            f"{OUTPUT_IMAGES_FOLDER}/{folder}", [".jpg", ".png", ".tif"]
        )
        total_polyps += count_lines_in_file(f"{OUTPUT_LABELS_FOLDER}/{folder}")

    print(f"Total images: {total_images}")
    print(f"Total polyps: {total_polyps}")

# %%
for dataset in [
    "cvc_clinic_db",
    "cvc_colon_db",
    "etis_laribpolypdb",
    "kvasir_seg",
    "sessile_main_kvasir_seg",
    "polypgen_single",
    "polypgen_sequence",
]:
    # Train model
    train_model(
        f"{BASE_PATH_MODEL}/{TRAIN_PATH}/{dataset}/yolo11n.pt",
        f"{BASE_PATH_YAML}/{dataset}/dataset.yaml",
        epoches=1000,
        image_size=640,
        batch_size=16,
        save_period=100,
        name=f"{dataset}",
        project=f"{BASE_PATH_MODEL}/{TRAIN_PATH}",
    )

# %%
for dataset in [
    "cvc_clinic_db",
    "cvc_colon_db",
    "etis_laribpolypdb",
    "kvasir_seg",
    "sessile_main_kvasir_seg",
    "polypgen_single",
    "polypgen_sequence",
]:
    # Export model
    export_model(f"{BASE_PATH_MODEL}/{TRAIN_PATH}/{dataset}", "onnx")
    export_model(f"{BASE_PATH_MODEL}/{TRAIN_PATH}/{dataset}", "coreml")

# %%
for dataset in [
    "cvc_clinic_db",
    "cvc_colon_db",
    "etis_laribpolypdb",
    "kvasir_seg",
    "sessile_main_kvasir_seg",
    "polypgen_single",
    "polypgen_sequence",
]:
    # Predict model
    make_predicts(
        f"{BASE_PATH_MODEL}/{TRAIN_PATH}/{dataset}",
        f"{PATH_CLEAN}/{dataset}/images/test_single"
        if dataset == "polypgen_single"
        else f"{PATH_CLEAN}/{dataset}/images/test_sequence"
        if dataset == "polypgen_sequence"
        else f"{PATH_CLEAN}/{dataset}/images/test",
        name=f"{dataset}",
        project=f"{BASE_PATH_MODEL}/{PREDICT_PATH}",
    )


# %%
for dataset in [
    "cvc_clinic_db",
    "cvc_colon_db",
    "etis_laribpolypdb",
    "kvasir_seg",
    "sessile_main_kvasir_seg",
    "polypgen_single",
    "polypgen_sequence",
]:
    # Validate model
    results = validate_model(
        f"{BASE_PATH_MODEL}/{TRAIN_PATH}/{dataset}",
        f"{BASE_PATH_YAML}/{dataset}/dataset.yaml",
        name=f"{dataset}",
        project=f"{BASE_PATH_MODEL}/{VALIDATE_PATH}",
    )


# %%
for dataset in [
    "cvc_clinic_db",
    "cvc_colon_db",
    "etis_laribpolypdb",
    "kvasir_seg",
    "sessile_main_kvasir_seg",
    "polypgen_single",
    "polypgen_sequence",
]:
    # Evalute model
    print(f"Evaluating {dataset} dataset")
    gt_image = (
        f"data/clean/{dataset}/labels/test_single"
        if dataset == "polypgen_single"
        else f"data/clean/{dataset}/labels/test_sequence"
        if dataset == "polypgen_sequence"
        else f"data/clean/{dataset}/labels/test"
    )
    pred_image = f"runs/predict/{dataset}/labels"
    evalute_predictions(gt_image, pred_image, iou_threshold=0.25)

# %%
