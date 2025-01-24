import os
import torch
from torchvision.io import read_image, ImageReadMode
from .manage_data import create_dir, detect_imgs

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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


# Calculate bounding box coordinates
def valRect(coord: torch.Tensor) -> list:
    """
    Calculate bounding box coordinates from a list of coordinates (x, y) of a mask object

    Args:
        coord (np.ndarray): List of coordinates (x, y) of a mask object in a binary mask with shape (n, 1, 2) where:
                            n: number of pixels in the mask object
                            1: number of channels
                            2: x and y coordinates

    Returns:
        list: Returns a list with the coordinates of the bounding box [xmin, ymin, xmax, ymax]
    """
    xmin = torch.min(coord[:, 0]) + 1
    ymin = torch.min(coord[:, 1]) + 1
    xmax = torch.max(coord[:, 0]) + 1
    ymax = torch.max(coord[:, 1]) + 1
    return [xmin.item(), ymin.item(), xmax.item(), ymax.item()]


# Convert bounding box to YOLO format
def yolo_format(class_index: int, coord: torch.Tensor, width: int, height: int) -> str:
    """
    Convert bounding box coordinates to YOLO format using the coordinates calculated from the mask object in the valRect function

    Args:
        class_index (int): Index of the class
        coord (np.ndarray): List of coordinates (x, y) of a mask object in a binary mask with shape (n, 1, 2)
        width (int): With of the all image
        height (int): Height of the all image

    Returns:
        str: Returns a string with the bounding box coordinates in YOLO format rounded to 6 decimal places
    """
    [xmin, ymin, xmax, ymax] = valRect(coord)
    x_center = (xmin + xmax) / (2.0 * width)
    y_center = (ymin + ymax) / (2.0 * height)
    x_width = (xmax - xmin) / width
    y_height = (ymax - ymin) / height
    return f"{class_index} {x_center:.6f} {y_center:.6f} {x_width:.6f} {y_height:.6f}"


# Main function to call and process images
def process_images(
    image_folder: str,
    image_ext: str,
    mask_folder: str,
    mask_ext: str,
    output_folder: str,
    class_index: int = 0,  # Only allows one class
) -> None:
    """
    Call all necessary functions to create the directory, get the coordinates, calculate the bounding box in YOLO format, and save the .txt file

    Args:
        image_folder (str): Directory path to images
        mask_folder (str): Directory path to masks
        output_folder (str): Directory path to save the .txt files
        class_index (int, optional): Index object classes. Defaults to 0.
    """
    create_dir(output_folder)  # Create output directory

    images = detect_imgs(image_folder, image_ext)  # Detect images in the folder
    masks = detect_imgs(mask_folder, mask_ext)  # Detect masks in the folder

    for img_path, mask_path in zip(
        images, masks
    ):  # Iterates and uncompress every path of images and masks
        image = read_image(img_path).to(device)  # Read image
        mask = read_image(mask_path, mode=ImageReadMode.GRAY).to(
            device
        )  # Read mask in grayscale

        h, w = image.shape[1], image.shape[2]  # Get image shape

        mask_coordinates = torch.nonzero(mask.squeeze(), as_tuple=False)[
            :, [1, 0]
        ]  # Find non-zero coordinates in the mask and reverse the order

        # If no objects found in the mask, skip the image
        if mask_coordinates.size(0) == 0:
            print(f"No objects found in mask: {mask_path}")
            continue

        # Convert mask coordinates to YOLO format
        yolo_line = yolo_format(class_index, mask_coordinates, w, h)

        # Save bounding box in YOLO format
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_txt_path = os.path.join(output_folder, base_name)
        save_bbox(output_txt_path, yolo_line)

        print(f"Processed: {img_path} -> {output_txt_path}")
