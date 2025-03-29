import os
import cv2
from .manage_data import detect_files, create_dir


def save_bbox(txt_path: str, line_to_write: str) -> None:
    """
    Save bounding box in a txt file with specified path

    Args:
        txt_path (str): Path to save the file
        line_to_write (str): Parameters of the bounding box to write in a file
    """
    txt_path = txt_path + ".txt"
    with open(txt_path, "w") as my_file:
        my_file.write(line_to_write + "\n")


def detect_object(mask: str, min_area: int = 35) -> list | None:
    """
    Detect objects in a binary mask and return the coordinates for every object detected in a list of tuples.

    Args:
        mask (str): Path to the binary mask image.

    Returns:
        list | None: List of tuples with the coordinates of the objects detected in the mask.
    """
    mask_image = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_image)
    objects_coordiantes = []
    for i in range(1, num_labels):
        area = stats[i][cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y, w, h = (
            stats[i][cv2.CC_STAT_LEFT],
            stats[i][cv2.CC_STAT_TOP],
            stats[i][cv2.CC_STAT_WIDTH],
            stats[i][cv2.CC_STAT_HEIGHT],
        )
        objects_coordiantes.append((x, y, x + w, y + h))
    return objects_coordiantes


def normalize_coordiantes(coordinates: list) -> list:
    """
    Normalize the coordinates to get the center, width, and height of objects detected in a mask.

    Args:
        coordinates (list): List of tuples with the coordinates of the objects detected in the mask.

    Returns:
        list: List of tuples with the normalized coordinates of the objects detected in the mask.
    """
    coordinates_normalized = []
    for coordiante in coordinates:
        x1, y1, x2, y2 = coordiante
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        coordinates_normalized.append((x_center, y_center, width, height))
    return coordinates_normalized


def yolo_format(
    class_index: int, norm_coordinates: list, image_width: int, image_height: int
) -> list:
    """
    Get the YOLO format of the coordinates of the objects detected in a mask.

    Args:
        class_index (int): Index of the class.
        norm_coordinates (list): List of tuples with the normalized coordinates of the objects detected in the mask.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        list: List of strings with the YOLO format of the coordinates of the objects detected in the mask.
    """
    yolo_format = []
    for coord in norm_coordinates:
        x_center, y_center, width, height = coord
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height
        yolo_format.append(f"{class_index} {x_center} {y_center} {width} {height}")
    return yolo_format


def annotate_images(
    images_path: str, masks_path: str, output_labels_path: str, class_index: int = 0
) -> None:
    """
    Annotate images with the bounding boxes of the objects detected in the masks and save the labels in a txt file.

    Args:
        images_path (str): Path to the images.
        masks_path (str): Path to the masks.
        output_labels_path (str): Path to save the labels.
        class_index (int, optional): Index of the class. Defaults to 0.
    """
    create_dir(output_labels_path)
    images = detect_files(images_path, [".png", ".jpg", ".tif"])
    masks = detect_files(masks_path, [".png", ".jpg", ".tif"])
    for image, mask in zip(images, masks):
        image_read = cv2.imread(image)
        objects_coordinates = detect_object(mask)
        objects_coordinates = normalize_coordiantes(objects_coordinates)
        yolo_labels = yolo_format(
            class_index,
            objects_coordinates,
            image_read.shape[1],
            image_read.shape[0],
        )
        base_name = os.path.splitext(os.path.basename(image))[0]
        output_txt_path = os.path.join(output_labels_path, base_name)
        save_bbox(output_txt_path, "\n".join(yolo_labels))


def draw_bounding_boxes_on_images(
    images_path: str,
    masks_path: str,
    output_bbox_images: str,
) -> None:
    """
    Draw bounding boxes on images using the coordinates obtained from the mask object

    Args:
        images_path (str): Path of images
        masks_path (str): Path of masks
        output_bbox_images (str): Path to save the images with bounding boxes
    """
    create_dir(output_bbox_images)

    images = detect_files(images_path, [".png", ".jpg", ".tif"])
    masks = detect_files(masks_path, [".png", ".jpg", ".tif"])
    for image_path, mask_path in zip(images, masks):
        image = cv2.imread(image_path)
        object_coordinates = detect_object(mask_path)
        for coord in object_coordinates:
            x1, y1, x2, y2 = coord
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            ext_type = os.path.splitext(os.path.basename(image_path))[1]
            output_image_path = os.path.join(output_bbox_images, base_name)
            cv2.imwrite(output_image_path + ext_type, image)
