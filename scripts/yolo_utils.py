import torch
from ultralytics import YOLO


def get_device() -> str:
    """
    Get the device to use for training the model (cuda, mps, cpu).

    Returns:
        str: string with the name of the device to use.
    """
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def train_model(
    model_path: str,
    yaml_path: str,
    epoches: int,
    image_size: int,
    batch_size: int,
    save_period: int,
    name: str,
    project: str,
) -> None:
    """
    Method to train a YOLO model.

    Args:
        model_path (str): Path to save the final model.
        yaml_path (str): Path from the yaml file with the dataset information.
        epoches (int): Number of epoches to train the model.
        image_size (int): Resize the images to this size.
        batch_size (int): Size of the batch to use for training (number of training).
        save_period (int): Period to save the model.
        name (str): Name of the model.
        project (str): Project to save the model.
    """
    model = YOLO(model_path)
    model.train(
        data=yaml_path,
        epochs=epoches,
        imgsz=image_size,
        batch=batch_size,
        save_period=save_period,
        device=get_device(),
        name=name,
        project=project,
        exist_ok=True,
    )


def get_best_model(
    model_path: str,
) -> YOLO:
    """
    Return the best model generated during the training.

    Args:
        model_path (str): Path to the model output in the trainin model method.

    Returns:
        YOLO: return a instance of YOLO class with the best model."""
    model = YOLO(f"{model_path}/weights/best.pt")
    return model


def validate_model(
    model_path: str,
    yaml_path: str,
    name: str,
    project: str,
) -> list:
    """
    Generate metrics using the best model generated during the training.

    Args:
        model_path (str): Path to the model output in the trainin model method.

    Returns:
        list: List with the metrics of the best_model.
    """
    best_model = get_best_model(model_path)
    best_model.val(data=yaml_path, name=name, project=project, device=get_device())


# REFACTOR: CHECK THE PARAMETERS IN THE EXPORT METHOD TO SEE IF IT IS POSSIBLE TO EXPORT TO OTHER FORMATS AND WHAT IS NEEDED
def export_model(model_path: str, format: str) -> None:
    """
    Export model to another format

    Args:
        model_path (str): Path to the model output in the trainin model method.
        format (str): Format to export the model.
    """
    best_model = get_best_model(model_path)
    best_model.export(
        format=format,
        opset=12,
        device=get_device(),
    )


def make_predicts(
    model_path: str,
    test_images_path: str,
    name: str,
    project: str,
) -> None:
    best_model = get_best_model(model_path)
    best_model.predict(
        test_images_path,
        save=True,
        name=name,
        project=project,
        device=get_device(),
        save_txt=True,
    )
